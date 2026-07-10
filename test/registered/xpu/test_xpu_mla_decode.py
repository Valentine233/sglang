"""Backend-level XPU MLA (DeepSeek) absorbed-decode tests.

Unlike a raw-kernel test, this drives the real
``sglang.srt.layers.attention.xpu_backend.XPUAttentionBackend.forward_decode``
MLA branch end-to-end so the backend-specific changes are actually exercised:

* the query packing ``[q_nope(512), q_pe(64)] = 576`` (``q_rope=None`` path),
* the KV write through ``MLATokenToKVPool.set_mla_kv_buffer`` (nope + rope),
* the paged ``flash_attn_with_kvcache`` call with ``v_cache`` taken as the
  leading ``v_head_dim(512)`` slice of the 576-wide latent+rope cache,
* the ``page_table`` / ``cache_seqlens`` metadata built by
  ``init_forward_metadata`` from ``req_to_token`` + ``seq_lens``.

The XPU MLA decode kernel only dispatches ``page_size`` in {64, 128}; both are
covered, including ``page_size=128`` with ``nheads_q=8`` which previously hung.

A dense fp32 PyTorch reference (576-wide QK score, 512-wide value output)
validates numerics with a bf16 tolerance of 2e-2.

Usage:
    python3 -m pytest test_xpu_mla_decode.py
"""

import math
import sys

import pytest
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
device = torch.device("xpu") if _XPU_AVAILABLE else torch.device("cpu")

# DeepSeek MLA latent dims (absorbed decode).
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 (packed QK)
V_HEAD_DIM = KV_LORA_RANK  # 512 (value output width)
CONTEXT_LEN = 2048
_TOL = 2e-2


class _MockModelRunner:
    """Minimal ModelRunner stand-in that satisfies ``XPUAttentionBackend``."""

    def __init__(self, batch_size, page_size, num_attention_heads):
        self.device = str(device)
        self.tp_size = 1
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.page_size = page_size

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": CONTEXT_LEN,
                "attention_arch": AttentionArch.MLA,
                "is_encoder_decoder": False,
                "is_local_attention_model": False,
                "hf_text_config": type(
                    "HFTextConfig",
                    (),
                    {"num_attention_heads": num_attention_heads},
                ),
            },
        )
        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
            },
        )
        self.kv_cache_dtype = torch.bfloat16

        # req_to_token holds *block ids* (the XPU backend passes req_to_token
        # straight through as the paged page_table).
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": batch_size,
                "req_to_token": torch.zeros(
                    batch_size, CONTEXT_LEN, dtype=torch.int32, device=device
                ),
            },
        )

        # Give the pool room for the worst-case padded length used by the tests.
        max_total_num_tokens = batch_size * CONTEXT_LEN
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )


def _mla_decode_reference(q_packed, kv_full, softmax_scale):
    """Dense fp32 reference for absorbed MLA decode.

    Arguments:
        q_packed: (batch, nheads, 576) query, [q_nope(512), q_pe(64)]
        kv_full: (batch, seqlen_k, 576) latent+rope KV rows (one MQA head)
        softmax_scale: float
    Returns:
        (batch, nheads, 512) fp32 output
    """
    qf = q_packed.float()
    kf = kv_full.float()
    scores = torch.einsum("bhd,bsd->bhs", qf * softmax_scale, kf)
    attn = torch.softmax(scores, dim=-1)
    vf = kf[..., :V_HEAD_DIM]
    return torch.einsum("bhs,bsd->bhd", attn, vf)


def _slot(b, t, seqlen_k_padded):
    """Physical KV slot for token ``t`` of sequence ``b``."""
    return b * seqlen_k_padded + t


@pytest.mark.skipif(not _XPU_AVAILABLE, reason="requires an XPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("nheads_q", [8, 4, 1])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("seqlen_k", [128, 500])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_xpu_mla_decode_backend(batch_size, seqlen_k, page_size, nheads_q, dtype):
    torch.manual_seed(0)

    nb = math.ceil(seqlen_k / page_size)  # blocks per sequence
    seqlen_k_padded = nb * page_size
    scaling = HEAD_DIM**-0.5

    runner = _MockModelRunner(batch_size, page_size, nheads_q)
    backend = XPUAttentionBackend(runner)

    layer = RadixAttention(
        num_heads=nheads_q,
        head_dim=HEAD_DIM,
        scaling=scaling,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=V_HEAD_DIM,
        prefix="attn_mqa",
    )

    # req_to_token holds *token-granular* physical KV slots; the XPU backend
    # itself strides by page_size and divides to derive the block page_table
    # (see init_forward_metadata: page_table[:, ::page_size] // page_size).
    req_to_token = runner.req_to_token_pool.req_to_token
    for b in range(batch_size):
        req_to_token[b, :seqlen_k] = b * seqlen_k_padded + torch.arange(
            seqlen_k, device=device
        )

    # Full KV sequence (bf16); split into prefix (pre-written) + current token.
    all_k = torch.randn(batch_size, seqlen_k, HEAD_DIM, device=device, dtype=dtype)
    prefix_len = seqlen_k - 1

    # Pre-write prefix tokens 0..seqlen_k-2 directly into the pool.
    if prefix_len > 0:
        prefix_slots = torch.cat(
            [
                _slot(b, torch.arange(prefix_len, device=device), seqlen_k_padded)
                for b in range(batch_size)
            ]
        )
        prefix_nope = all_k[:, :prefix_len, :KV_LORA_RANK].reshape(
            batch_size * prefix_len, 1, KV_LORA_RANK
        )
        prefix_rope = all_k[:, :prefix_len, KV_LORA_RANK:].reshape(
            batch_size * prefix_len, 1, QK_ROPE_HEAD_DIM
        )
        runner.token_to_kv_pool.set_mla_kv_buffer(
            layer, prefix_slots, prefix_nope, prefix_rope
        )

    # Current (being-decoded) token — written by forward_decode itself.
    cur = all_k[:, prefix_len, :]  # (batch, 576)
    k = cur[:, :KV_LORA_RANK].unsqueeze(1).contiguous()  # (batch, 1, 512)
    k_rope = cur[:, KV_LORA_RANK:].unsqueeze(1).contiguous()  # (batch, 1, 64)
    v = torch.randn(1, device=device, dtype=dtype)  # unused by MLA
    out_cache_loc = torch.tensor(
        [_slot(b, prefix_len, seqlen_k_padded) for b in range(batch_size)],
        device=device,
    )

    # Packed query (q_rope=None path): pass the full 576-wide query directly.
    q = torch.randn(batch_size, nheads_q, HEAD_DIM, device=device, dtype=dtype)

    forward_batch = ForwardBatch(
        batch_size=batch_size,
        input_ids=torch.randint(0, 100, (batch_size, 1), device=device),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=batch_size * seqlen_k,
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.arange(batch_size, device=device),
        seq_lens=torch.full((batch_size,), seqlen_k, device=device),
        seq_lens_cpu=torch.full((batch_size,), seqlen_k, device="cpu"),
    )

    backend.init_forward_metadata(forward_batch)
    out = backend.forward_decode(q, k, v, layer, forward_batch, k_rope=k_rope)

    assert out.shape == (batch_size, nheads_q * V_HEAD_DIM)
    assert out.dtype == dtype
    assert torch.isnan(out).sum().item() == 0

    out = out.view(batch_size, nheads_q, V_HEAD_DIM).float()
    ref = _mla_decode_reference(q, all_k, scaling)
    max_diff = (out - ref).abs().max().item()
    assert max_diff <= _TOL, (
        f"MLA decode mismatch: batch={batch_size} seqlen_k={seqlen_k} "
        f"page_size={page_size} nheads_q={nheads_q} max_diff={max_diff:.4e}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

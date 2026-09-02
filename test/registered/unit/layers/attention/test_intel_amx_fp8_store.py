import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIntelAMXFP8DecodeStore(CustomTestCase):
    def _make_backend(self, *, is_swa_layer=None):
        backend = object.__new__(IntelAMXAttnBackend)
        backend.draft_decode_metadata = None
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8, dtype=torch.int32).unsqueeze(0)
        )
        backend.v_head_dim = 4
        backend.num_head = 1
        backend.forward_metadata = (
            torch.empty((1, 1, 8, 5), dtype=torch.float32),
            None,
        )
        backend._attn_logits_buffers = {}

        key_buffer = torch.zeros((8, 1, 4), dtype=torch.float8_e4m3fn)
        value_buffer = torch.zeros_like(key_buffer)
        pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: key_buffer,
            get_value_buffer=lambda _layer_id: value_buffer,
        )
        backend.use_sliding_window_kv_pool = is_swa_layer is not None
        if is_swa_layer is not None:
            pool.layers_mapping = {0: (0, is_swa_layer)}
        backend.token_to_kv_pool = pool
        backend.swa_out_cache_loc = torch.tensor([6], dtype=torch.int64)

        calls = []

        def decode_attention(*args):
            calls.append(args)

        backend.decode_attention_fwd = decode_attention
        return backend, calls

    @staticmethod
    def _inputs(*, is_cross_attention=False):
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            qk_head_dim=4,
            v_head_dim=4,
            k_scale_float=0.5,
            v_scale_float=0.25,
            scaling=0.5,
            logit_cap=0.0,
            is_cross_attention=is_cross_attention,
            sliding_window_size=-1,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([1], dtype=torch.int64),
            out_cache_loc=torch.tensor([2], dtype=torch.int64),
            encoder_out_cache_loc=torch.tensor([3], dtype=torch.int64),
            encoder_lens=None,
        )
        q = torch.randn((1, 4), dtype=torch.bfloat16)
        k = torch.randn((1, 1, 4), dtype=torch.bfloat16)
        v = torch.randn((1, 1, 4), dtype=torch.bfloat16)
        return q, k, v, layer, forward_batch

    def test_decode_forwards_kv_only_when_cache_save_is_enabled(self):
        q, k, v, layer, forward_batch = self._inputs()

        backend, calls = self._make_backend()
        backend.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=True)
        self.assertIs(calls[0][6], k)
        self.assertIs(calls[0][7], v)

        backend, calls = self._make_backend()
        backend.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=False)
        self.assertIsNone(calls[0][6])
        self.assertIsNone(calls[0][7])

    def test_decode_selects_full_or_swa_physical_location_by_layer(self):
        q, k, v, layer, forward_batch = self._inputs()

        full_backend, full_calls = self._make_backend(is_swa_layer=False)
        full_backend.forward_decode(q, k, v, layer, forward_batch)
        self.assertIs(full_calls[0][8], forward_batch.out_cache_loc)

        swa_backend, swa_calls = self._make_backend(is_swa_layer=True)
        swa_backend.forward_decode(q, k, v, layer, forward_batch)
        self.assertIs(swa_calls[0][8], swa_backend.swa_out_cache_loc)

    def test_cross_attention_retains_encoder_cache_location(self):
        q, k, v, layer, forward_batch = self._inputs(is_cross_attention=True)
        backend, calls = self._make_backend(is_swa_layer=True)

        backend.forward_decode(q, k, v, layer, forward_batch)

        self.assertIs(calls[0][8], forward_batch.encoder_out_cache_loc)


if __name__ == "__main__":
    unittest.main()

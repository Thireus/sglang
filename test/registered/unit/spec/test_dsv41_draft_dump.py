"""The measurement-only dump: it writes prefill rows only, stops at its cap, and needs its flag."""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_pp_dump import (
    DraftHiddenDump,
    install_draft_dump,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

WIDE = 15


class _Inner:
    def forward(self, input_ids, positions, forward_batch, *_):
        rows = input_ids.numel()
        return (torch.zeros(rows, 4), torch.zeros(rows, 4)), positions.float().repeat(
            WIDE, 1
        ).t()


class _CausalLM:
    def __init__(self):
        self.model = _Inner()
        self.layers = None

    def set_dspark_layers_to_capture(self, layers):
        self.layers = layers


def _batch(mode, lens):
    return SimpleNamespace(
        forward_mode=mode,
        rids=[f"r{i}" for i in range(len(lens))],
        extend_seq_lens_cpu=lens,
    )


class TestDraftHiddenDump(CustomTestCase):
    def test_prefill_rows_are_written_and_decode_rows_are_not(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _CausalLM()
            dump = install_draft_dump(model, folder, [37, 38, 39])
            self.assertEqual(model.layers, [37, 38, 39])
            ids = torch.arange(5, dtype=torch.int64)
            positions = torch.arange(100, 105, dtype=torch.int64)
            out = model.model.forward(ids, positions, _batch(ForwardMode.EXTEND, [5]))
            self.assertEqual(out[1].shape, (5, WIDE))
            model.model.forward(ids[:1], positions[:1], _batch(ForwardMode.DECODE, []))
            self.assertEqual((dump.seq, dump.rows), (1, 5))
            saved = torch.load(f"{folder}/forward-0000000.pt", weights_only=True)
            self.assertEqual(saved["rids"], ["r0"])
            self.assertEqual(saved["extend_seq_lens"], [5])
            self.assertEqual(saved["input_ids"].tolist(), list(range(5)))
            self.assertEqual(saved["positions"].tolist(), list(range(100, 105)))
            self.assertEqual(saved["layer_ids"], [37, 38, 39])
            self.assertEqual(saved["hidden"].dtype, torch.bfloat16)
            self.assertEqual(
                saved["hidden"][:, 0].float().tolist(),
                [100.0, 101.0, 102.0, 103.0, 104.0],
            )

    def test_the_cap_stops_it_and_the_forward_still_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _CausalLM()
            dump = install_draft_dump(model, folder, [37])
            dump.max_bytes = 10 * WIDE * 2
            ids = torch.arange(6, dtype=torch.int64)
            batch = _batch(ForwardMode.EXTEND, [6])
            model.model.forward(ids, ids, batch)
            out = model.model.forward(ids, ids, batch)
            self.assertEqual(out[1].shape, (6, WIDE))
            self.assertTrue(dump.full)
            self.assertEqual(dump.seq, 1)

    def test_no_partial_file_survives_a_write(self):
        with tempfile.TemporaryDirectory() as folder:
            dump = DraftHiddenDump(folder, [37])
            ids = torch.arange(2, dtype=torch.int64)
            dump.record(_batch(ForwardMode.EXTEND, [2]), ids, ids, torch.zeros(2, WIDE))
            import os

            self.assertEqual(os.listdir(folder), ["forward-0000000.pt"])

    def test_off_by_default(self):
        self.assertFalse(envs.SGLANG_DSV41_DRAFT_DUMP_DIR.get())


if __name__ == "__main__":
    unittest.main()

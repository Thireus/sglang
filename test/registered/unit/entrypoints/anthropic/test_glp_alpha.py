"""Per-request GLP steering alpha: the request field, its cache namespace and
the per-token row vector a forward carries.

The edit itself lives in the Weightless plugin; these tests pin the routing
half: a valid alpha (0 to 4.00, at most two decimals) reads back as its
two-decimal value and one with more decimals is refused, never rounded; an
invalid one is refused with a client-facing message that never echoes a
non-number; alpha 0 keeps the empty namespace (today's cache) and every other
value one namespace per two-decimal value; a client's custom_params may not
carry the key; rows follow ``batch.reqs`` for decode (beam member rows at
alpha 0), extend (including a mixed batch) and target verify; a mixed batch
reaches the device without a synchronizing copy; and the engine refuses the
flag at boot, at its call site too, where its forwards carry no rows.
"""

import math
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
from pydantic import ValidationError

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt import glp_alpha  # noqa: E402
from sglang.srt.batch_overlap.two_batch_overlap import (  # noqa: E402
    TboForwardBatchPreparer,
)
from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.glp_alpha import (  # noqa: E402
    GLP_ALPHA_MAX,
    GLP_ALPHA_PARAM,
    alpha_namespace,
    build_alpha_rows,
    client_custom_params_refusal,
    client_sampling_params_refusal,
    forward_alpha_rows,
    read_request_alpha,
    refusal_at_boot,
    refusal_for_nonzero_alpha,
    request_token_rows,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import (  # noqa: E402
    GraphSlot,
    PaddingPolicy,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.model_executor.runner import eager_runner  # noqa: E402
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (  # noqa: E402
    DecodeCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.speculative.spec_registry import CustomSpecAlgo  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CPU = torch.device("cpu")
_HUGE = 10**400


class TestReadRequestAlpha(unittest.TestCase):
    def test_absent_and_other_keys_read_zero(self):
        for params in (None, {}, {"other": 1}, {GLP_ALPHA_PARAM: None}):
            self.assertEqual(read_request_alpha(params), (0.0, None))

    def test_a_custom_params_that_is_not_a_dict_reads_zero(self):
        # Every other reader of custom_params ignores a non-dict; so does this.
        for params in ([1], [GLP_ALPHA_PARAM], "x", 5, True, 2.0):
            self.assertEqual(read_request_alpha(params), (0.0, None), params)

    def test_values_round_trip(self):
        # The floats nearest two-decimal values, 0.07 * 100 not being 7.0 among
        # them, read back unchanged.
        for value in (0, 0.0, 2, 2.0, 3.5, GLP_ALPHA_MAX, 4, 0.07, 2.25, 3.1, 0.01):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertIsNone(error)
            self.assertEqual(alpha, float(value))
            self.assertIs(type(alpha), float)

    def test_every_two_decimal_value_is_accepted_as_itself(self):
        for hundredths in range(0, 401):
            text = f"{hundredths // 100}.{hundredths % 100:02d}"
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: float(text)})
            self.assertIsNone(error, text)
            self.assertEqual(alpha, float(text))
            self.assertEqual(f"{alpha:.2f}", text)

    def test_more_than_two_decimals_is_refused_not_rounded(self):
        # A third decimal, and a digit as far out as the twelfth.
        for value in (2.255, 0.001, 3.999, 0.005, 2.2500001, 2.25 + 1e-12):
            with self.subTest(value=value):
                alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
                self.assertEqual(alpha, 0.0)
                self.assertIn("more than two decimals", error)
                self.assertIn("not rounded", error)
                self.assertIn(repr(value), error)

    def test_float_noise_reads_as_its_two_decimal_value(self):
        # A client's own float arithmetic, not a third decimal.
        for value, expected in ((0.1 + 0.2, 0.3), (2.2500000000000004, 2.25)):
            self.assertNotEqual(value, expected)
            self.assertEqual(
                read_request_alpha({GLP_ALPHA_PARAM: value}), (expected, None)
            )

    def test_negative_zero_is_a_plain_zero(self):
        alpha, error = read_request_alpha({GLP_ALPHA_PARAM: -0.0})
        self.assertIsNone(error)
        self.assertEqual(math.copysign(1.0, alpha), 1.0)

    def test_non_numbers_are_refused_by_type(self):
        for value in (True, False, "2.0", [2.0], {"a": 1}, "secret text"):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertEqual(alpha, 0.0)
            self.assertIn(f"got {type(value).__name__}", error)
            # The client's value never reaches the message (it is logged).
            if isinstance(value, str):
                self.assertNotIn(value, error)

    def test_an_int_past_the_float_range_is_refused_not_raised(self):
        for value in (_HUGE, -_HUGE):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertEqual(alpha, 0.0)
            self.assertIn("beyond the float range", error)
            self.assertLess(len(error), 120)

    def test_non_finite_and_out_of_range_are_refused(self):
        for value in (math.nan, math.inf, -math.inf):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertEqual(alpha, 0.0)
            self.assertIn("must be finite", error)
        for value in (-0.5, -1, -0.01, GLP_ALPHA_MAX + 0.5, 4.01, 5, 1e300):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertEqual(alpha, 0.0)
            self.assertIn("is outside [0, 4.00]", error)


class TestNamespace(unittest.TestCase):
    def test_zero_keeps_today_namespace(self):
        self.assertEqual(alpha_namespace(0.0), "")
        self.assertEqual(alpha_namespace(-0.0), "")

    def test_nonzero_is_salted_and_two_decimal(self):
        with mock.patch.object(glp_alpha, "_NAMESPACE_SALT", "0123abcd"):
            self.assertEqual(alpha_namespace(2.0), "|glp:0123abcd:2.00")
            self.assertEqual(alpha_namespace(3.5), "|glp:0123abcd:3.50")
            self.assertEqual(alpha_namespace(2.25), "|glp:0123abcd:2.25")
            self.assertEqual(alpha_namespace(0.07), "|glp:0123abcd:0.07")
        with mock.patch.object(glp_alpha, "_NAMESPACE_SALT", "feedbeef"):
            self.assertEqual(alpha_namespace(2.0), "|glp:feedbeef:2.00")

    def test_one_namespace_per_two_decimal_value(self):
        def namespace(value):
            alpha, error = read_request_alpha({GLP_ALPHA_PARAM: value})
            self.assertIsNone(error)
            return alpha_namespace(alpha)

        # A JSON 2, 2.0 or 2.00 reads as the int 2 or the float 2.0.
        self.assertEqual(namespace(2), namespace(2.0))
        self.assertEqual(namespace(0.1 + 0.2), namespace(0.3))
        self.assertNotEqual(namespace(2.25), namespace(2.0))
        self.assertNotEqual(namespace(2.25), namespace(2.26))
        names = {namespace(h / 100) for h in range(1, 401)}
        self.assertEqual(len(names), 400)

    def test_the_process_salt_is_unguessable_and_shared(self):
        # A per-process salt keeps the namespace out of reach of a caller's own
        # extra_key; two requests at the same strength still share it.
        salt = glp_alpha._NAMESPACE_SALT
        self.assertRegex(salt, r"^[0-9a-f]{8}$")
        self.assertEqual(alpha_namespace(2.0), f"|glp:{salt}:2.00")
        self.assertNotEqual(alpha_namespace(2.0), "|glp:2.00")
        self.assertEqual(
            alpha_namespace(2.0),
            alpha_namespace(read_request_alpha({GLP_ALPHA_PARAM: 2})[0]),
        )


class TestClientCustomParams(unittest.TestCase):
    """A client's custom_params may not carry glp_alpha: the field belongs to
    the Messages body; every other key is the route's own."""

    def test_the_key_is_refused_whatever_its_value(self):
        for value in (2.0, 0, None, "x"):
            with self.subTest(value=value):
                message = client_custom_params_refusal({GLP_ALPHA_PARAM: value})
                self.assertIn("accepted on the Messages route only", message)
                self.assertIn("/v1/messages", message)

    def test_other_keys_and_shapes_pass(self):
        for params in (None, {}, {"thinking_budget": 8}, [GLP_ALPHA_PARAM], "x", 5):
            with self.subTest(params=params):
                self.assertIsNone(client_custom_params_refusal(params))

    def test_native_sampling_params_one_or_one_per_prompt(self):
        steered = {"custom_params": {GLP_ALPHA_PARAM: 2.0}}
        for sampling_params in (steered, [{}, steered], [None, steered]):
            with self.subTest(sampling_params=sampling_params):
                self.assertIn(
                    "Messages route only",
                    client_sampling_params_refusal(sampling_params),
                )
        for sampling_params in (None, {}, [{}, {"custom_params": {"k": 1}}], "x"):
            with self.subTest(sampling_params=sampling_params):
                self.assertIsNone(client_sampling_params_refusal(sampling_params))


class TestTokenRows(unittest.TestCase):
    def _rows(self, mode, num_reqs, extend_lens=None, tokens_per_req=1):
        return request_token_rows(
            forward_mode=mode,
            num_reqs=num_reqs,
            extend_lens=extend_lens,
            tokens_per_req=tokens_per_req,
        )

    def test_decode_one_row_per_request(self):
        self.assertEqual(self._rows(ForwardMode.DECODE, 3, tokens_per_req=4), [1] * 3)

    def test_extend_uses_each_requests_row_count(self):
        # A mixed chunked-prefill batch: a 4-token chunk and two decode rows,
        # exactly how ScheduleBatch.extend_lens spells it.
        self.assertEqual(self._rows(ForwardMode.EXTEND, 3, [4, 1, 1]), [4, 1, 1])
        self.assertEqual(self._rows(ForwardMode.MIXED, 2, [2000, 1]), [2000, 1])
        # extend_lens that do not cover the requests: not wired.
        self.assertIsNone(self._rows(ForwardMode.EXTEND, 3, [4, 1]))
        self.assertIsNone(self._rows(ForwardMode.EXTEND, 1, None))

    def test_verify_repeats_the_verify_width(self):
        self.assertEqual(
            self._rows(ForwardMode.TARGET_VERIFY, 2, tokens_per_req=6), [6, 6]
        )
        self.assertEqual(
            self._rows(ForwardMode.TARGET_VERIFY, 2, tokens_per_req=None), [1, 1]
        )

    def test_idle_and_unwired_modes(self):
        # Idle and an empty batch carry an empty plan; a mode the engine does
        # not wire carries None.
        self.assertEqual(self._rows(ForwardMode.IDLE, 1), [])
        self.assertEqual(self._rows(ForwardMode.DECODE, 0), [])
        self.assertIsNone(self._rows(ForwardMode.DRAFT_EXTEND_V2, 2))
        self.assertIsNone(self._rows(ForwardMode.DLLM_EXTEND, 1, [4]))


class TestBuildRows(unittest.TestCase):
    def _build(self, alphas, rows_per_req, tail_rows=0):
        return build_alpha_rows(
            alphas=alphas, rows_per_req=rows_per_req, device=_CPU, tail_rows=tail_rows
        )

    def test_rows_follow_requests(self):
        rows, steered = self._build([0.0, 2.0], [3, 1])
        self.assertTrue(steered)
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(rows.tolist(), [0.0, 0.0, 0.0, 2.0])
        rows, steered = self._build([2.0, 0.0, 3.5], [2, 2, 2])
        self.assertTrue(steered)
        self.assertEqual(rows.tolist(), [2.0, 2.0, 0.0, 0.0, 3.5, 3.5])

    def test_one_alpha_for_the_batch(self):
        rows, steered = self._build([2.0, 2.0, 2.0], [1, 1, 1])
        self.assertTrue(steered)
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(rows.tolist(), [2.0, 2.0, 2.0])

    def test_all_zero_is_unsteered_but_sized(self):
        rows, steered = self._build([0.0, 0.0], [2, 1])
        self.assertFalse(steered)
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(rows.tolist(), [0.0, 0.0, 0.0])

    def test_tail_rows_run_at_alpha_zero(self):
        # Beam member rows after the request rows: alpha 0, steered or not.
        rows, steered = self._build([0.0, 0.0], [1, 1], tail_rows=3)
        self.assertFalse(steered)
        self.assertEqual(rows.tolist(), [0.0] * 5)
        rows, steered = self._build([2.0, 2.0], [1, 1], tail_rows=2)
        self.assertTrue(steered)
        self.assertEqual(rows.tolist(), [2.0, 2.0, 0.0, 0.0])

    def test_idle_forward_gets_an_empty_vector(self):
        # Not None: DP padding extends it with zeros, so a padded replay
        # never runs on the rows of an earlier batch.
        rows, steered = self._build([], [])
        self.assertFalse(steered)
        self.assertEqual(tuple(rows.shape), (0,))
        self.assertEqual(rows.dtype, torch.float32)

    def test_steered_request_with_no_rows_is_a_bug(self):
        for alphas, rows_per_req in (([2.0], [0]), ([0.0, 2.0], [3, 0])):
            with self.assertRaisesRegex(RuntimeError, "contributed no token rows"):
                self._build(alphas, rows_per_req)
        # An unsteered request without rows is harmless.
        rows, _ = self._build([0.0, 2.0], [0, 2])
        self.assertEqual(rows.tolist(), [2.0, 2.0])

    def test_unwired_mode_with_a_steered_request_is_a_bug(self):
        with self.assertRaisesRegex(RuntimeError, "not wired"):
            self._build([2.0], None)
        rows, steered = self._build([0.0], None)
        self.assertIsNone(rows)
        self.assertFalse(steered)


class TestMixedRowsAreSyncFree(unittest.TestCase):
    """A mixed batch reaches the device as one small pinned copy of the
    per-request values, expanded there: no pageable copy and nothing read
    back from the device."""

    def _build(self, alphas, rows_per_req, tail_rows=0):
        staged, repeats = [], []
        real_repeat_interleave = torch.repeat_interleave

        def async_h2d(args, *, dtype, device):
            staged.append((list(args), dtype, device))
            return torch.tensor(args, dtype=dtype)

        def repeat_interleave(*args, **kwargs):
            repeats.append(kwargs)
            return real_repeat_interleave(*args, **kwargs)

        def read_back(*_args, **_kwargs):
            raise AssertionError("the builder read a device value back")

        with (
            mock.patch("sglang.srt.utils.device.async_h2d", async_h2d),
            mock.patch.object(torch, "repeat_interleave", repeat_interleave),
            mock.patch.object(torch.Tensor, "item", read_back),
            mock.patch.object(torch.Tensor, "tolist", read_back),
            mock.patch.object(torch.Tensor, "cpu", read_back),
            mock.patch.object(torch.Tensor, "numpy", read_back),
            mock.patch.object(torch.Tensor, "to", read_back),
        ):
            rows, steered = build_alpha_rows(
                alphas=alphas,
                rows_per_req=rows_per_req,
                device=_CPU,
                tail_rows=tail_rows,
            )
        return rows, steered, staged, repeats

    def test_decode_and_verify_stage_the_per_request_values_only(self):
        rows, steered, staged, repeats = self._build([0.0, 2.0, 0.0], [4, 4, 4])
        self.assertTrue(steered)
        self.assertEqual(staged, [([0.0, 2.0, 0.0], torch.float32, _CPU)])
        self.assertEqual(repeats, [])
        self.assertEqual(rows.tolist(), [0.0] * 4 + [2.0] * 4 + [0.0] * 4)

    def test_extend_expands_on_the_device_with_a_known_size(self):
        rows, steered, staged, repeats = self._build([0.0, 2.0], [5, 1])
        self.assertTrue(steered)
        self.assertEqual(
            staged,
            [([0.0, 2.0], torch.float32, _CPU), ([5, 1], torch.int64, _CPU)],
        )
        # output_size given: repeat_interleave never reads the total back.
        self.assertEqual(repeats, [{"output_size": 6}])
        self.assertEqual(rows.tolist(), [0.0] * 5 + [2.0])

    def test_one_alpha_and_zeros_copy_nothing(self):
        for alphas, steered_expected in (([2.0, 2.0], True), ([0.0, 0.0], False)):
            rows, steered, staged, repeats = self._build(alphas, [3, 1])
            self.assertIs(steered, steered_expected)
            self.assertEqual((staged, repeats), ([], []))

    def test_beam_tail_is_zeros_on_the_device(self):
        rows, steered, staged, _ = self._build([2.0, 2.0], [1, 1], tail_rows=2)
        self.assertTrue(steered)
        self.assertEqual(staged, [([2.0, 2.0], torch.float32, _CPU)])
        self.assertEqual(rows.tolist(), [2.0, 2.0, 0.0, 0.0])


def _schedule_batch(
    mode, alphas, *, extend_lens=None, spec_info=None, beam=0, tokens=None
):
    reqs = [SimpleNamespace(glp_alpha=a) for a in alphas]
    return SimpleNamespace(
        forward_mode=mode,
        reqs=reqs,
        extend_lens=extend_lens,
        spec_info=spec_info,
        beam_tail=SimpleNamespace(num_base_rows=len(reqs)) if beam else None,
        seq_lens=torch.ones(len(reqs) + beam, dtype=torch.int64),
        input_ids=None if tokens is None else torch.zeros(tokens, dtype=torch.int64),
    )


class TestForwardRows(unittest.TestCase):
    def _rows(self, batch):
        rows, steered = forward_alpha_rows(batch, device=_CPU)
        return (None if rows is None else rows.tolist()), steered

    def test_decode_beam_members_follow_the_requests_at_zero(self):
        batch = _schedule_batch(ForwardMode.DECODE, [0.0, 2.0], beam=3, tokens=5)
        self.assertEqual(self._rows(batch), ([0.0, 2.0, 0.0, 0.0, 0.0], True))
        batch = _schedule_batch(ForwardMode.DECODE, [0.0, 0.0], beam=2, tokens=4)
        self.assertEqual(self._rows(batch), ([0.0] * 4, False))

    def test_verify_uses_the_verify_width(self):
        spec = SimpleNamespace(num_tokens_per_req=3)
        batch = _schedule_batch(
            ForwardMode.TARGET_VERIFY, [2.0, 0.0], spec_info=spec, tokens=6
        )
        self.assertEqual(self._rows(batch), ([2.0] * 3 + [0.0] * 3, True))
        # A spec input that declares no per-request width falls back to its
        # verify tree's draft_token_num.
        spec = SimpleNamespace(num_tokens_per_req=0, draft_token_num=2)
        batch = _schedule_batch(ForwardMode.TARGET_VERIFY, [2.0], spec_info=spec)
        self.assertEqual(self._rows(batch), ([2.0, 2.0], True))

    def test_extend_uses_the_extend_lens(self):
        batch = _schedule_batch(
            ForwardMode.EXTEND, [0.0, 2.0], extend_lens=[3, 1], tokens=4
        )
        self.assertEqual(self._rows(batch), ([0.0, 0.0, 0.0, 2.0], True))

    def test_a_batch_without_input_ids_still_gets_rows(self):
        # A decode graph that loads its own inputs leaves input_ids None.
        batch = _schedule_batch(ForwardMode.DECODE, [2.0])
        self.assertEqual(self._rows(batch), ([2.0], True))

    def test_idle_gets_an_empty_vector(self):
        batch = _schedule_batch(ForwardMode.IDLE, [], tokens=0)
        self.assertEqual(self._rows(batch), ([], False))

    def test_a_mis_sized_vector(self):
        # Rows that do not match the forward's tokens: zeros of the right size
        # when nothing is steered (exact at alpha 0), an error when something is.
        spec = SimpleNamespace(num_tokens_per_req=3)
        batch = _schedule_batch(
            ForwardMode.TARGET_VERIFY, [0.0, 0.0], spec_info=spec, tokens=4
        )
        self.assertEqual(self._rows(batch), ([0.0] * 4, False))
        batch = _schedule_batch(
            ForwardMode.TARGET_VERIFY, [2.0, 0.0], spec_info=spec, tokens=4
        )
        with self.assertRaisesRegex(RuntimeError, "6 rows for a TARGET_VERIFY"):
            forward_alpha_rows(batch, device=_CPU)


class TestRefusal(unittest.TestCase):
    def _refusal(self, **kwargs):
        base = dict(
            enabled=True,
            armed=True,
            disaggregated=False,
            dllm=False,
            beam=False,
            ragged_verify_mode="static",
        )
        base.update(kwargs)
        return refusal_for_nonzero_alpha(**base)

    def test_armed_engine_serves(self):
        self.assertIsNone(self._refusal())
        self.assertIsNone(self._refusal(streaming_session=False, embedding=False))

    def test_each_missing_condition_refuses(self):
        cases = {
            "--glp-alpha-per-request": dict(enabled=False, armed=False),
            "not armed": dict(armed=False),
            "disaggregation": dict(disaggregated=True),
            "dLLM": dict(dllm=True),
            "speculative verify mode": dict(ragged_verify_mode="compact"),
            "beam": dict(beam=True),
            "streaming session": dict(streaming_session=True),
            "embedding": dict(embedding=True),
        }
        for needle, kwargs in cases.items():
            with self.subTest(needle=needle):
                message = self._refusal(**kwargs)
                self.assertIn(needle, message)
                # Clients learn what is refused and why, and no engine setting
                # beyond the launch flag.
                self.assertNotIn("SGLANG_", message)
                self.assertNotIn("WEIGHTLESS_", message)
                self.assertNotIn("compact", message)
                if needle != "--glp-alpha-per-request":
                    self.assertNotIn("--", message)


class TestBootRefusal(unittest.TestCase):
    _CLEAN = dict(
        disaggregated=False,
        dllm=False,
        ragged_verify_mode="static",
        uno=False,
        plugin_speculative=False,
        context_parallel=False,
        sequence_sharded=False,
        prefill_cuda_graph=False,
        cpu_graph=False,
        document_kv_reuse=False,
        rust_server=False,
    )

    def test_a_wired_engine_boots(self):
        self.assertIsNone(refusal_at_boot(**self._CLEAN))

    def test_each_unwired_configuration_refuses_in_one_line(self):
        cases = {
            "disaggregation": dict(disaggregated=True),
            "dLLM": dict(dllm=True),
            "SGLANG_RAGGED_VERIFY_MODE='compact'": dict(ragged_verify_mode="compact"),
            "UNO": dict(uno=True),
            "a plugin speculative algorithm": dict(plugin_speculative=True),
            "context parallelism": dict(context_parallel=True),
            "attention-TP sequence sharding": dict(sequence_sharded=True),
            "prefill CUDA graphs": dict(prefill_cuda_graph=True),
            "CPU graphs": dict(cpu_graph=True),
            "Unlimited-OCR v2": dict(document_kv_reuse=True),
            "SGLANG_RUST_SERVER": dict(rust_server=True),
        }
        for needle, kwargs in cases.items():
            with self.subTest(needle=needle):
                message = refusal_at_boot(**{**self._CLEAN, **kwargs})
                self.assertTrue(
                    message.startswith("--glp-alpha-per-request is not supported with ")
                )
                self.assertIn(needle, message)
                self.assertNotIn("\n", message)
        message = refusal_at_boot(**{**self._CLEAN, "dllm": True, "uno": True})
        self.assertIn("dLLM", message)
        self.assertIn("; UNO", message)


def _model_runner(**overrides):
    """A stand-in target model runner: what EagerRunner.__init__ and
    BaseRunner._dummy_run read of one, on the CPU."""
    mr = SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        is_draft_worker=False,
        is_generation=True,
        max_running_requests=4,
        max_total_num_tokens=64,
        decode_num_tokens_per_req=lambda num_draft_tokens=None: 1,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            hidden_size=8,
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ),
        server_args=SimpleNamespace(
            disaggregation_mode="null",
            dllm_algorithm=None,
            enable_attn_tp_input_scattered=False,
            enable_layernorm_sp=False,
        ),
    )
    for key, value in overrides.items():
        setattr(mr, key, value)
    return mr


class TestEagerBootRefusal(unittest.TestCase):
    """The boot refusal where it runs, EagerRunner.__init__ of a target model
    runner, under a published (dummy) engine config."""

    def _build(self, mr=None, *, env=None, **server):
        server.setdefault("glp_alpha_per_request", True)
        with ExitStack() as stack:
            stack.enter_context(get_context().override_server_args(**server))
            stack.enter_context(
                get_parallel().override(tp_size=1, attn_tp_size=1, moe_tp_size=1)
            )
            stack.enter_context(
                mock.patch.object(
                    eager_runner.DllmConfig, "from_server_args", return_value=None
                )
            )
            # After the publish, which reads the environment once.
            stack.enter_context(mock.patch.dict(os.environ, env or {}))
            return eager_runner.EagerRunner(mr or _model_runner(), run_warmup=False)

    def assertRefused(self, needle, mr=None, **kwargs):
        with self.assertRaises(RuntimeError) as caught:
            self._build(mr, **kwargs)
        message = str(caught.exception)
        self.assertTrue(
            message.startswith("--glp-alpha-per-request is not supported with ")
        )
        self.assertIn(needle, message)

    def test_a_wired_target_runner_boots_with_its_slot(self):
        runner = self._build()
        self.assertTrue(runner._eager_registry.has_slot("glp_alpha_rows"))

    def test_a_plugin_speculative_algorithm_refuses(self):
        algo = CustomSpecAlgo("glp-test", factory=lambda server_args: None)
        self.assertRefused(
            "a plugin speculative algorithm", _model_runner(spec_algorithm=algo)
        )

    def test_cpu_graphs_refuse(self):
        self.assertRefused("CPU graphs", enable_torch_compile=True)

    def test_unlimited_ocr_v2_refuses(self):
        uocr = SimpleNamespace(
            architectures=["UnlimitedOCRForCausalLM"],
            sparse_attention_config={"training_stage": "dash"},
        )
        mr = _model_runner()
        mr.model_config.hf_config = uocr
        self.assertRefused("Unlimited-OCR v2", mr)

    def test_the_rust_server_refuses(self):
        self.assertRefused("SGLANG_RUST_SERVER", env={"SGLANG_RUST_SERVER": "1"})

    def test_flag_off_and_draft_runners_are_not_refused(self):
        runner = self._build(
            env={"SGLANG_RUST_SERVER": "1"}, glp_alpha_per_request=False
        )
        self.assertFalse(runner._eager_registry.has_slot("glp_alpha_rows"))
        runner = self._build(
            _model_runner(is_draft_worker=True), enable_torch_compile=True
        )
        self.assertFalse(runner._eager_registry.has_slot("glp_alpha_rows"))


class TestDummyRunRows(unittest.TestCase):
    """The dummy forwards of the warm-ups (FlashInfer autotune, the PP-parallel
    DeepGEMM warm-up) carry zero rows of their token count on a target runner,
    and none on a draft runner or with the flag off."""

    def _dummy_forward(self, mr, *, flag, mode, batch_size=3):
        seen = []
        mr.lora_manager = None
        mr.runner_ids = SimpleNamespace(bind_runner_slots=lambda fb: None)
        mr.prepare_dummy_forward_batch = lambda fb: fb
        mr.attn_backend = SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: 1,
            init_forward_metadata=lambda fb: None,
        )
        mr.model = SimpleNamespace(
            forward=lambda input_ids, positions, fb, **kwargs: seen.append(fb)
        )
        buffers = SimpleNamespace(
            input_ids=torch.zeros(8, dtype=torch.int64),
            positions=torch.zeros(8, dtype=torch.int64),
            out_cache_loc=torch.zeros(8, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 8), dtype=torch.int64),
            num_token_non_padded=None,
            req_pool_indices=torch.zeros(8, dtype=torch.int64),
            seq_lens=torch.ones(8, dtype=torch.int64),
            seq_lens_cpu=torch.ones(8, dtype=torch.int64),
            next_token_logits_buffer=None,
            encoder_lens=None,
            global_num_tokens_gpu=torch.zeros(1, dtype=torch.int32),
            global_num_tokens_for_logprob_gpu=torch.zeros(1, dtype=torch.int32),
            custom_mask=None,
            rids_int=None,
            bootstrap_room_ids_int=None,
            ngram_embedding_info=None,
        )
        runner = eager_runner.EagerRunner.__new__(eager_runner.EagerRunner)
        runner.model_runner = mr
        group = SimpleNamespace(world_size=1, barrier=lambda: None)
        with (
            get_context().override_server_args(glp_alpha_per_request=flag),
            get_parallel().override(tp_group=group),
        ):
            runner._dummy_run(
                batch_size=batch_size, forward_mode_override=mode, buffers=buffers
            )
        (forward_batch,) = seen
        return forward_batch

    def test_target_runner_dummies_carry_zero_rows(self):
        for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
            with self.subTest(mode=mode):
                fb = self._dummy_forward(_model_runner(), flag=True, mode=mode)
                rows = fb.glp_alpha_rows
                self.assertEqual(rows.dtype, torch.float32)
                self.assertEqual(rows.shape, fb.input_ids.shape)
                self.assertEqual(rows.tolist(), [0.0] * 3)
                self.assertFalse(fb.glp_any_steered)

    def test_draft_runner_and_flag_off_carry_none(self):
        for mr, flag in ((_model_runner(is_draft_worker=True), True), (None, False)):
            with self.subTest(flag=flag):
                fb = self._dummy_forward(
                    mr or _model_runner(), flag=flag, mode=ForwardMode.DECODE
                )
                self.assertIsNone(fb.glp_alpha_rows)


class TestArmed(unittest.TestCase):
    def test_the_plugin_arms_this_process(self):
        with mock.patch.object(glp_alpha, "_ARMED", False):
            self.assertFalse(glp_alpha.per_request_armed())
            glp_alpha.mark_per_request_armed(file_sha256="0" * 64)
            self.assertTrue(glp_alpha.per_request_armed())

    def test_a_second_arming_keeps_the_process_armed(self):
        with mock.patch.object(glp_alpha, "_ARMED", False):
            glp_alpha.mark_per_request_armed(file_sha256="0" * 64)
            glp_alpha.mark_per_request_armed(file_sha256=None)
            self.assertIs(glp_alpha.per_request_armed(), True)

    def test_only_the_fact_is_kept(self):
        # The plugin passes its file's hash and nothing else; the bound on
        # alpha is the engine's own.
        with mock.patch.object(glp_alpha, "_ARMED", False):
            with self.assertRaises(TypeError):
                glp_alpha.mark_per_request_armed(file_sha256=None, max_alpha=2.0)
            self.assertIs(glp_alpha.per_request_armed(), False)
        self.assertEqual(glp_alpha.GLP_ALPHA_MAX, 4.0)


class TestHttpCheck(unittest.TestCase):
    """The HTTP process refuses an invalid custom_params.glp_alpha (a 400)
    before the request reaches the scheduler."""

    def test_invalid_values_fail_verify(self):
        for value in ("x", True, [2.0], _HUGE, -_HUGE, math.nan, 4.5, -1, 2.255):
            with self.subTest(value=value if isinstance(value, float) else type(value)):
                params = SamplingParams(custom_params={GLP_ALPHA_PARAM: value})
                with self.assertRaisesRegex(ValueError, "^Invalid glp_alpha: "):
                    params.verify(vocab_size=128)

    def test_valid_and_absent_values_pass_verify(self):
        for custom_params in (
            None,
            {},
            {GLP_ALPHA_PARAM: 0},
            {GLP_ALPHA_PARAM: 2.5},
            {GLP_ALPHA_PARAM: 2.25},
            {"other": "x"},
            [1],
        ):
            with self.subTest(custom_params=custom_params):
                SamplingParams(custom_params=custom_params).verify(vocab_size=128)


class TestAnthropicField(unittest.TestCase):
    def _request(self, **extra):
        body = dict(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        body.update(extra)
        return AnthropicMessagesRequest(**body)

    def test_field_is_accepted(self):
        self.assertEqual(self._request(glp_alpha=2.0).glp_alpha, 2.0)
        self.assertEqual(self._request(glp_alpha=2).glp_alpha, 2.0)
        self.assertEqual(self._request(glp_alpha=2.25).glp_alpha, 2.25)
        self.assertEqual(self._request(glp_alpha=0.07).glp_alpha, 0.07)
        self.assertEqual(self._request(glp_alpha=GLP_ALPHA_MAX).glp_alpha, 4.0)
        self.assertEqual(self._request(glp_alpha=0).glp_alpha, 0.0)
        self.assertIsNone(self._request().glp_alpha)
        # The field holds the two-decimal value the engine reads.
        self.assertEqual(self._request(glp_alpha=0.1 + 0.2).glp_alpha, 0.3)

    def test_more_than_two_decimals_is_rejected(self):
        for value in (2.255, 0.001, 3.999):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError) as caught:
                    self._request(glp_alpha=value)
                self.assertIn("more than two decimals", str(caught.exception))

    def test_out_of_range_and_non_finite_are_rejected(self):
        for value in (-1.0, GLP_ALPHA_MAX + 1.0, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    self._request(glp_alpha=value)

    def test_an_int_past_the_float_range_is_a_validation_error(self):
        # Not an OverflowError: pydantic turns only ValueError into a 400.
        for value in (_HUGE, -_HUGE):
            with self.assertRaises(ValidationError) as caught:
                self._request(glp_alpha=value)
            self.assertIn("beyond the float range", str(caught.exception))

    def test_non_numbers_are_rejected(self):
        # A JSON true is an int in Python and a numeric string coerces; both
        # mean something other than "steer at this strength", so both are
        # refused, like the engine's own custom_params reader refuses them.
        for value in (True, False, "2.0", [2.0], "secret text"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError) as caught:
                    self._request(glp_alpha=value)
                self.assertIn("glp_alpha must be a number", str(caught.exception))


def _verify_batch(rows: int, glp_rows) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=rows,
        input_ids=torch.zeros(rows, dtype=torch.long),
        positions=torch.zeros(rows, dtype=torch.long),
        out_cache_loc=torch.zeros(rows, dtype=torch.long),
        req_pool_indices=torch.zeros(rows, dtype=torch.long),
        seq_lens=torch.ones(rows, dtype=torch.int32),
        seq_lens_cpu=torch.ones(rows, dtype=torch.int32),
        seq_lens_sum=rows,
        spec_info=None,
        glp_alpha_rows=glp_rows,
        glp_any_steered=glp_rows is not None,
    )


class TestTwoBatchOverlapSplit(unittest.TestCase):
    def _filter(self, batch, *, lo, hi):
        with (
            get_context().override_server_args(
                attention_backend="fa3", moe_dense_tp_size=None
            ),
            get_parallel().override(attn_tp_size=1),
        ):
            return TboForwardBatchPreparer.filter_batch(
                batch,
                start_token_index=lo,
                end_token_index=hi,
                start_seq_index=lo,
                end_seq_index=hi,
                out_num_token_non_padded=torch.tensor(hi - lo),
            )

    def test_children_take_their_own_rows(self):
        rows = torch.arange(8, dtype=torch.float32)
        child = self._filter(_verify_batch(8, rows), lo=4, hi=8)
        self.assertEqual(child.glp_alpha_rows.tolist(), [4.0, 5.0, 6.0, 7.0])
        self.assertTrue(child.glp_any_steered)
        child = self._filter(_verify_batch(8, None), lo=0, hi=4)
        self.assertIsNone(child.glp_alpha_rows)
        self.assertFalse(child.glp_any_steered)

    def test_rows_are_checked_like_the_other_token_fields(self):
        with self.assertRaisesRegex(AssertionError, "glp_alpha_rows has 6 rows"):
            self._filter(_verify_batch(8, torch.zeros(6)), lo=0, hi=4)


class TestPrePlannedReplayRows(unittest.TestCase):
    """A pre-planned verify replay copies the steering rows of the batch it
    runs, also when the input restage is skipped as redundant."""

    def _runner(self, *, slot=True):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.buffers = SimpleNamespace(
            input_ids=torch.full((8,), -1, dtype=torch.int64),
            positions=torch.full((8,), -1, dtype=torch.int64),
        )
        runner.raw_num_token = 3
        runner._prefilled_inputs = None
        runner.num_prefilled_input_skips = 0
        runner._restage_skip = True
        runner.device_module = SimpleNamespace(current_stream=lambda: "forward")
        if slot:
            glp_slot = GraphSlot(
                "glp_alpha_rows",
                lambda _bs, tokens: (tokens,),
                torch.float32,
                axis="tokens",
                padding_policy=PaddingPolicy.ZERO,
            )
            glp_slot.buffer = torch.full((8,), 9.0)
            runner._glp_alpha_slot = glp_slot
        return runner

    def _batch(self, rows):
        return SimpleNamespace(
            input_ids=torch.arange(10, 13),
            positions=torch.arange(3),
            glp_alpha_rows=rows,
        )

    def test_rows_refresh_when_the_restage_is_redundant(self):
        runner = self._runner()
        fb = self._batch(torch.tensor([0.0, 2.0, 0.0]))
        runner._staged_inputs = runner._staged_signature(fb)
        runner._copy_replay_inputs(fb)
        # The full load staged these inputs, so their copy was skipped...
        self.assertEqual(runner.buffers.input_ids[:3].tolist(), [-1, -1, -1])
        # ...but the rows are the replayed batch's, and the tail is untouched.
        self.assertEqual(
            runner._glp_alpha_slot.buffer.tolist(), [0.0, 2.0, 0.0] + [9.0] * 5
        )

    def test_a_batch_without_rows_replays_zeros(self):
        runner = self._runner()
        runner._copy_replay_inputs(self._batch(None))
        self.assertEqual(runner._glp_alpha_slot.buffer[:3].tolist(), [0.0] * 3)
        self.assertEqual(runner.buffers.input_ids[:3].tolist(), [10, 11, 12])

    def test_a_runner_without_the_slot_ignores_the_rows(self):
        # The class default: a draft runner, or the flag off.
        runner = self._runner(slot=False)
        self.assertIsNone(runner._glp_alpha_slot)
        fb = SimpleNamespace(input_ids=torch.arange(10, 13), positions=torch.arange(3))
        runner._copy_replay_inputs(fb)
        self.assertEqual(runner.buffers.input_ids[:3].tolist(), [10, 11, 12])


if __name__ == "__main__":
    unittest.main()

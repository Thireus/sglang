"""Scheduler admission of a per-request GLP steering alpha: a refused request
is aborted at alpha 0 on its stock cache key, an admitted one runs at its
two-decimal value in that value's namespace and is logged once after every
admission check, a value with more decimals is refused, and embedding
requests and streaming sessions refuse a non-zero alpha."""

import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.glp_alpha import alpha_namespace  # noqa: E402
from sglang.srt.managers.req_types import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.session.session_controller import SessionController  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.managers.scheduler"


def _scheduler(spec_algorithm=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.validate_dllm_request = lambda req: None
    scheduler.spec_algorithm = spec_algorithm or SimpleNamespace(
        is_dflash_family=lambda: False,
        is_uno=lambda: False,
        is_none=lambda: True,
    )
    # The key each request has when the elastic namespacing sees it.
    scheduler.keys_at_namespacing = []
    scheduler._maybe_namespace_elastic_radix_cache = lambda req: (
        scheduler.keys_at_namespacing.append(req.extra_key)
    )
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler._create_generation_req = lambda recv_req, *args, **kw: Req(*args, **kw)
    scheduler._add_request_to_queue = MagicMock()
    scheduler._bind_pp_arrival_prefetch = MagicMock()
    scheduler.init_req_max_new_tokens = MagicMock()
    scheduler.invariant_checker = MagicMock()
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.process_req_with_grammar.return_value = False
    scheduler.beam_coordinator = MagicMock()
    scheduler.beam_coordinator.validate_and_init.return_value = None
    scheduler.tree_cache = MagicMock()
    scheduler.max_req_input_len = 1024
    return scheduler


def _recv(custom_params=None, *, session_params=None, **sampling):
    return MagicMock(
        rid="r",
        input_text="",
        input_ids=array("q", [5, 6, 7]),
        mm_inputs=None,
        session_id=None,
        session_params=session_params,
        sampling_params=SamplingParams(
            max_new_tokens=8, custom_params=custom_params, **sampling
        ),
        input_embeds=None,
        positional_embed_overrides=None,
        token_type_ids=None,
        custom_logit_processor=None,
        lora_id=None,
        lora_name=None,
        extra_key=None,
        cache_salt=None,
        routing_key=None,
        priority=None,
        kv_hints=None,
        time_stats=None,
        http_worker_ipc=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        multi_item_delimiter_indices=None,
        token_indices_to_pool=None,
        bootstrap_host=None,
        bootstrap_port=1,
        bootstrap_room=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        pp_prefetch_ticketed=False,
    )


class _GlpCase(CustomTestCase):
    def _handle(self, scheduler, recv_req, *, flag=False, armed=False):
        with (
            get_context().override_server_args(glp_alpha_per_request=flag),
            patch(f"{_LOGGER}.per_request_armed", return_value=armed),
            patch("sglang.srt.managers.schedule_batch.get_parallel") as parallel,
        ):
            parallel().tp_rank = 0
            scheduler.handle_generate_request(recv_req)
        queued = scheduler._add_request_to_queue.call_args_list
        self.assertEqual(len(queued), 1)
        return queued[0].args[0]

    def assertRefused(self, scheduler, req, *needles):
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        message = req.to_finish.message
        for needle in needles:
            self.assertIn(needle, message)
        # Back to alpha 0 on the stock key before the namespacing, so the
        # one-token abort pass is a stock forward.
        self.assertEqual(req.glp_alpha, 0.0)
        self.assertIsNone(req.extra_key)
        self.assertEqual(scheduler.keys_at_namespacing, [None])
        return message


class TestGenerateRefusals(_GlpCase):
    def test_flag_off_refuses_a_nonzero_alpha(self):
        scheduler = _scheduler()
        req = self._handle(scheduler, _recv({"glp_alpha": 2.0}), armed=True)
        message = self.assertRefused(scheduler, req, "--glp-alpha-per-request")
        self.assertNotIn("WEIGHTLESS", message)
        scheduler.init_req_max_new_tokens.assert_called_once_with(req)

    def test_unarmed_engine_refuses_without_naming_its_settings(self):
        scheduler = _scheduler()
        req = self._handle(scheduler, _recv({"glp_alpha": 2.0}), flag=True)
        message = self.assertRefused(scheduler, req, "not armed")
        self.assertNotIn("WEIGHTLESS", message)
        self.assertNotIn("--", message)

    def test_invalid_value_is_refused_without_echoing_it(self):
        # Reached only past the HTTP check (another caller of the scheduler).
        scheduler = _scheduler()
        req = self._handle(scheduler, _recv({"glp_alpha": "secret text"}))
        message = self.assertRefused(scheduler, req, "Invalid glp_alpha", "str")
        self.assertNotIn("secret", message)

    def test_more_than_two_decimals_is_refused_not_rounded(self):
        scheduler = _scheduler()
        req = self._handle(
            scheduler, _recv({"glp_alpha": 2.255}), flag=True, armed=True
        )
        self.assertRefused(
            scheduler, req, "Invalid glp_alpha", "more than two decimals"
        )

    def test_beam_search_refuses_a_nonzero_alpha(self):
        scheduler = _scheduler()
        req = self._handle(
            scheduler, _recv({"glp_alpha": 2.0}, beam_width=2), flag=True, armed=True
        )
        self.assertRefused(scheduler, req, "beam")

    def test_streaming_session_refuses_a_nonzero_alpha(self):
        scheduler = _scheduler()
        scheduler.session_controller = SessionController(MagicMock())
        scheduler.session_controller.open(
            SimpleNamespace(
                session_id="s", capacity_of_str_len=0, streaming=True, timeout=None
            )
        )
        session_params = SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
        )
        req = self._handle(
            scheduler,
            _recv({"glp_alpha": 2.0}, session_params=session_params),
            flag=True,
            armed=True,
        )
        self.assertIs(req.session.streaming, True)
        self.assertRefused(scheduler, req, "streaming session")


class TestGenerateAdmission(_GlpCase):
    def test_admitted_request_keeps_its_alpha_and_is_logged_once(self):
        scheduler = _scheduler()
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            req = self._handle(
                scheduler, _recv({"glp_alpha": 2.0}), flag=True, armed=True
            )
        self.assertIsNone(req.to_finish)
        self.assertEqual(req.glp_alpha, 2.0)
        self.assertEqual(req.extra_key, alpha_namespace(2.0))
        steering = [line for line in logs.output if "GLP per-request" in line]
        self.assertEqual(len(steering), 1)
        self.assertIn("alpha=2.0 rid=r", steering[0])
        scheduler.grammar_manager.process_req_with_grammar.assert_called_once_with(req)

    def test_one_namespace_per_two_decimal_value(self):
        # 2 and 2.0 run at 2.00 in one namespace; 2.25 has its own; float
        # noise from a client's arithmetic runs at the value it carries.
        keys = []
        for value, expected in ((2, 2.0), (2.0, 2.0), (2.25, 2.25), (0.1 + 0.2, 0.3)):
            with self.subTest(value=value):
                scheduler = _scheduler()
                req = self._handle(
                    scheduler, _recv({"glp_alpha": value}), flag=True, armed=True
                )
                self.assertIsNone(req.to_finish)
                self.assertEqual(req.glp_alpha, expected)
                self.assertTrue(req.extra_key.endswith(f":{expected:.2f}"))
                keys.append(req.extra_key)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[1], keys[2])

    def test_a_request_aborted_by_a_later_check_is_not_logged(self):
        # min_p with speculative decoding aborts after the GLP admission.
        scheduler = _scheduler(spec_algorithm=SpeculativeAlgorithm.EAGLE)
        with patch(f"{_LOGGER}.logger") as logger:
            req = self._handle(
                scheduler, _recv({"glp_alpha": 2.0}, min_p=0.5), flag=True, armed=True
            )
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIn("min_p", req.to_finish.message)
        logger.info.assert_not_called()

    def test_alpha_zero_is_stock_and_silent(self):
        for custom_params in (None, {"glp_alpha": 0}, {"other": 1}, [1], "x"):
            with self.subTest(custom_params=custom_params):
                scheduler = _scheduler()
                with patch(f"{_LOGGER}.logger") as logger:
                    req = self._handle(scheduler, _recv(custom_params))
                self.assertIsNone(req.to_finish)
                self.assertEqual((req.glp_alpha, req.extra_key), (0.0, None))
                logger.info.assert_not_called()


class TestEmbeddingRefusal(CustomTestCase):
    def test_embedding_request_refuses_a_nonzero_alpha(self):
        scheduler = _scheduler()
        recv_req = _recv({"glp_alpha": 2.0})
        recv_req.dimensions = recv_req.encoding_format = None
        recv_req.decision_layout = None
        recv_req.return_pooled_hidden_states = False
        with (
            get_context().override_server_args(glp_alpha_per_request=True),
            patch(f"{_LOGGER}.per_request_armed", return_value=True),
            patch("sglang.srt.managers.schedule_batch.get_parallel") as parallel,
        ):
            parallel().tp_rank = 1
            scheduler._handle_embedding_request(recv_req, ExitStack())
        (call,) = scheduler._add_request_to_queue.call_args_list
        req = call.args[0]
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIn("embedding", req.to_finish.message)
        self.assertEqual((req.glp_alpha, req.extra_key), (0.0, None))
        self.assertEqual(scheduler.keys_at_namespacing, [None])


if __name__ == "__main__":
    unittest.main()

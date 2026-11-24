"""Per-request GLP steering alpha over HTTP: /v1/messages takes the field (0
to 4.00, at most two decimals) and hands it to the scheduler, and every other
route that takes custom_params from a client refuses a glp_alpha key there
with a 400, leaving every other key alone. In-process clients against
stand-in serving handlers: nothing listens and no model runs."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ROUTE_ONLY = "glp_alpha is accepted on the Messages route only"
_MESSAGES = [{"role": "user", "content": "hi"}]
_STATE = ("anthropic_serving", "openai_serving_chat", "openai_serving_completion")


class _StandInModel:
    """The real request checks of an OpenAI handler over a stand-in model:
    each request that would reach the engine is recorded, and the answer is
    fixed."""

    def _stand_in(self):
        self.tokenizer_manager = SimpleNamespace(
            request_logger=SimpleNamespace(log_requests=False, log_requests_level=0)
        )
        self.served = []

    def _validate_request(self, request, **kwargs):
        return None

    def _convert_to_internal_request(self, request, raw_request=None):
        self.served.append(request)
        return SimpleNamespace(), request

    async def _handle_non_streaming_request(self, adapted, request, raw_request):
        return {"served": True}


class _Chat(_StandInModel, OpenAIServingChat):
    def __init__(self):  # no tokenizer or template setup
        self._stand_in()
        self._doc_splitter = None
        self._encoding_adapter = None  # the stock OpenAI wire
        self.supports_inline_system = False

    def supports_native_reasoning_history(self):
        return False

    async def _chat_completion_response(self, adapted, request, raw_request):
        return ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )


class _Completion(_StandInModel, OpenAIServingCompletion):
    def __init__(self):  # no tokenizer or template setup
        self._stand_in()


class _TokenizerManager:
    """Answers each /generate, /encode or /classify request it is given."""

    def __init__(self):
        self.served = []

    async def generate_request(self, obj, request=None):
        self.served.append(obj)
        yield {"text": "ok"}


class _HttpCase(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        state = http_server.app.state
        for name in _STATE:
            self.addCleanup(setattr, state, name, getattr(state, name, None))
        self.chat = _Chat()
        self.completion = _Completion()
        state.openai_serving_chat = self.chat
        state.openai_serving_completion = self.completion
        # The Messages route converts its body and calls the very chat handler
        # whose client route refuses the key.
        state.anthropic_serving = AnthropicServing(self.chat)
        prior = http_server.get_global_state()
        self.addCleanup(setattr, http_server, "_global_state", prior)
        self.tokenizer_manager = _TokenizerManager()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=self.tokenizer_manager)
        )
        self.client = TestClient(http_server.app)

    def messages(self, **fields):
        body = {"model": "test-model", "max_tokens": 16, "messages": _MESSAGES}
        return self.client.post("/v1/messages", json={**body, **fields})


class TestMessagesRoute(_HttpCase):
    def test_an_accepted_value_reaches_the_scheduler_key(self):
        for value, carried in ((2.25, 2.25), (2, 2.0), (4.0, 4.0)):
            with self.subTest(value=value):
                response = self.messages(glp_alpha=value)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["content"][0]["text"], "ok")
                # Built after the body's own check and handed on: the route
                # refusal of the same chat handler does not see it.
                self.assertEqual(
                    self.chat.served[-1].custom_params, {"glp_alpha": carried}
                )

    def test_a_refused_value_is_a_400_in_both_stream_modes(self):
        for value in (2.255, 4.5, -1, "2.0", True):
            for stream in (False, True):
                with self.subTest(value=value, stream=stream):
                    response = self.messages(glp_alpha=value, stream=stream)
                    self.assertEqual(response.status_code, 400)
                    error = response.json()["error"]
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertIn("glp_alpha", error["message"])
        self.assertEqual(self.chat.served, [])
        message = self.messages(glp_alpha=2.255).json()["error"]["message"]
        self.assertIn("more than two decimals", message)

    def test_an_absent_field_or_zero_keeps_the_stock_body(self):
        for fields in ({}, {"glp_alpha": 0}, {"glp_alpha": None}):
            with self.subTest(fields=fields):
                self.assertEqual(self.messages(**fields).status_code, 200)
                self.assertIsNone(self.chat.served[-1].custom_params)

    def test_its_body_has_no_custom_params(self):
        # The schema drops unknown fields: only the field steers here.
        response = self.messages(custom_params={"glp_alpha": 2.0})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.chat.served[-1].custom_params)


class TestOtherRoutesRefuseTheKey(_HttpCase):
    def assertRefused(self, response, served):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(_ROUTE_ONLY, response.text)
        self.assertIn("/v1/messages", response.text)
        self.assertEqual(served, [])

    def test_chat_completions(self):
        for value in (2.0, 0, None):
            with self.subTest(value=value):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={"messages": _MESSAGES, "custom_params": {"glp_alpha": value}},
                )
                self.assertRefused(response, self.chat.served)

    def test_completions(self):
        response = self.client.post(
            "/v1/completions",
            json={"prompt": "hi", "custom_params": {"glp_alpha": 2.0}},
        )
        self.assertRefused(response, self.completion.served)

    def test_sagemaker_invocations(self):
        response = self.client.post(
            "/invocations",
            json={"messages": _MESSAGES, "custom_params": {"glp_alpha": 2.0}},
        )
        self.assertRefused(response, self.chat.served)

    def test_generate_one_or_one_per_prompt(self):
        steered = {"custom_params": {"glp_alpha": 2.0}}
        for body in (
            {"text": "hi", "sampling_params": steered},
            {"text": ["a", "b"], "sampling_params": [{}, steered]},
        ):
            with self.subTest(body=body):
                response = self.client.post("/generate", json=body)
                self.assertRefused(response, self.tokenizer_manager.served)

    def test_encode_and_classify(self):
        body = {"text": "hi", "sampling_params": {"custom_params": {"glp_alpha": 2}}}
        for route in ("/encode", "/classify"):
            with self.subTest(route=route):
                response = self.client.post(route, json=body)
                self.assertRefused(response, self.tokenizer_manager.served)

    def test_every_other_key_is_untouched(self):
        params = {"thinking_budget": 8, "k": "v"}
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": _MESSAGES, "custom_params": params},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.chat.served[-1].custom_params, params)
        response = self.client.post(
            "/v1/completions", json={"prompt": "hi", "custom_params": params}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.completion.served[-1].custom_params, params)
        for route in ("/generate", "/encode"):
            with self.subTest(route=route):
                response = self.client.post(
                    route,
                    json={"text": "hi", "sampling_params": {"custom_params": params}},
                )
                self.assertEqual(response.status_code, 200)
                served = self.tokenizer_manager.served[-1]
                self.assertEqual(served.sampling_params["custom_params"], params)


class TestGrpcBridge(CustomTestCase):
    """The gRPC bridge's native generate and embed requests refuse the key
    before anything is submitted (a ValueError is INVALID_ARGUMENT there), and
    its OpenAI requests take the handler's own refusal."""

    def test_openai_requests_take_the_handler_refusal(self):
        chat = _Chat()
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle._get_openai_serving = lambda: {"chat": chat}
        calls = []
        body = {"messages": _MESSAGES, "custom_params": {"glp_alpha": 2.0}}
        asyncio.run(
            handle._run_openai_request(
                "chat",
                json.dumps(body).encode(),
                lambda payload, **kwargs: calls.append((payload, kwargs)),
                streaming=True,
            )
        )
        ((payload, kwargs),) = calls
        self.assertEqual(kwargs["status_code"], 400)
        self.assertIn(_ROUTE_ONLY, payload.decode())
        self.assertEqual(chat.served, [])

    def _handle(self, submitted):
        def submit(coro):
            submitted.append(coro.cr_code.co_name)
            coro.close()

        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle._submit_on_tm_loop = submit
        return handle

    def test_the_key_is_refused_and_other_keys_pass(self):
        submitted = []
        handle = self._handle(submitted)
        for req_type in ("generate", "embed"):
            with self.subTest(req_type=req_type):
                steered = {"custom_params": {"glp_alpha": 2.0}}
                with self.assertRaisesRegex(ValueError, _ROUTE_ONLY):
                    handle.submit_request(
                        req_type=req_type,
                        req_dict={"text": "hi", "sampling_params": steered},
                        chunk_callback=None,
                    )
                self.assertEqual(submitted, [])
        handle.submit_request(
            req_type="generate",
            req_dict={"text": "hi", "sampling_params": {"custom_params": {"k": 1}}},
            chunk_callback=None,
        )
        self.assertEqual(submitted, ["_run_generate"])


if __name__ == "__main__":
    unittest.main()

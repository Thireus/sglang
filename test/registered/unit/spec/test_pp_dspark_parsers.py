"""A verify round hands the parsers up to six tokens at once. They must not care."""

import random
import unittest

from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.parser.reasoning_parser import DeepSeekV4Detector
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="read_file",
            description="read",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        ),
    )
]
DSML = "｜DSML｜"
ANSWER = (
    "Let me look.\n\n"
    f"<{DSML} calls>\n"
    f'<{DSML} invoke name="read_file">\n'
    f'<{DSML} parameter name="path" string="true">/tmp/a.txt</{DSML} parameter>\n'
    f'<{DSML} parameter name="limit" string="false">20</{DSML} parameter>\n'
    f"</{DSML} invoke>\n"
    f"</{DSML} calls>"
)
THOUGHT = "<think>first a thought, with <b>markup</b> inside</think>then the answer"


def _bursts(text, rng, *, lo=1, hi=6):
    """Pieces of a few characters, grouped in bursts of up to six: what one to six
    decoded tokens hand the stream at a time."""
    pieces, i = [], 0
    while i < len(text):
        n = rng.randrange(1, 5)
        pieces.append(text[i : i + n])
        i += n
    out, i = [], 0
    while i < len(pieces):
        n = rng.randrange(lo, hi + 1)
        out.append("".join(pieces[i : i + n]))
        i += n
    return out


def _tool_calls(chunks):
    detector = DeepSeekV41Detector()
    calls, normal = {}, ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        normal += result.normal_text or ""
        for call in result.calls:
            entry = calls.setdefault(call.tool_index, {"name": None, "args": ""})
            if call.name:
                entry["name"] = call.name
            entry["args"] += call.parameters or ""
    return normal, calls


def _reasoning(chunks):
    detector = DeepSeekV4Detector(stream_reasoning=True)
    reasoning, normal = "", ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk)
        reasoning += result.reasoning_text
        normal += result.normal_text
    return reasoning, normal


class TestBursts(CustomTestCase):
    def test_tool_parser_is_the_same_for_bursts_of_one_to_six(self):
        """Invariant 25."""
        want = _tool_calls(list(ANSWER))
        self.assertEqual(want[1][0]["name"], "read_file")
        self.assertIn("/tmp/a.txt", want[1][0]["args"])
        for seed in range(40):
            got = _tool_calls(_bursts(ANSWER, random.Random(seed)))
            self.assertEqual(got, want, seed)

    def test_reasoning_parser_is_the_same_for_bursts_of_one_to_six(self):
        want = _reasoning(list(THOUGHT))
        self.assertIn("thought", want[0])
        self.assertIn("answer", want[1])
        for seed in range(40):
            got = _reasoning(_bursts(THOUGHT, random.Random(seed)))
            self.assertEqual(got, want, seed)


if __name__ == "__main__":
    unittest.main()

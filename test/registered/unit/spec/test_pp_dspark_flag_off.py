"""With the flag off, the code path is the code path without the port.

Behaviour first: every new branch is unreachable off the relayed launch. Then, given the
tree the port was cut from (SGLANG_TEST_DSPARK_BASE_TREE), the proof by construction: fold
the flag's values through both trees' syntax and the results must be the same program,
file by file. A leftover names the file and the function.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_pp_launch import (
    pp_dspark_relay_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

TOUCHED = (
    "arg_groups/speculative_hook.py",
    "arg_groups/validation_hook.py",
    "environ.py",
    "managers/scheduler.py",
    "managers/scheduler_pp_mixin.py",
    "managers/utils.py",
    "model_executor/model_runner_components/cuda_graph_setup.py",
    "model_executor/pool_configurator.py",
    "model_executor/runner/decode_cuda_graph_runner.py",
    "models/deepseek_v4.py",
    "speculative/dspark_components/dspark_verify.py",
    "speculative/dspark_components/dspark_worker_v2.py",
)
NEW_FUNCTIONS = {
    "_pp_dspark_launches_first",
    "_pp_dspark_draft_layers",
    "_maybe_add_pp_dspark_plain_family",
}
NEW_ATTRIBUTES = {
    "_pp_dspark_relay",
    "_pp_dspark_launch_first",
    "spec_verify_ids",
    "pp_last_stage",
    "plain_decode_family",
}
NEW_PARAMETERS = {"launch_first", "plain_decode_family", "pp_proxy_tensors"}
NEW_MODULES = (
    "dspark_pp_",
    "scheduler_pp_dspark",
    "decode_family_dispatch",
)


class _FlagOff(ast.NodeTransformer):
    """Partial evaluation of the flag being off, and of nothing else."""

    def __init__(self, path):
        self.verify_file = path.endswith("dspark_verify.py")

    # -- values --------------------------------------------------------------
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in ("_pp_dspark_relay", "_pp_dspark_launch_first"):
                return ast.Constant(False)
            if node.attr == "plain_decode_family":
                return ast.Constant(False)
            if node.attr == "pp_last_stage":
                return ast.Constant(None)
        return node

    def visit_Name(self, node):
        if node.id in ("launch_first", "plain_decode_family"):
            return ast.Constant(False)
        if node.id == "relay_cls":
            return ast.Name("PPSpecRelayInput", ast.Load())
        if self.verify_file and node.id == "pp_proxy_tensors":
            return ast.Constant(None)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        text = ast.unparse(node.func)
        if text == "envs.SGLANG_ENABLE_PP_SPEC.get":
            return ast.Constant(False)
        if text == "envs.SGLANG_DSV41_DRAFT_DUMP_DIR.get":
            return ast.Constant(False)
        if text in ("self._pp_dspark_launches_first", "pp_dspark_relay_enabled"):
            return ast.Constant(False)
        if text == "_maybe_add_pp_dspark_plain_family":
            return node.args[1]
        keywords = []
        for kw in node.keywords:
            if kw.arg in ("launch_first", "spec_verify_ids", "plain_decode_family"):
                continue
            if kw.arg == "speculative_output_stride" and (
                isinstance(kw.value, ast.Constant) and kw.value.value is None
            ):
                continue
            if self.verify_file and kw.arg == "pp_proxy_tensors":
                continue
            if kw.arg is None and isinstance(kw.value, ast.Dict) and not kw.value.keys:
                continue
            keywords.append(kw)
        node.keywords = keywords
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        values = []
        is_and = isinstance(node.op, ast.And)
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, bool):
                if v.value is (not is_and):
                    return ast.Constant(not is_and)
                continue
            values.append(v)
        if not values:
            return ast.Constant(is_and)
        if len(values) == 1:
            return values[0]
        node.values = values
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.left, ast.Constant):
            right = node.comparators[0]
            if isinstance(right, ast.Constant):
                op = node.ops[0]
                if isinstance(op, ast.Gt):
                    return ast.Constant(node.left.value > right.value)
                if isinstance(op, ast.IsNot):
                    return ast.Constant(node.left.value is not right.value)
                if isinstance(op, ast.Is):
                    return ast.Constant(node.left.value is right.value)
        # EAGLE's relay always has a topology when it has a chain, in that one branch.
        if ast.unparse(node) == "result.next_verify_parent_list is not None":
            return ast.Constant(True)
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    # -- statements ----------------------------------------------------------
    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return (node.body if node.test.value else node.orelse) or None
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_Assign(self, node):
        text = ast.unparse(node.targets[0])
        if text in ("verify_max_bs", "relay_cls", "self.plain_decode_family"):
            return None
        if text in ("self._pp_dspark_relay", "self._pp_dspark_launch_first"):
            return None
        if text == "self.pp_last_stage" or text.startswith(
            ("SGLANG_PP_DSPARK_", "SGLANG_DSV41_DRAFT_DUMP_")
        ):
            return None
        if text in NEW_ATTRIBUTES:
            return None
        node = self.generic_visit(node)
        if text == "runner" and ast.unparse(node.value) == "runner":
            return None
        return node

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.target) in NEW_ATTRIBUTES:
            return None
        return node

    def visit_ImportFrom(self, node):
        if node.module and any(m in node.module for m in NEW_MODULES):
            return None
        return node

    def visit_FunctionDef(self, node):
        if node.name in NEW_FUNCTIONS:
            return None
        # Staticmethods and helpers the port added carry their decorators with them.
        args = node.args
        kept, defaults = [], []
        offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            default = args.defaults[i - offset] if i >= offset else None
            if arg.arg in NEW_PARAMETERS and default is not None and (
                self._drops(arg.arg)
            ):
                continue
            kept.append(arg)
            if default is not None:
                defaults.append(default)
        args.args, args.defaults = kept, defaults
        kw_kept, kw_defaults = [], []
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if arg.arg in NEW_PARAMETERS and default is not None and self._drops(arg.arg):
                continue
            kw_kept.append(arg)
            kw_defaults.append(default)
        args.kwonlyargs, args.kw_defaults = kw_kept, kw_defaults
        return self.generic_visit(node)

    def _drops(self, name):
        return name != "pp_proxy_tensors" or self.verify_file


def _strip_docstring_and_pass(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [
                s
                for s in body
                if not (
                    isinstance(s, ast.Pass)
                    and len(body) > 1
                )
            ]
            if (
                isinstance(node, ast.FunctionDef)
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                # The one docstring the port extends says what the new branch is.
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def _folded(source, path):
    tree = ast.parse(source)
    tree = _FlagOff(path).visit(tree)
    return _strip_docstring_and_pass(tree)


def _first_difference(base, patched):
    """The outermost function or class whose syntax differs."""
    base_defs = {
        n.name: n
        for n in ast.walk(base)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    patched_defs = {
        n.name: n
        for n in ast.walk(patched)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    out = sorted(set(base_defs) ^ set(patched_defs))
    for name in sorted(set(base_defs) & set(patched_defs)):
        if isinstance(base_defs[name], ast.FunctionDef) and ast.dump(
            base_defs[name]
        ) != ast.dump(patched_defs[name]):
            out.append(name)
    return out


class TestFlagOffBehaviour(CustomTestCase):
    def test_nothing_turns_the_relay_on_without_the_flag(self):
        stage = SchedulerPPMixin()
        self.assertFalse(stage._pp_dspark_relay)
        self.assertFalse(stage._pp_dspark_launches_first(None))
        for pp_size, algorithm in ((3, None), (3, "EAGLE"), (3, "DSPARK"), (1, "DSPARK")):
            self.assertFalse(
                pp_dspark_relay_enabled(pp_size=pp_size, spec_algorithm=algorithm)
            )

    def test_the_flag_alone_does_not_touch_eagle_or_plain(self):
        with envs.SGLANG_ENABLE_PP_SPEC.override(True):
            self.assertFalse(pp_dspark_relay_enabled(pp_size=3, spec_algorithm="EAGLE"))
            self.assertFalse(pp_dspark_relay_enabled(pp_size=3, spec_algorithm=None))
            self.assertFalse(pp_dspark_relay_enabled(pp_size=1, spec_algorithm="DSPARK"))

    def test_exchange_order_without_the_argument_is_the_stock_order(self):
        for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
            for spec_relay in (False, True):
                for is_last in (False, True):
                    for depth in (0, 1):
                        batch = SimpleNamespace(
                            forward_mode=mode, is_extend_in_batch=False
                        )
                        want = depth > 0 or (
                            spec_relay and not is_last and mode == ForwardMode.DECODE
                        )
                        self.assertEqual(
                            _pp_exchange_outputs_before_forward(
                                batch, spec_relay, is_last, depth
                            ),
                            want,
                        )

    def test_eagle_output_keeps_its_topology_keys(self):
        stage = SchedulerPPMixin()
        stage._pp_spec_relay = True
        result = GenerationBatchResult(
            next_token_ids=torch.zeros(4, dtype=torch.int64),
            accept_lens=torch.ones(1, dtype=torch.int32),
            new_seq_lens=torch.ones(1, dtype=torch.int64),
            next_draft_input=SimpleNamespace(
                bonus_tokens=torch.tensor([1]), topk_p=None
            ),
            next_verify_chain=torch.zeros(4, dtype=torch.int64),
            next_verify_parent_list=torch.zeros((1, 3), dtype=torch.int64),
            next_verify_top_scores_index=torch.zeros((1, 3), dtype=torch.int64),
        )
        batch = SimpleNamespace(
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
        )
        tensor_dict = stage._pp_prepare_tensor_dict(result, batch)
        self.assertIn("spec_next_parents", tensor_dict)
        self.assertIn("spec_next_top_scores", tensor_dict)


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_DSPARK_BASE_TREE"),
    "needs SGLANG_TEST_DSPARK_BASE_TREE, the tree the port was cut from",
)
class TestFlagOffIsTheBaseProgram(CustomTestCase):
    def test_every_touched_file_folds_to_the_base(self):
        base_root = Path(os.environ["SGLANG_TEST_DSPARK_BASE_TREE"]) / "python/sglang/srt"
        here = Path(__file__).resolve().parents[4] / "python/sglang/srt"
        for rel in TOUCHED:
            with self.subTest(file=rel):
                base = _folded((base_root / rel).read_text(), rel)
                patched = _folded((here / rel).read_text(), rel)
                self.assertEqual(
                    ast.dump(base),
                    ast.dump(patched),
                    f"{rel}: differs in {_first_difference(base, patched)}",
                )


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Per-request GLP steering alpha: the request field, its isolation key, the
process-wide armed flag, and the per-token row vector a forward carries.

GLP (Weightless's projective activation edit) is normally one scalar per
server, fixed at boot through ``WEIGHTLESS_STEER_ALPHA``. Per-request mode
carries one alpha per request instead, so conversations that share an engine
can steer at different strengths. The edit itself belongs to the Weightless
plugin (``weightless_sglang``); this module only routes and isolates it:

- ``glp_alpha`` is a field of the Anthropic Messages body only. That route
  hands it to the scheduler as the ``glp_alpha`` key of ``custom_params``,
  and every route that takes ``custom_params`` from a client refuses the
  key. The value is a number in [0, 4.00] with at most two decimals, never
  rounded, checked in the HTTP process and again on ``Req`` in every
  scheduler process;
- a non-zero alpha gets its own prefix-cache namespace, keyed on the value
  written with two decimals, because KV computed under one alpha is not the
  KV of another;
- each target forward carries one alpha per token row in ``ForwardBatch``;
- a non-zero alpha is refused unless the plugin armed per-request mode in
  this process, and the flag itself is refused at boot where forwards would
  carry no rows or misplaced ones, or where KV is lent across requests with
  no alpha in its cache key. Refused, never served unsteered.

Alpha 0 adds no namespace, so unsteered conversations share the cache as
they do today. Its rows are not always the plugin's stock path, though: the
plugin skips an eager forward whose rows are all zero, but every captured
decode and verify graph runs the edit, and so does a batch that mixes
alphas; there an alpha-0 row goes through the edit as a numeric no-op.
"""

from __future__ import annotations

import math
import secrets
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch

# The request field name: the Anthropic body's field, and the custom_params
# key that route hands to the scheduler.
GLP_ALPHA_PARAM = "glp_alpha"

# The one route a client may send the field to.
GLP_ALPHA_ROUTE = "/v1/messages"

# The value rule: a strength in [0, GLP_ALPHA_MAX] with at most two decimals,
# so a whole number of GLP_ALPHA_STEP. The Weightless client window is [0, 4]
# (WEIGHTLESS_CONTROL_MIN_ALPHA / _MAX_ALPHA); this is the same ceiling,
# enforced here because a graph replay cannot check a value.
GLP_ALPHA_MAX = 4.0
GLP_ALPHA_STEP = 0.01

# How far alpha * 100 may lie from a whole number and still be that many
# hundredths. The float nearest a two-decimal value in [0, 4] lands within
# 5.1e-14 of it (one rounding to read the decimal, one to multiply), and a
# client's own float arithmetic adds a few ulps; a value written with up to 13
# decimals that is not a whole number of hundredths lies at least 1e-11 away.
# So a value with more decimals is refused, and float noise reads as the
# two-decimal value it carries.
_HUNDREDTHS_TOLERANCE = 1e-12


def read_request_alpha(
    custom_params: Optional[Dict[str, object]],
) -> Tuple[float, Optional[str]]:
    """``(alpha, error)`` for one request's custom params; never raises.

    Alpha defaults to 0.0 (unsteered), also when ``custom_params`` is not a
    dict, which every other reader of it ignores as well. A present value
    must be a real number in [0, GLP_ALPHA_MAX] with at most two decimals, and
    the alpha returned is that two-decimal value, so 2, 2.0 and 2.00 read the
    same. ``error`` is a client-facing message when it is not, in which case
    the alpha reads 0.0 and the caller must refuse the request: a value is
    never rounded to fit. The message names the type of a value that is not a
    number, never the value.
    """
    raw = (
        custom_params.get(GLP_ALPHA_PARAM) if isinstance(custom_params, dict) else None
    )
    if raw is None:
        return 0.0, None
    # bool is an int in Python; True/False here is a client mistake, not 1/0.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0, (
            f"{GLP_ALPHA_PARAM} must be a number in [0, {GLP_ALPHA_MAX:.2f}] with "
            f"at most two decimals, got {type(raw).__name__}"
        )
    try:
        alpha = float(raw)
    except (OverflowError, TypeError, ValueError):
        # An int past the float range (310 digits or more).
        return 0.0, (
            f"{GLP_ALPHA_PARAM} must be a finite number in [0, {GLP_ALPHA_MAX:.2f}], "
            f"got {type(raw).__name__} beyond the float range"
        )
    if not math.isfinite(alpha):
        return 0.0, f"{GLP_ALPHA_PARAM} must be finite, got {alpha!r}"
    if not 0.0 <= alpha <= GLP_ALPHA_MAX:
        return 0.0, f"{GLP_ALPHA_PARAM}={alpha!r} is outside [0, {GLP_ALPHA_MAX:.2f}]"
    hundredths = round(alpha * 100)
    if abs(alpha * 100 - hundredths) > _HUNDREDTHS_TOLERANCE:
        return 0.0, (
            f"{GLP_ALPHA_PARAM}={alpha!r} has more than two decimals and is not "
            f"rounded: send a multiple of {GLP_ALPHA_STEP} in "
            f"[0, {GLP_ALPHA_MAX:.2f}]"
        )
    # The two-decimal value itself; -0.0 reads as the plain zero.
    return hundredths / 100, None


# Random per process, so no client can name another alpha's namespace with its
# own extra_key: ``extra_key`` is caller-supplied, and a stock request that
# guessed ``|glp:2.00`` would otherwise read KV computed under the alpha-2 edit.
# A steered request only ever meets namespaces this process minted, and two
# requests at the same alpha still share one namespace (one salt per process).
_NAMESPACE_SALT = secrets.token_hex(4)


def alpha_namespace(alpha: float) -> str:
    """The ``extra_key`` suffix that isolates one alpha's KV; empty at 0.

    ``alpha`` is a value ``read_request_alpha`` returned, a whole number of
    hundredths, and the namespace names it written with two decimals: 2, 2.0
    and 2.00 share one, 2.25 has its own, and a process mints at most 400.
    Alpha 0 keeps the empty suffix so unsteered conversations share the cache
    as they do today.
    """
    if alpha == 0.0:
        return ""
    return f"|glp:{_NAMESPACE_SALT}:{alpha:.2f}"


def client_custom_params_refusal(custom_params) -> Optional[str]:
    """The 400 text for a client's ``custom_params`` that carries the
    ``glp_alpha`` key, or None; every other key is left to the route.

    Only the Messages route takes ``glp_alpha``, as a field of its body, and
    it sets the key itself once that body is parsed. A route that takes
    ``custom_params`` from a client calls this at its input boundary, so no
    client reaches the key around the Messages route's own check.
    """
    if isinstance(custom_params, dict) and GLP_ALPHA_PARAM in custom_params:
        return (
            f"{GLP_ALPHA_PARAM} is accepted on the Messages route only: send it "
            f"as the {GLP_ALPHA_PARAM} field of {GLP_ALPHA_ROUTE}, not in "
            "custom_params"
        )
    return None


def client_sampling_params_refusal(sampling_params) -> Optional[str]:
    """``client_custom_params_refusal`` for the ``sampling_params`` of a
    native body (/generate, /encode): one dict, or one per prompt."""
    for params in (
        sampling_params if isinstance(sampling_params, list) else [sampling_params]
    ):
        if isinstance(params, dict):
            refusal = client_custom_params_refusal(params.get("custom_params"))
            if refusal is not None:
                return refusal
    return None


def request_token_rows(
    *,
    forward_mode,
    num_reqs: int,
    extend_lens: Optional[Sequence[int]],
    tokens_per_req: Optional[int],
) -> Optional[List[int]]:
    """Token rows each request contributes to this forward, in ``batch.reqs``
    order, or None for a mode per-request alpha is not wired into.

    Rows follow ``batch.reqs``: an extend forward gives each request its
    ``extend_len`` (1 for a decode request inside a mixed batch), a decode
    forward gives every request one row, and a target verify gives every
    request its verify width.
    """
    if num_reqs == 0 or forward_mode.is_idle():
        # No request rows (an idle forward gets an empty vector), and nothing
        # to complain about on the next real forward.
        return []
    if forward_mode.is_decode():
        return [1] * num_reqs
    if forward_mode.is_target_verify():
        return [int(tokens_per_req or 1)] * num_reqs
    if forward_mode.is_extend() and not forward_mode.is_dllm_extend():
        if extend_lens is None or len(extend_lens) != num_reqs:
            return None
        return [int(n) for n in extend_lens]
    # DRAFT_EXTEND_V2 runs on draft workers, which carry no vector; DLLM_EXTEND
    # and PREBUILT need dLLM and disaggregation, which refuse the flag at boot.
    return None


def refusal_for_nonzero_alpha(
    *,
    enabled: bool,
    armed: bool,
    disaggregated: bool,
    dllm: bool,
    beam: bool,
    ragged_verify_mode: str,
    streaming_session: bool = False,
    embedding: bool = False,
) -> Optional[str]:
    """Why this engine refuses a request's non-zero alpha (None when it serves
    it). The text goes back to the client.

    Fail closed: every reason here is a configuration or a request in which
    the edit would be missing or applied to the wrong rows, and the request is
    refused rather than served as if it were stock. The texts say what is
    refused and why, and name no engine setting but the launch flag.
    """
    if not enabled:
        return (
            "per-request GLP steering is off on this engine "
            "(launched without --glp-alpha-per-request)"
        )
    if not armed:
        return (
            "per-request GLP steering is not armed on this engine: its "
            "steering plugin is not loaded for per-request alphas"
        )
    if disaggregated:
        return "per-request GLP steering is not available with disaggregation"
    if dllm:
        return "per-request GLP steering is not available for dLLM models"
    if ragged_verify_mode != "static":
        return (
            "per-request GLP steering is not available with this engine's "
            "speculative verify mode"
        )
    if beam:
        return (
            "a non-zero glp_alpha is refused with beam search: the beam adds "
            "token rows that no request owns. Send beam_width=1."
        )
    if streaming_session:
        return (
            "a non-zero glp_alpha is refused on a streaming session: its turns "
            "reuse KV computed under the previous turn's alpha"
        )
    if embedding:
        return (
            "a non-zero glp_alpha is refused on an embedding request: steering "
            "applies to generation only"
        )
    return None


def refusal_at_boot(
    *,
    disaggregated: bool,
    dllm: bool,
    ragged_verify_mode: str,
    uno: bool,
    plugin_speculative: bool,
    context_parallel: bool,
    sequence_sharded: bool,
    prefill_cuda_graph: bool,
    cpu_graph: bool,
    document_kv_reuse: bool,
    rust_server: bool,
) -> Optional[str]:
    """Why ``--glp-alpha-per-request`` cannot run on this engine, as one line
    for the operator (None when it can).

    Each reason is a configuration whose target forwards would carry no rows,
    or rows the model sees in another order or size than its tokens; one that
    lends KV across requests with no alpha in its cache key; or one whose
    routes pass a client's custom_params to the scheduler unchecked. The flag
    is refused at boot instead of a steered request being served unsteered,
    or a client's custom_params steering, later.
    """
    reasons = [
        why
        for unsupported, why in (
            (disaggregated, "prefill-decode disaggregation (no rows on its forwards)"),
            (dllm, "dLLM inference (no rows on its forwards)"),
            (
                ragged_verify_mode != "static",
                f"SGLANG_RAGGED_VERIFY_MODE={ragged_verify_mode!r} "
                "(the rows need the 'static' verify widths)",
            ),
            (uno, "UNO speculative decoding (no rows on its target verify)"),
            (
                plugin_speculative,
                "a plugin speculative algorithm (its target forwards are its own)",
            ),
            (
                context_parallel,
                "context parallelism (it shards the tokens, not the rows)",
            ),
            (
                sequence_sharded,
                "attention-TP sequence sharding (it shards the tokens, not the rows)",
            ),
            (
                prefill_cuda_graph,
                "prefill CUDA graphs (no rows on their replays; pass "
                "--disable-prefill-cuda-graph)",
            ),
            (
                cpu_graph,
                "CPU graphs (no rows on their forwards; run without "
                "--enable-torch-compile)",
            ),
            (
                document_kv_reuse,
                "Unlimited-OCR v2 (its document windows reuse KV with no alpha "
                "in the cache key)",
            ),
            (
                rust_server,
                "SGLANG_RUST_SERVER (its routes take glp_alpha in a client's "
                "custom_params)",
            ),
        )
        if unsupported
    ]
    if not reasons:
        return None
    return "--glp-alpha-per-request is not supported with " + "; ".join(reasons)


def build_alpha_rows(
    *,
    alphas: Sequence[float],
    rows_per_req: Optional[Sequence[int]],
    device: torch.device,
    tail_rows: int = 0,
) -> Tuple[Optional[torch.Tensor], bool]:
    """``(rows, any_steered)``: one fp32 alpha per token row, and whether any
    request in this forward is steered.

    ``rows_per_req`` gives each request's rows in ``batch.reqs`` order, and
    ``tail_rows`` rows no request owns follow them at alpha 0. ``rows`` is
    None for a forward the engine does not carry alphas into
    (``rows_per_req`` None), which only an unsteered batch may reach. An idle
    forward gets an empty vector, which padding extends with zeros.

    Nothing here waits for the device: zeros and a single alpha are filled on
    it, and a mixed batch copies its per-request values once from pinned
    memory and expands them to rows on the device.
    """
    import torch

    from sglang.srt.utils.device import async_h2d

    if rows_per_req is None:
        if any(a != 0.0 for a in alphas):
            # Admission refuses a steered request on every mode this module
            # does not wire; reaching here means that check missed a mode.
            raise RuntimeError(
                "per-request GLP alpha reached a forward mode it is not wired "
                f"into ({len(alphas)} requests); the scheduler admission check "
                "is missing this mode"
            )
        return None, False
    if any(a != 0.0 and not n for a, n in zip(alphas, rows_per_req)):
        # A scheduled request always contributes rows; a steered one with none
        # would be served stock, which is the failure this mode exists to
        # prevent.
        raise RuntimeError(
            "per-request GLP alpha: a steered request contributed no token "
            "rows to this forward; the engine's batch is inconsistent"
        )
    steered = [a for a in alphas if a != 0.0]
    num_rows = sum(int(n) for n in rows_per_req)
    total = num_rows + tail_rows
    if not steered:
        return torch.zeros(total, dtype=torch.float32, device=device), False
    if not tail_rows and len(steered) == len(alphas) and len(set(steered)) == 1:
        # One alpha for the whole batch (the common steered case).
        rows = torch.full((total,), steered[0], dtype=torch.float32, device=device)
        return rows, True
    per_req = async_h2d(list(alphas), dtype=torch.float32, device=device)
    if len(set(rows_per_req)) == 1:
        # Decode and verify: one width for every request.
        rows = per_req[:, None].expand(-1, int(rows_per_req[0])).reshape(-1)
    else:
        counts = async_h2d(
            [int(n) for n in rows_per_req], dtype=torch.int64, device=device
        )
        # output_size spares repeat_interleave reading the total back.
        rows = torch.repeat_interleave(per_req, counts, output_size=num_rows)
    if tail_rows:
        rows = torch.cat([rows, rows.new_zeros(tail_rows)])
    return rows, True


def forward_alpha_rows(
    batch, *, device: torch.device
) -> Tuple[Optional[torch.Tensor], bool]:
    """``build_alpha_rows`` for one target forward of a ``ScheduleBatch``.

    Decode and verify rows use the per-request verify width, an extend
    (including a mixed batch) each request's own row count, and a decode
    batch's beam member rows follow the request rows at alpha 0: no request
    owns them, and a steered beam request is refused. The caller picks the
    device, so a batch whose graph loads its own inputs (``input_ids`` still
    None) needs nothing from them. When ``input_ids`` is there the vector
    must match it: an unsteered one that does not is rebuilt as zeros of that
    size (exact at alpha 0), and a steered one raises, since its rows would
    land on other tokens.
    """
    spec_info = batch.spec_info
    tokens_per_req = 1
    if spec_info is not None:
        tokens_per_req = int(spec_info.num_tokens_per_req)
        if tokens_per_req <= 0:
            # Only the spec inputs that carry a verify tree declare
            # draft_token_num; the base SpecInput has no such field.
            tokens_per_req = int(getattr(spec_info, "draft_token_num", 1))
    tail = batch.beam_tail
    rows, any_steered = build_alpha_rows(
        alphas=[req.glp_alpha for req in batch.reqs],
        rows_per_req=request_token_rows(
            forward_mode=batch.forward_mode,
            num_reqs=len(batch.reqs),
            extend_lens=batch.extend_lens,
            tokens_per_req=tokens_per_req,
        ),
        device=device,
        tail_rows=len(batch.seq_lens) - tail.num_base_rows if tail is not None else 0,
    )
    input_ids = batch.input_ids
    if rows is None or input_ids is None or rows.shape[0] == input_ids.shape[0]:
        return rows, any_steered
    if any_steered:
        raise RuntimeError(
            f"per-request GLP alpha: {rows.shape[0]} rows for a "
            f"{batch.forward_mode.name} forward of {input_ids.shape[0]} tokens; "
            "the steered rows cannot be placed"
        )
    import torch

    return torch.zeros(input_ids.shape[0], dtype=torch.float32, device=device), False


# The plugin's per-request arming, per process. The Weightless plugin installs
# itself in each scheduler process after the weights load and calls
# ``mark_per_request_armed``; a request with a non-zero alpha is refused while
# this is unset, so a deployment that did not load the plugin never serves a
# steered request as if it were stock.
_ARMED = False


def mark_per_request_armed(*, file_sha256: Optional[str]) -> None:
    """Called by the steering plugin once its sites are installed.

    Only the fact is kept: ``file_sha256`` names the plugin's steering file
    in its own record, and the bound on alpha is GLP_ALPHA_MAX. Arming is
    one-way for the life of the process: a second call (the plugin installs
    again only on a new model object) leaves it armed, and nothing disarms
    it, since the plugin's sites stay on the model and weight updates load in
    place.
    """
    global _ARMED
    _ARMED = True


def per_request_armed() -> bool:
    """Whether the steering plugin armed per-request mode in this process."""
    return _ARMED

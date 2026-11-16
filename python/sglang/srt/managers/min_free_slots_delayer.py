import os
import re
from typing import Mapping, Optional, Tuple

PP_DSPARK_MIN_FREE_SLOTS_ENV = "SGLANG_PP_DSPARK_MIN_FREE_SLOTS"


def resolve_min_free_slots(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
) -> Optional[int]:
    """Resolve the min-free-slots threshold (None = disabled).

    An explicit user value always wins, capped by max_running_requests
    (<= 1 disables). When unset, DFlash workloads fall back to the legacy
    formula (preserving the always-on behavior, disabled when
    max_running_requests < 8); other workloads stay disabled.
    """
    max_running_requests = max(0, int(max_running_requests))
    if user_value is not None:
        threshold = min(user_value, max_running_requests)
        return threshold if threshold > 1 else None
    if is_dflash_family and max_running_requests >= 8:
        return min(4, max(2, (max_running_requests + 5) // 6))
    return None


def resolve_min_free_slots_rule(
    user_value: Optional[int],
    max_running_requests: int,
    is_dflash_family: bool = False,
    *,
    pp_size: int = 1,
    pp_micro_batch_cap: int = 0,
    pp_value: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    """The threshold the scheduler uses (None = no hold) and why, for its log.

    Under pipeline parallelism --min-free-slots-delay is refused, but a
    DFlash-family algorithm (DSpark) still gets the default above, counted
    against all running requests while a micro-batch has only
    pp_micro_batch_cap slots: with 27 and 9, a micro-batch running 6 or more
    admits nobody. pp_value (SGLANG_PP_DSPARK_MIN_FREE_SLOTS) replaces that
    default there: <= 1 turns the hold off, N >= 2 holds for N free slots of
    the micro-batch (at most its size). Unset, or in any other launch, the
    result is resolve_min_free_slots's.
    """
    stock = resolve_min_free_slots(user_value, max_running_requests, is_dflash_family)
    if user_value is not None:
        why = f"--min-free-slots-delay {user_value}"
    elif stock is not None:
        why = (
            "the DFlash-family default min(4, max(2, "
            f"({max_running_requests} + 5) // 6))"
        )
    else:
        why = "no --min-free-slots-delay and no DFlash-family default"
    if pp_value is None:
        return stock, why
    if pp_size <= 1 or not is_dflash_family:
        return stock, (
            f"{why}; SGLANG_PP_DSPARK_MIN_FREE_SLOTS={pp_value} ignored "
            "(it applies to a DFlash-family algorithm under pipeline parallelism)"
        )
    cap = int(pp_micro_batch_cap) or max(0, int(max_running_requests))
    threshold = min(int(pp_value), cap)
    stock_text = "off" if stock is None else str(stock)
    capped = "; capped at the micro-batch size" if int(pp_value) > cap else ""
    why = (
        f"SGLANG_PP_DSPARK_MIN_FREE_SLOTS={pp_value} under pipeline parallelism "
        f"(micro-batch of {cap}{capped}; stock: {stock_text})"
    )
    return (threshold if threshold > 1 else None), why


def read_pp_min_free_slots(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[int]:
    """SGLANG_PP_DSPARK_MIN_FREE_SLOTS from the launch environment (None = unset).

    A value that is not a whole number stops the launch instead of falling back
    to the default hold, so that a typo cannot go unnoticed.
    """
    raw = (os.environ if environ is None else environ).get(PP_DSPARK_MIN_FREE_SLOTS_ENV)
    if raw is None:
        return None
    if re.fullmatch(r"[+-]?[0-9]+", raw.strip()) is None:
        raise ValueError(
            f"{PP_DSPARK_MIN_FREE_SLOTS_ENV}={raw!r} is not a whole number. "
            "Accepted values: unset (the default hold), 1 or less (no hold), or "
            "N >= 2 (hold until N slots of the micro-batch are free, at most its "
            "size)."
        )
    return int(raw.strip())


class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots

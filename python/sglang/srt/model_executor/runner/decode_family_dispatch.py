"""Two families of decode graphs on one stage, chosen by the forward's mode.

DSpark under pipeline parallelism verifies chains at their width and runs plain rounds
at one row. Everything outside these three calls reads the verify runner, the family a
speculative launch has always had.
"""

from __future__ import annotations


class DecodeFamilyDispatch:
    def __init__(self, *, verify, plain) -> None:
        self.verify = verify
        self.plain = plain

    def _family(self, forward_batch):
        return self.plain if forward_batch.forward_mode.is_decode() else self.verify

    def can_run_graph(self, forward_batch) -> bool:
        return self._family(forward_batch).can_run_graph(forward_batch)

    def load_batch(self, forward_batch, pp_proxy_tensors=None):
        return self._family(forward_batch).load_batch(forward_batch, pp_proxy_tensors)

    def execute(self, forward_batch, pp_proxy_tensors=None):
        return self._family(forward_batch).execute(
            forward_batch, pp_proxy_tensors=pp_proxy_tensors
        )

    def __getattr__(self, name):
        return getattr(self.__dict__["verify"], name)

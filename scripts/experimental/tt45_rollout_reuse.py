"""Opt-in exact reuse of the traced prefix of an ordinary elastic MPC search.

No rollout state, engine pointer, or native memory is cached. The only saved
objects are the input goals and scores of the first call in one search.
"""
from dataclasses import dataclass
from hashlib import sha256
from inspect import getsource

import numpy as np

from . import tt38_joint_mpc as mpc

STOCK_MPC = mpc.MPC
REVIEWED = {
    "search": "3f8d69065ff59562b00c7d6b1ade67db6729c1c39fbcc1038309eebccee25090",
    "trial": "bb862bc27cee953d7260cf334ba8e9f6cac471666f2aa8e0c9a69bf73a5dc5d8",
    "transport": "46d68d01e15f69239f6bb81e726db999cb5bc327db13116a8ffcc00ab01cf745",
    "pupil_prior": "b06a8782e0d5f87d12a79ec15f9a59d92b6689703f28d208a1d6170526126d0a",
}


def compatible():
    """A different search implementation may deliberately reinitialize physics."""
    try:
        return all(sha256(getsource(getattr(STOCK_MPC, name)).encode()).hexdigest() == signature
                   for name, signature in REVIEWED.items())
    except (OSError, TypeError):
        return False


@dataclass
class Prefix:
    goals: np.ndarray
    loss: np.ndarray
    legal: np.ndarray
    outcome: np.ndarray
    feedback: np.ndarray

    @classmethod
    def capture(cls, goals, results):
        loss, legal, _, outcome, feedback = results
        # The caller owns the trace and could append to its list of goals.
        return cls(np.array(goals, dtype=np.float64, copy=True), loss, legal, outcome, feedback)

    def match(self, goals):
        n = len(self.goals)
        if len(goals) < n:
            return 0
        current = np.asarray(goals[:n], dtype=np.float64)
        if current.shape != self.goals.shape or not np.array_equal(
                current.view(np.uint64), self.goals.view(np.uint64)):
            return 0
        return n

    def prepend(self, results):
        if results is None:
            return self.loss, self.legal, None, self.outcome, self.feedback
        loss, legal, _, outcome, feedback = results
        return (
            np.concatenate((self.loss, loss)),
            np.concatenate((self.legal, legal)),
            None,
            np.concatenate((self.outcome, outcome)),
            np.concatenate((self.feedback, feedback)),
        )


class ReuseMPC(STOCK_MPC):
    """Same stock planner, except it does not simulate identical prefix rows twice."""

    def __init__(self, *args, reuse=True, audit_prefix=False, **kw):
        super().__init__(*args, **kw)
        self.reuse = reuse
        self.audit_prefix = audit_prefix
        self.reviewed = compatible()
        self.open = False
        self.calls = 0
        self.prefix = None

    def search(self, tick, x, q, reference, base):
        # Explicitly run the stock implementation. In stock elastic and
        # fragment search, only pure goal construction, pose differential
        # queries (separate BladeIK data), and raw field reads separate the
        # initial trace call and its enlarged short screen. Guarded has a
        # different preamble with deliberate ball-state perturbations.
        self.open = self.reviewed and self.variant in ("elastic", "fragment") and all(
            getattr(getattr(self, name), "__func__", None) is getattr(STOCK_MPC, name)
            for name in ("transport", "pupil_prior")
        )
        self.calls = 0
        self.prefix = None
        try:
            return STOCK_MPC.search(self, tick, x, q, reference, base)
        finally:
            self.open = False
            self.calls = 0
            self.prefix = None

    def trial(self, goals, *, long=False, traces=False):
        if not self.open or not (self.reuse or self.audit_prefix):
            return super().trial(goals, long=long, traces=traces)
        self.calls += 1
        if self.calls == 1 and traces and not long:
            ans = super().trial(goals, traces=True)
            self.prefix = Prefix.capture(goals, ans)
            return ans
        if self.calls == 2 and not long and not traces and self.prefix is not None:
            cache, self.prefix = self.prefix, None
            n = cache.match(goals)
            if n:
                if self.audit_prefix:
                    full = super().trial(goals)
                    for actual, prior in zip((full[0], full[1], full[3], full[4]),
                                             (cache.loss, cache.legal, cache.outcome, cache.feedback), strict=True):
                        if actual.ndim > 0 and actual.dtype == np.float64:
                            actual, prior = actual.view(np.uint64), prior.view(np.uint64)
                        if not np.array_equal(actual[:n], prior):
                            raise ValueError("Traced prefix disagrees with the actual repeated short screen")
                    self.diag["audited_duplicate_candidates"] += n
                    return full
                extra = super().trial(goals[n:]) if n < len(goals) else None
                self.diag["reused_native_candidates"] += n
                # Existing reports counted candidate rows, including these
                # duplicated rows. Keep that logical accounting constant.
                self.diag["native_candidates"] += n
                return cache.prepend(extra)
        return super().trial(goals, long=long, traces=traces)

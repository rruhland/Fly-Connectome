from collections import Counter, deque
from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt42_direct_execution import Execution
from scripts.experimental.tt43_outgoing_coverage import Actor
from scripts.experimental.tt43_temporal_servo import Temporal
from scripts.experimental.tt46_checked_execution import CheckedCadence


def test_history_fallback_feeds_checked_cadence_and_keeps_outgoing_ranker(monkeypatch):
    assert CheckedCadence.step is Temporal.step
    assert CheckedCadence.search is Actor.search
    assert CheckedCadence.robust is Actor.robust
    calls = []

    def cadence(self, tick, eye, x, q, reference, base):
        calls.append(tick)
        return base

    monkeypatch.setattr(Execution, "step", cadence)
    actor = object.__new__(CheckedCadence)
    actor.samples = deque((np.zeros(11) for _ in range(6)), maxlen=7)
    actor.motor = SimpleNamespace(predict=lambda _: np.full((1, 5), 0.1))
    actor.diag = Counter()
    x = np.array([0.0, -0.4, 1.0, 0.0, -3.0, 0.0])
    base = np.zeros(6)
    selected = actor.step(20, None, x, base, base, base)
    np.testing.assert_equal(selected, [0.065] * 5 + [0.0])
    np.testing.assert_equal(base, np.zeros(6))
    assert calls == [20]
    assert actor.diag["temporal_inbound_steps"] == 1

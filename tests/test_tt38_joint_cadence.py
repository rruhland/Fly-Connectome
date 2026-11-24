from collections import Counter
from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt38_joint_cadence import AdaptiveClock, IMPLEMENTATION


def test_failed_clock_uses_a_new_exposure_and_stable_plan_can_wait(monkeypatch):
    def mock_step(self, tick, eye, x, q, ref, base):
        if tick >= self.next:
            self.next = tick + 5
            stable = len(self.notes) > 0
            self.notes.append(dict(chosen="warm" if stable else None, bounced=True,
                                   robust=2 if stable else 0, contact_delay_s=.2))
        return base

    monkeypatch.setattr(IMPLEMENTATION, "step", mock_step)
    model = AdaptiveClock.__new__(AdaptiveClock)
    model.next, model.notes, model.diag = 0, [], Counter()
    model.accelerated_until, model.last_search_frame = None, -np.inf
    model.best, model.tick, model.lease = None, -10, 5
    x = np.array([0., -.9, 1., 0., -3., 1.])
    q = np.zeros(6)
    eye = SimpleNamespace(latest=SimpleNamespace(time=0.))
    model.step(0, eye, x, q, q, q)
    assert model.next == 2
    model.step(2, eye, x, q, q, q)
    assert model.next == 3
    assert len(model.notes) == 1
    eye.latest.time = 2 / 60.
    model.step(3, eye, x, q, q, q)
    assert model.next == 10
    assert model.diag["fresh_burst_rechecks"] == 1

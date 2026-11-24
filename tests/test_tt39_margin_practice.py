from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experimental import tt39_margin_practice as practice


def test_borrowed_reference_phase_aligns_camera_arrival():
    p = practice.ProtectedPractice.__new__(practice.ProtectedPractice)
    trace = np.zeros((20, 6))
    trace[:, 2] = .01 * np.arange(20)
    p.samples = [dict(eta=.30, targets=trace)]
    p.known = np.zeros((1, 5))
    original = p.move(0, .30, 1/60, 13, np.zeros(5))
    aligned = p.move(0, .25, 1/60, 10, np.array([.4, 0., 0., 0., 0.]))
    np.testing.assert_allclose(aligned[:, 2], original[3:, 2], atol=1e-10)
    np.testing.assert_allclose(aligned[:, 0], .18)
    np.testing.assert_allclose(trace[:, 0], 0.)


def test_live_point_cannot_supply_preparation_model():
    with pytest.raises(ValueError, match='optical arena'):
        practice.ProtectedPractice.guard(SimpleNamespace(twin=SimpleNamespace(launches=1, card=None)))


def test_keep_chart_goal_when_stock_already_has_robust_option(monkeypatch):
    p = practice.ProtectedPractice.__new__(practice.ProtectedPractice)
    p.begin()
    p.matched = lambda opt, q: ([(0, .1)], np.zeros(10))
    p.move = lambda *args: pytest.fail('Protected stock still queried a borrowed movement')
    monkeypatch.setattr(practice.clinic, 'geometry', lambda s, o: (np.zeros(5), np.zeros(5)))
    engine = SimpleNamespace(twin=SimpleNamespace(launches=0, card=None, control_dt=1/60), tick=25,
                             previous=np.zeros(6), sync=lambda: True)
    p.predict = lambda e, goals, cutoff: (np.zeros((len(goals), 13)), None, None, np.full(len(goals), 3))
    q = np.r_[np.arange(5) * .001, -.1]
    surf = SimpleNamespace(ready=q.copy())
    np.testing.assert_equal(p.prepare(engine, surf, np.r_[.30, .1, 1., 0., -3., .4], [], q), q)
    assert p.notes[0]['status'] == 'stock protected early'
    assert p.active is None

from collections import Counter

import numpy as np
import pytest

from scripts.experimental import tt45_rollout_reuse as fast


def sample(n):
    return np.arange(n * fast.mpc.HORIZON * 6, dtype=float).reshape(n, fast.mpc.HORIZON, 6) / 300.


def result(goals):
    v = np.asarray(goals)[:, 0, 0]
    return v.copy(), v != 0, None, v[:, None].copy(), np.c_[v, v]


def test_prefix_preserves_exact_values_order_and_rejects_signed_zero():
    g = sample(5)
    copy = fast.Prefix.capture(g[:3], result(g[:3]))
    assert copy.match(g) == 3
    assert copy.match(g[:2]) == 0
    for j in (0, 1, 3, 4):
        assert np.array_equal(copy.prepend(result(g[3:]))[j], result(g)[j])
    g[0, 0, 0] = -0.0
    assert copy.match(g) == 0


def setup(monkeypatch, variant):
    calls = []
    answers = []

    def solve(self, goals, *, long=False, traces=False):
        calls.append((len(goals), long, traces))
        self.diag["native_candidates"] += len(goals)
        return result(goals)

    def search(self, _tick, x, q, ref, base):
        self.trial(self.g[:3], traces=True)
        answers.append(self.trial(self.g))
        self.trial(self.g, long=True)

    monkeypatch.setattr(fast.mpc.MPC, "trial", solve)
    monkeypatch.setattr(fast.mpc.MPC, "search", search)
    model = fast.ReuseMPC.__new__(fast.ReuseMPC)
    model.variant = variant
    model.reuse = True
    model.audit_prefix = False
    model.reviewed = True
    model.diag = Counter()
    model.g = sample(8)
    return model, calls, answers


def test_opt_in_reuses_only_once_within_search(monkeypatch):
    model, calls, answers = setup(monkeypatch, "elastic")
    model.search(0, None, None, None, None)
    assert calls == [(3, False, True), (5, False, False), (8, True, False)]
    assert model.diag["reused_native_candidates"] == 3
    assert model.diag["native_candidates"] == 19
    assert np.array_equal(result(model.g)[4], answers[0][4])
    assert not model.open
    assert model.prefix is None
    model.trial(model.g)
    assert calls[-1] == (8, False, False)


def test_guarded_search_never_reuses_after_calibration_changes(monkeypatch):
    model, calls, _ = setup(monkeypatch, "guarded")
    model.search(0, None, None, None, None)
    assert calls == [(3, False, True), (8, False, False), (8, True, False)]
    assert model.diag["reused_native_candidates"] == 0


def test_stock_audit_rejects_a_nonidentical_short_prefix(monkeypatch):
    model, _, _ = setup(monkeypatch, "elastic")
    model.reuse, model.audit_prefix = False, True

    def changing(self, goals, *, long=False, traces=False):
        out = result(goals)
        if not traces:
            out[4][0, 0] = 13.
        return out

    monkeypatch.setattr(fast.mpc.MPC, "trial", changing)
    with pytest.raises(ValueError, match="prefix"):
        model.search(0, None, None, None, None)
    assert not model.open and model.prefix is None


def test_changed_stock_source_falls_back_and_class_patch_does_not_recurse(monkeypatch):
    assert fast.compatible()
    model, calls, _ = setup(monkeypatch, "elastic")
    assert not fast.compatible()
    model.reviewed = False
    monkeypatch.setattr(fast.mpc, "MPC", fast.ReuseMPC)
    model.search(0, None, None, None, None)
    assert calls == [(3, False, True), (8, False, False), (8, True, False)]
    assert model.diag["reused_native_candidates"] == 0

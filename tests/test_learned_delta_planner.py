"""Public timing, competing event mass, compiled parameters and serialization."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
import pong_learned_delta_planner as core
from pong_smdp_adaptive_forest import AdaptiveForest, state_features
from fly_connectome.control.durable import fingerprint


def constant_forest(features, outputs, value):
    return core.CompiledForest(features, outputs, [0, 1], [0], [-2], [-2.], [-1], [-1],
        np.asarray(value, float).reshape(1, outputs))


def fake_tail():
    shape = (32, 1)
    return AdaptiveForest(np.ones(4), np.full(shape, -2), np.full(shape, -2.),
        np.full(shape, -1), np.full(shape, -1), np.full(shape, .4), np.ones((*shape, 4), int))


def model(monkeypatch):
    tail = fake_tail()
    monkeypatch.setattr(core, 'sha', lambda _:core.TAIL_SHA)
    monkeypatch.setattr(core.AdaptiveForest, 'load', lambda _:tail)
    return core.DeltaPlannerModel(np.ones(4), np.zeros(117), np.ones(117), np.ones(28),
        np.zeros((118, 28)), constant_forest(117, 28, np.zeros(28)),
        constant_forest(147, 4, [0, 0, 0, .99]), 'frozen-tail')


def one_state(phase=0):
    return state_features([[.1, -.2, .3, .4, 1, 1]], phase, True, np.ones(4))


def test_planned_phase_interval_and_actual_past_action_inputs():
    assert np.array_equal(core.scheduled_seconds(np.arange(10)) * 120, [2, 2, 3, 2, 3] * 2)
    states = np.array([one_state(1), one_state(2)])
    history = np.zeros((2, 16, 3)); history[:, -1, 0] = 1
    X, E = core.inputs(states, history, [2, 2])
    assert X.shape == (2, 117) and E.shape == (2, 147)
    assert np.array_equal(X[:, 85], [2 / 120, 3 / 120])
    assert np.array_equal(X[:, 37:85], history.reshape(2, 48))
    decoded, valid = core.decode_scene(states[:, :28], states)
    assert valid.all() and np.argmax(decoded[:, 29:], axis=1).tolist() == [2, 3]


def test_competing_hit_end_mass_absorbs_and_tail_prunes_every_depth():
    class MassModel(core.DeltaPlannerModel):
        def tail_values(self, states):
            self.calls += 1
            return np.full(len(states), .8), np.ones(len(states), bool)
        def step(self, states, histories, actions):
            n = len(states)
            return states, histories, np.full(n, .25 * .95), np.full(n, .5 * .95), np.ones(n, bool), np.full(n, .25), np.full(n, .25)
    actual = object.__new__(MassModel); actual.calls = 0
    action, decision = actual.plan(one_state(), np.zeros((16, 3)), 2)
    expected = .25 * .95 * sum((.5 * .95) ** t for t in range(8)) + (.5 * .95) ** 8 * .8
    assert action == 2 and actual.calls == 9 and decision['expanded_nodes'] == 156
    assert all(v == pytest.approx(expected) for v in decision['root_scores'])
    assert max(decision['root_scores']) <= 1


def test_compiled_output_and_serialized_plans_are_exact(tmp_path, monkeypatch):
    actual = model(monkeypatch)
    X = np.zeros((24, 117), np.float32)
    expected = actual.residual.trees[0].predict(X)[:, :, 0]
    assert np.array_equal(actual.residual.predict(X), expected)
    path = tmp_path / 'planner.npz'; actual.save(path)
    restored = core.DeltaPlannerModel.load(path)
    assert actual.plan(one_state(2), np.zeros((16, 3)), 1) == restored.plan(one_state(2), np.zeros((16, 3)), 1)


def test_fingerprint_covers_ridge_and_actual_compiled_heads_and_tail(monkeypatch):
    actual = model(monkeypatch)
    baseline = fingerprint(actual.frozen_parameters())
    actual.ridge.flags.writeable = True; actual.ridge[0, 0] += 1
    assert fingerprint(actual.frozen_parameters()) != baseline
    actual.ridge[0, 0] -= 1; actual.ridge.flags.writeable = False
    assert fingerprint(actual.frozen_parameters()) == baseline
    for tree in (actual.residual.trees[0], actual.events.trees[0], actual.compiled_tail.estimators_[0].tree_):
        old = tree.value[0, 0, 0]; tree.value[0, 0, 0] = old + 1
        assert fingerprint(actual.frozen_parameters()) != baseline
        tree.value[0, 0, 0] = old
        assert fingerprint(actual.frozen_parameters()) == baseline


def test_controller_uses_only_chosen_actual_history_after_observation(monkeypatch):
    actual = model(monkeypatch); controller = core.DeltaPlannerController(actual)
    scene = np.array([[.1, -.2, .3, .4, 1, 1]])
    action, _ = controller.choose(scene, 2)
    assert action == 2 and not controller.history.any()
    controller.observe(1)
    assert np.array_equal(controller.history[-1], [0, 0, 1]) and controller.ordinal == 1
    before = controller.history.copy(); controller.choose(scene, 1)
    assert np.array_equal(controller.history, before)

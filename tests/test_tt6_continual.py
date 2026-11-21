"""Check that an interface and its learned state survive a return visit."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt5_player_body import ANGLE_HIGH, ANGLE_LOW, FittedArm, centers
from scripts.experimental.tt6_memory import load, save
from scripts.experimental.tt6_recall import InterfaceMemory


def test_anonymous_interface_is_identified_and_remembered():
    fit = FittedArm(
        geometry=np.array([0., -1.8, .80, .39]), scripts=np.arange(4),
        home=np.array([0., -.55, 1.05, .15]), gains=np.array([.14, .13, .16, .21]),
        lags=np.array([.35, .38, .4, .12]), error=.005, samples=180, pitch_samples=160,
    )
    memory = InterfaceMemory(fit)
    true_order = np.array([1, 0, 3, 2])
    gains = fit.gains * [-1., 1., 1., -1.]
    costs = []
    for _ in range(2):
        memory.reset()
        requested = fit.home.copy()
        actual = requested.copy()
        preceding = None
        while True:
            if preceding is not None:
                requested = np.clip(requested + gains * preceding[true_order], ANGLE_LOW, ANGLE_HIGH)
                actual += (1. - fit.lags) * (requested - actual)
            seen = SimpleNamespace(observed=True, position=centers(actual, fit.geometry))
            if memory.update(preceding, seen, actual[3]) is not None:
                break
            preceding = memory.tape[memory.elapsed]
        np.testing.assert_array_equal(memory.adapter().scripts, true_order)
        np.testing.assert_array_equal(memory.adapter().gains, gains)
        costs.append(memory.elapsed)
        memory.store()
    assert costs[1] <= costs[0]


def test_continual_checkpoint_preserves_choices_dynamics_and_latency(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'artifacts/tt6/continual-player.json.gz'
    library, prior, racket, timing = load(path)
    copied = tmp_path / 'state.json.gz'
    save(copied, library, prior, racket, timing)
    library2, prior2, racket2, timing2 = load(copied)
    assert prior.state() == prior2.state()
    assert prior.choose() == prior2.choose()
    assert racket.speed == racket2.speed and racket.reference == racket2.reference
    np.testing.assert_array_equal(timing.probabilities, timing2.probabilities)
    for model, restored in zip(library.models, library2.models, strict=True):
        np.testing.assert_array_equal(model.kernels, restored.kernels)
        probe = np.linspace(-1., 1., model.action_size)
        np.testing.assert_array_equal(model.pose(probe), restored.pose(probe))


def test_long_games_balance_conditions_without_exposing_the_schedule():
    from scripts.experimental.tt6_continuous_trial import match_phases

    phases = ['ordinary', 'ordinary', 'soft', 'mixed', 'delay']
    schedule = match_phases(phases, 13, [88, 0, 9])
    assert len(schedule) == 13
    assert sorted(schedule[:5]) == sorted(schedule[5:10]) == sorted(phases)
    assert schedule == match_phases(phases, 13, [88, 0, 9])

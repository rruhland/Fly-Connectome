"""Opt-in recurrent visual state learns only from locally visible evidence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_transition_units import TransitionPopulation
from local_hidden_transition import LocalHiddenTransition
from run_local_hidden_transition import ARMS, evaluate_gaps, gap_cases


def impulse(y=16, x=16):
    current = torch.zeros((16, 32, 64))
    current[1, y, x] = 1
    return current


def test_hidden_state_advances_without_visible_input():
    model = LocalHiddenTransition(TransitionPopulation(channels=16))
    model.weights[:, :, 2, 3] = 1
    model.step(impulse())
    model.step(torch.zeros((16, 32, 64)))
    assert model.observed.sum() == 0
    assert model.state[:, 16, 17].sum() > 0


def test_visible_next_state_supplies_delayed_local_credit():
    model = LocalHiddenTransition(TransitionPopulation(channels=16), eta=1)
    model.step(impulse(), learn=True)
    model.step(impulse(x=17), learn=True)
    assert model.weights[:, :, 2, 3].sum() > 0
    assert model.weights[:, :, 0, 0].sum() == 0


def test_blank_frame_does_not_train_state_disappearance():
    model = LocalHiddenTransition(TransitionPopulation(channels=16), eta=1)
    model.weights[:, :, 2, 3] = .5
    model.step(impulse(), learn=True)
    before = model.weights.clone()
    model.step(torch.zeros((16, 32, 64)), learn=True)
    torch.testing.assert_close(model.weights, before)


def test_gap_evaluation_includes_translated_unseen_positions():
    cases = gap_cases()
    assert len(cases) == 432
    assert {row[6] for row in cases} == {(16, 32), (12, 24), (20, 40)}


def test_gap_evaluation_scores_one_translated_case():
    models = {name: LocalHiddenTransition(
        TransitionPopulation(channels=16)) for name in ARMS}
    case = next(row for row in gap_cases() if row[6] == (12, 24))
    result = evaluate_gaps(models, [case])
    assert result['occupancy'][case[0]]['1']['learned']['frames'] == 1
    assert result['motion'][case[0]]['learned']['cases'] == 1

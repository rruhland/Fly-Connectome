"""Behavior of the opt-in locally plastic visual field."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sparse_recurrent_field import SparseRecurrentField


def event(y, x, *, channel=0):
    image = torch.zeros((2, 8, 8))
    image[channel, y, x] = 1.
    return image


def centered_field():
    field = SparseRecurrentField(units=1, height=8, width=8,
                                 max_winners=1)
    field.dictionary.zero_()
    field.dictionary[0, 0, 2, 2] = 1.
    return field


def test_shared_motif_activates_at_translated_event_site():
    field = centered_field()
    first = field.step(event(2, 2))
    field.reset_state()
    second = field.step(event(4, 5))
    assert first.sum() == 1
    assert second.sum() == 1
    assert first[0, 2, 2] == 1
    assert second[0, 4, 5] == 1


def test_sensory_residual_strengthens_repeated_local_event_motif():
    field = centered_field()
    field.dictionary[0, 0, 2, 2] = .2
    field.step(event(4, 4), learn_sensory=True)
    assert field.dictionary[0, 0, 2, 2] > .2
    assert field.dictionary[0, 1].sum() == 0


def test_local_recurrent_credit_predicts_experienced_displacement():
    field = centered_field()
    left = field.step(event(4, 3)).clone()
    field.reset_state()
    right = field.step(event(4, 4)).clone()
    field.credit_transition(left, right)
    prediction = field.predict_field(left)
    assert prediction[0, 4, 4] > 0
    assert prediction[0, 4, 4] > prediction[0, 4, 2]


def test_quiet_frame_retains_field_without_new_winners():
    field = centered_field()
    active = field.step(event(4, 4)).clone()
    quiet = field.step(torch.zeros((2, 8, 8)))
    assert torch.equal(quiet, active)

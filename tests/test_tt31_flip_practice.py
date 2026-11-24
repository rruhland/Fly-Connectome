import numpy as np
import pytest

from scripts.experimental.tt31_flip_practice import infer_delay, summary


def encoder_trial(start, transport):
    ticks = np.arange(65)
    roll = -.045 * np.maximum(np.minimum(ticks - start - transport, 30), 0)
    issued = np.zeros((len(ticks), 5))
    issued[start:start + 37, 4] = -1.
    encoders = np.zeros_like(issued)
    encoders[:, 4] = roll
    return dict(issued=issued, encoders=encoders)


def test_roll_delay_is_measured_relative_to_each_issued_command():
    reference = [encoder_trial(14, 0), encoder_trial(17, 0)]
    delayed = [encoder_trial(16, 3), encoder_trial(21, 3)]
    lag, losses = infer_delay(reference, delayed)
    assert lag == 3
    assert losses[3] == 0
    assert all(error > 0 for i, error in enumerate(losses) if i != 3)


def test_delay_cannot_be_inferred_from_a_stationary_encoder():
    reference = [encoder_trial(14, 0)]
    delayed = [encoder_trial(14, 3)]
    delayed[0]['encoders'][:] = 0
    with pytest.raises(ValueError, match='observable motor response'):
        infer_delay(reference, delayed)


def test_only_consecutive_qualified_impacts_count_as_a_face_change():
    rows = [
        dict(rebounds=[dict(face=-1, impact_index=1, airborne_s=.7),
                       dict(face=1, impact_index=2, airborne_s=None)]),
        dict(rebounds=[dict(face=-1, impact_index=1, airborne_s=.6),
                       dict(face=1, impact_index=3, airborne_s=None)]),
        dict(rebounds=[dict(face=-1, impact_index=1, airborne_s=.5),
                       dict(face=-1, impact_index=2, airborne_s=None)]),
        dict(rebounds=[]),
    ]
    result = summary(rows)
    assert result['qualified_rebounds'] == 6
    assert result['with_opposite_face'] == result['alternating_pairs'] == 1
    assert result['median_first_airborne_s'] == .6

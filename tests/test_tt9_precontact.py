import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt5_player import GRAVITY
from scripts.experimental.tt9_precontact import (
    FACE_DISTANCE, ApproachMemory, context, measure_approach,
)


def optical_pass(height=-.032, lateral=.020):
    trace = []
    for tick in range(7, 21):
        time = tick - 16.
        ball_z = .45 - .080 * time - .5 * GRAVITY * time**2
        trace.append(dict(
            tick=tick,
            ball=[.09, -1.57, ball_z] if tick < 15 else None,
            blade=[.09 - lateral, -1.57 - FACE_DISTANCE + .06 * (time - .5), .45 - height],
            blade_pitch=0.,
            actor=[9., 9., 9.]))
    return trace


def test_precontact_interpolates_a_falling_ball_without_departure_frames():
    measured = measure_approach(optical_pass(), 16.)
    assert measured['crossing']
    assert measured['tick'] == pytest.approx(16.5)
    assert measured['closing_speed'] == pytest.approx(.06)
    assert measured['lateral'] == pytest.approx(.020)
    assert measured['vertical'] == pytest.approx(-.072 - GRAVITY / 4)


def test_missing_blade_sightings_do_not_fall_back_to_model_pose():
    trace = optical_pass()
    for row in trace:
        row['blade'] = None
    assert measure_approach(trace, 16.) is None


def test_an_off_edge_pass_can_teach_timing_from_a_public_loss():
    measured = measure_approach(optical_pass(lateral=.11), 16.)
    assert measured['radius'] > .11
    slow = context(.13, 1.15)
    rapid = context(.04, .25)
    memory = ApproachMemory()
    assert memory.lead(slow) == 0.
    memory.observe(slow, 0., measured, -1.)
    assert memory.lead(slow) > memory.lead(rapid)
    assert memory.lead(slow, contextual=False) == memory.lead(rapid, contextual=False)
    reloaded = ApproachMemory(**memory.state())
    np.testing.assert_array_equal(memory.coefficients(), reloaded.coefficients())


def test_gaps_without_a_reliable_crossing_do_not_teach_a_timing_direction():
    trace = optical_pass()
    for row in trace:
        if row['tick'] >= 15:
            row['blade'][1] -= .45
    measured = measure_approach(trace, 16.)
    assert measured is not None and not measured['crossing']
    memory = ApproachMemory()
    memory.observe(context(.10, 1.0), 0., measured, -1.)
    assert not memory.samples


def test_lateral_registration_uses_the_sign_of_a_missed_blade_pass():
    from scripts.experimental.tt9_precontact_registration import RegistrationMemory

    features = context(.11, .75)
    right = measure_approach(optical_pass(lateral=.11), 16.)
    left = measure_approach(optical_pass(lateral=-.11), 16.)
    memory = RegistrationMemory()
    memory.record(features, 0., 0., right, -1.)
    assert features @ memory.lateral_coefficients() > 0.
    memory.record(features, 0., 0., left, -1.)
    assert features @ memory.lateral_coefficients() == pytest.approx(0., abs=1e-12)
    restored = RegistrationMemory(**memory.state())
    np.testing.assert_array_equal(memory.lateral_coefficients(), restored.lateral_coefficients())

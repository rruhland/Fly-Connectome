import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from planar_position_arm import MAX_JOINT_SPEED, MAX_RELATIVE_SETPOINT, TICK_SECONDS, _configuration_for
from pong_button_panel import (
    FAST_GEAR,
    PRESS_RADIUS,
    PRESS_TICKS,
    RELEASE_RADIUS,
    SOURCE_GEAR,
    ButtonPanel,
    VisualButtonMotor,
    decode_panel,
    pretrain_motor,
)


def _move_toward(panel, point):
    joints = panel._arm._joints
    target = _configuration_for(point, joints[2])
    return np.clip(np.subtract(target, joints) / np.multiply(MAX_RELATIVE_SETPOINT, panel._arm.response), -1., 1.)


def _move_to(panel, point):
    for _ in range(20):
        panel.step(_move_toward(panel, point))
        if math.dist(panel.tip, point) < 1e-7:
            return
    raise AssertionError('test position was unreachable')


def test_physical_dwell_and_hysteresis_gate_paddle_output():
    panel = ButtonPanel(14)
    button = np.asarray(panel.buttons[0])
    center_to_neutral = panel.neutral - button
    annulus = button + center_to_neutral * (.5 * (PRESS_RADIUS + RELEASE_RADIUS) / np.linalg.norm(center_to_neutral))
    boundary = button + center_to_neutral * ((PRESS_RADIUS - .003) / np.linalg.norm(center_to_neutral))

    _move_to(panel, annulus)
    for _ in range(PRESS_TICKS + 1):
        assert panel.step(np.zeros(3)) == 0
    assert panel.press_events == []

    for dwell in range(1, PRESS_TICKS + 1):
        output = panel.step(_move_toward(panel, boundary))
        assert math.dist(panel.tip, button) <= PRESS_RADIUS
        assert output == (0 if dwell < PRESS_TICKS else panel.wiring[0])
    assert panel.press_events == [{'tick': panel.steps, 'button': 0, 'output': panel.wiring[0]}]

    _move_to(panel, annulus)
    assert panel.step(np.zeros(3)) == panel.wiring[0]
    _move_to(panel, panel.neutral)
    assert panel.step(np.zeros(3)) == 0
    assert len(panel.release_events) == 1


def test_hidden_wiring_and_rewiring_leave_camera_and_contact_schedule_identical():
    first = ButtonPanel(12, wiring_seed=0)
    second = ButtonPanel(12, wiring_seed=3)
    assert first.wiring == second.wiring[::-1]
    motor = VisualButtonMotor(1)
    for _ in range(14):
        frame = first.frame()
        np.testing.assert_array_equal(frame, second.frame())
        action = motor.act(frame, 0)
        assert first.step(action) == -second.step(action)
        assert first.active_button == second.active_button
    assert first.active_button == 0
    before = first.frame()
    result = first.output
    first.rewire()
    assert first.output == -result
    assert first.active_button == 0
    assert len(first.press_events) == 1
    assert first.wiring_events == [{'tick': first.steps, 'wiring': first.wiring}]
    np.testing.assert_array_equal(before, first.frame())


def test_source_speed_uses_native_tick_seconds_and_fast_equivalent_uses_original_servo_step():
    source = ButtonPanel(2, gear=SOURCE_GEAR)
    reaching = ButtonPanel(2, gear=SOURCE_GEAR, tick_seconds=TICK_SECONDS)
    fast = ButtonPanel(2, gear=FAST_GEAR)
    for command in ((1., 1., -1.), (.2, .4, -.8), (0., 1., 0.), (-.3, .2, .7)):
        before = source._arm._joints
        source.step(command)
        reaching.step(command)
        fast.step(command)
        assert np.all(np.abs(np.subtract(source._arm._joints, before)) <= np.array(MAX_JOINT_SPEED) / 120 + 1e-12)
        np.testing.assert_allclose(reaching.tip, fast.tip, atol=1e-12)


@pytest.mark.parametrize('gear', (SOURCE_GEAR, FAST_GEAR))
@pytest.mark.parametrize('warm', (False, True))
def test_rgb_motor_learns_and_releases_both_buttons(gear, warm):
    panel = ButtonPanel(44, gear=gear)
    memory = None
    if warm:
        memory, practice = pretrain_motor(507, gear=gear)
        assert practice['ticks'] == 80
        assert practice['targets_reached'] > 0
    motor = VisualButtonMotor(50, memory=memory)
    for target in (1, 0, None):
        for _ in range(30):
            frame = panel.frame()
            observation = decode_panel(frame)
            action = motor.act(frame, target, observation=observation)
            panel.step(action)
            if panel.active_button == target:
                break
        else:
            raise AssertionError('RGB motor failed to complete a physical transition')
    assert len(motor.probe_ticks) == (0 if warm else 6)
    assert motor.samples > (memory['samples'] if warm else 6)
    assert [event['button'] for event in panel.press_events] == [1, 0]
    assert [event['button'] for event in panel.release_events] == [1, 0]

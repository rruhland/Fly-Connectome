"""Check signed visual servo terms and causal repeated phase triggers."""

import numpy as np

from scripts.experimental import tt22_servo as servo


def test_relative_terms_can_follow_ignore_or_oppose_the_same_motion():
    position = np.array((.035, -.25, .2))
    velocity = np.array((-.001, -.06, 0.))
    follow = servo.relative_lateral(0., position, velocity, 8., (1.05, 1.14))
    np.testing.assert_allclose(follow, 1.05*.035 - .008*1.14)
    np.testing.assert_allclose(servo.relative_lateral(0., position, velocity, 8., (-1.05, -1.14)), -follow)
    assert servo.relative_lateral(0., position, velocity, 8., (0., 0.)) == 0.


def test_seen_outbound_crossing_rearms_only_a_finished_visual_pulse(monkeypatch):
    def optical(history, _gravity):
        _, point = history[-1]
        return point, np.array((0., .065 if point[1] > 0 else -.065, 0.))

    def approach(_law, _point, velocity):
        return (7.25, np.array((.03, -.835, .17))) if velocity[1] < 0 else None

    monkeypatch.setattr(servo, 'optical_start', optical)
    monkeypatch.setattr(servo, 'approach', approach)
    p = np.r_[np.zeros(5), np.ones(5), 16., 4., 50.]
    controller = servo.RecurrentServo(p, (0., 0.), lead=8., amplitude=1., continuous_phase=True)
    inbound, outbound = np.array((0., -.35, .18)), np.array((0., .12, .22))
    controller.command(0, inbound)
    controller.command(2, outbound)
    controller.command(3, inbound)
    assert len(controller.pulses) == 1 and controller.pulses[0]['phase_origin'] == -.75
    controller.command(8, outbound)
    controller.command(10, inbound)
    assert len(controller.pulses) == 2 and controller.pulses[1]['phase_origin'] == 9.25

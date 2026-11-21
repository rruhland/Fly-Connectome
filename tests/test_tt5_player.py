import numpy as np

from scripts.experimental.tt3_visual import Track
from scripts.experimental.tt4_transfer import ZERO, calibration_commands
from scripts.experimental.tt5_player import BallFlight, fit_arm
from scripts.experimental.tt5_player_body import BodyServo, centers


class OpticalProbe:
    def __init__(self):
        self.appearances = {}

    def update(self, frames):
        position, self.actor_pitch = frames
        self.actor = Track(0, tuple(position), (0., 0., 0.))
        return self.actor,


def test_probe_recovers_unknown_signed_wiring_including_the_wrist():
    geometry = np.array([.018, -1.97, .19, .37])
    angles = np.array([.02, -.60, 1.33, .28])
    gain = np.array([.15, -.23, .28, -.24])
    lag = np.array([.55, .68, .40, .25])
    channels = np.array([2, 3, 0, 1])
    histories = []
    for seed in range(8):
        episode = []
        command = angles.copy()
        actual = angles.copy()
        for action in np.vstack([ZERO, calibration_commands(69209 + seed, 34)]):
            command += action[channels] * gain
            actual = lag * actual + (1. - lag) * command
            episode.append((action, (centers(actual, geometry), actual[3])))
        histories.append(episode)
    fit = fit_arm(histories, OpticalProbe)
    np.testing.assert_array_equal(fit.scripts, channels)
    np.testing.assert_array_equal(np.sign(fit.gains), np.sign(gain))
    np.testing.assert_allclose(fit.ready, centers(angles, geometry), atol=.01)
    assert fit.error < .012
    servo = BodyServo(fit)
    command = servo.command(fit.ready, fit.ready, .47)
    assert abs(command[channels[3]]) <= 1.
    assert command[channels[3]] < 0.


def test_occluded_ball_coasts_through_table_bounce_and_later_expires():
    tracker = BallFlight()
    initial = np.array([.13, -.60, .22])
    for tick in range(4):
        velocity = np.array([0., -.08, -.033 - .0109 * tick])
        position = initial + np.array([0., -.08 * tick, -.033 * tick - .00545 * tick**2])
        track = Track(1, tuple(position), tuple(velocity), age=tick+1)
        estimate = tracker.update(tick, track, (.003, .008, .003), .87)
    assert estimate[1][2] < -.04
    for tick in range(4, 7):
        estimate = tracker.update(tick, None, None, .87)
    assert estimate[0][1] < -.97
    assert estimate[0][2] > .06
    assert estimate[1][2] > .015
    for tick in range(7, 12):
        estimate = tracker.update(tick, None, None, .87)
    assert estimate is None

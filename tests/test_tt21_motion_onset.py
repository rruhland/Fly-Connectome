import numpy as np

from scripts.experimental.tt21_motion_memory import OnsetDelay


def test_learns_changed_latency_from_prior_quiet_frames_and_ordinary_motion():
    learner = OnsetDelay()
    commands = np.zeros((28, 6))
    commands[7:17, 2] = 0.6
    for lag in (3, 1, 0):
        motion = np.full(27, 0.1)
        motion[7 + lag : 17 + lag] += 6
        result = learner.observe(commands, motion)
        assert result["delay"] == lag
        assert result["rendered"] == 0


def test_retains_previous_lag_when_scene_motion_has_no_identifiable_onset():
    learner = OnsetDelay()
    learner.delay = 4
    for first in (0, 8):
        commands = np.zeros((24, 6))
        commands[first:, 0] = 0.8
        result = learner.observe(commands, np.ones(23))
        assert result["status"] == "insufficient_motion"
        assert learner.delay == 4

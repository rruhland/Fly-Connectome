from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_pong_calibration import ShotCalibrationController
from button_pong_skill import PongScreen


class _ObservedPolicy:
    crossings = SimpleNamespace(lead=1.)
    ball_velocity = (-.5, 0.)
    paddle_velocity = 0.

    def begin(self, seed):
        pass

    def target(self, screen, tick):
        return .60

    def finish(self, score):
        pass


def test_arm_shot_calibration_measures_optical_paddle_execution_and_native_score():
    policy = _ObservedPolicy()
    controller = ShotCalibrationController(policy, None, 1)
    controller.begin_point()
    controller.aim(PongScreen(.51, .5, (.10, .5)), 0)
    controller.aim(PongScreen(.54, .5, None), 12)
    measured = controller.crossings[0]
    assert measured['aimed']
    assert measured['goal_y'] == pytest.approx(.591)
    assert measured['paddle_y'] == pytest.approx(.5325)
    assert controller.offset_limit < .091
    policy.ball_velocity = (.5, 0.)
    controller.aim(PongScreen(.54, .5, (.13, .5)), 20)
    controller.finish_point(-1)
    assert controller.score_margin == pytest.approx(-.003)
    assert controller.score_updates[0]['return_visible']


def test_arm_shot_guard_protects_interception_when_target_has_no_travel_room():
    policy = _ObservedPolicy()
    controller = ShotCalibrationController(policy, None, 1, guard=True)
    controller.begin_point()
    controller.aim(PongScreen(.47, .5, (.10, .5)), 0)
    controller.aim(PongScreen(.49, .5, None), 12)
    assert controller.crossings[0]['goal_y'] == pytest.approx(.50)
    assert controller.adjusted_frames == 1

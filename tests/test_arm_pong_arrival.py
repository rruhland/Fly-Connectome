import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_pong_arrival import ArrivalController
from button_pong_skill import PongScreen


class TestPolicy:
    paddle_velocity = .5
    ball_velocity = (-.5, 0.)

    def target(self, screen, tick):
        return .57


def test_near_arrival_uses_remaining_travel_and_far_arrival_brakes():
    controller = ArrivalController(TestPolicy(), None, 17, release=False)
    assert controller.aim(PongScreen(.5, .5, (.08, .6)), 12) == 1
    assert controller.aim(PongScreen(.5, .5, (.8, .6)), 16) == -1


def test_optically_held_button_changes_expected_release_position():
    policy = TestPolicy()
    policy.paddle_velocity = .15
    screen = PongScreen(.5, .5, (.75, .7))
    controller = ArrivalController(policy, None, 17, release=True)
    assert controller.aim(screen, 12) == 1
    controller.pressed = 1.
    assert controller.aim(screen, 16) == -1

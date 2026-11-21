from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_pong_camera_rate import CameraRateController
from button_pong_skill import DesiredPongPolicy, PongScreen
from pong_button_panel import ButtonPanel, VisualButtonMotor


def test_slow_arm_uses_new_pixels_and_motor_updates_only_on_screen_frames():
    panel = ButtonPanel(17)
    motor = VisualButtonMotor(2)
    policy = DesiredPongPolicy()
    controller = CameraRateController(policy, motor, 3, hold=True)
    controller.begin_point()
    screen = PongScreen(.5, .5, (.5, .5))
    first = controller.command(panel.frame(), screen, 0)
    panel.step(first.joint_command)
    for tick in (1, 2, 3):
        held = controller.command(None, None, tick)
        assert held is first
        panel.step(held.joint_command)
    assert len(motor.probe_ticks) == 1
    controller.command(panel.frame(), screen, 4)
    assert len(motor.probe_ticks) == 2
    assert controller.arm_camera_reads == 2
    assert controller.held_command_ticks == 3

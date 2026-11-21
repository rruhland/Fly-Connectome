"""Visual motor control and frame-age compensation for traveling switches."""

from collections import deque
import math

import numpy as np

from arm_plays_pong import NATIVE_HZ, SCREEN_TICKS
from arm_pong_rewire import IndependentEffects
from button_pong_skill import CONTACT_X, PongScreen
from camera_motor_learner import MotorModel
from camera_reaching_arena import CameraView
from pong_button_panel import _PanelCamera, decode_panel
from travel_button_panel import HOME_Z, MAX_TIP_Z, PitchArm, decode_travel


ZERO_COMMAND = (0., 0., 0.)
PRACTICE_SEED = 2_182_647
WRIST_METRES_PER_TICK = .0031
XY_SATURATION = np.array((.64, .64))


class CameraPipe:
    """Synchronously capture RGB and Pong at 30 Hz and deliver both after a fixed lag."""

    def __init__(self, delay_ticks):
        if type(delay_ticks) is not int or delay_ticks < 0:
            raise ValueError('camera delay must be a nonnegative native tick count')
        self.delay_ticks = delay_ticks
        self.pending = deque()
        self.captured = self.delivered = self.age_sum = 0

    def sample(self, tick, point, arm_frame, game_frame):
        if tick % SCREEN_TICKS == 0:
            self.captured += 1
            self.pending.append((tick, point, arm_frame(), game_frame()))
        if self.pending and self.pending[0][0] + self.delay_ticks == tick:
            self.delivered += 1
            self.age_sum += self.delay_ticks
            return self.pending.popleft()
        return None


def practice_motor():
    """Fit image motion with two horizontal servos on a separate, switch-free scene."""
    rng = np.random.default_rng(PRACTICE_SEED)
    home = (.585, 0.)
    arm = PitchArm(home)
    camera = _PanelCamera(CameraView.sample(rng), ((.6, -.09), (.55, .08)), home)
    motor = MotorModel.empty(2, 0)
    probes = np.array(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
    previous = decode_panel(camera.frame(arm)).tip
    for _ in range(5):
        for probe in probes[rng.permutation(len(probes))]:
            command = tuple(.58 * probe) + (0.,)
            for _ in range(SCREEN_TICKS):
                arm.step(command)
            tip = decode_panel(camera.frame(arm)).tip
            motor.observe(np.array(command[:2]), tip - previous)
            previous = tip
    return motor


class TravelController:
    """Learn two button effects from pixels and steer using only captured frames."""

    def __init__(self, policy, practice, seed, *, compensate):
        self.policy = policy
        self.compensate = compensate
        self.rng = np.random.default_rng(seed)
        self.binding = IndependentEffects(int(self.rng.integers(2)), lock_after_acquisition=False)
        self.motor = MotorModel(0, practice.gram.copy(), practice.cross.copy())
        self.motor_samples = 0
        self.previous_tip = None
        self.previous_capture = None
        self.last_visible = None
        self.issued_commands = deque([(0, np.zeros(3))], maxlen=16)
        self.game_reads = self.arm_reads = self.discarded_game_reads = 0
        self.pressed = 0.
        self.desired = 0

    def begin_point(self):
        self.policy.begin(int(self.rng.integers(2**32)))
        self.binding.reset_trace()
        self.pressed = 0.
        self.desired = 0

    def finish_point(self, score):
        self.policy.finish(score)

    def _command_integral(self, first, last):
        integral = np.zeros(3)
        end = last
        for issued, command in reversed(self.issued_commands):
            if issued < end:
                integral += (end - max(first, issued)) * command
                end = issued
            if issued <= first:
                break
        return integral

    def _track_motor(self, tip, capture):
        if self.previous_capture is not None:
            motion = self._command_integral(self.previous_capture, capture)[:2] / SCREEN_TICKS
            if np.linalg.norm(motion) > .14:
                self.motor.observe(motion, tip - self.previous_tip, discount=.97)
                self.motor_samples += 1
        self.previous_tip = tip
        self.previous_capture = capture

    def _paddle_direction(self, screen, capture, age):
        target = self.policy.target(screen, capture)
        horizon = float('inf')
        velocity = self.policy.ball_velocity
        if screen.ball is not None and velocity is not None and velocity[0] < -.15:
            horizon = max(0., (CONTACT_X - screen.ball[0]) / velocity[0] - age)
        coast = .30 * -math.expm1(-horizon / .30)
        paddle_velocity = self.policy.paddle_velocity
        now = min(.90, max(.10, screen.paddle_y + age * paddle_velocity))
        release = .052 * self.pressed * (-math.expm1(-horizon / .16))**2
        error = target - (now + coast * paddle_velocity + release)
        return int(error > .012) - int(error < -.012)

    def _wrist_and_steering(self, observation, capture, now, target):
        effects = self.motor.effects()
        pending = (self._command_integral(capture, now) if self.compensate else np.zeros(3))
        tip = observation.tip + effects @ (pending[:2] / SCREEN_TICKS)
        height = np.clip(observation.height + pending[2] * WRIST_METRES_PER_TICK, -.005, MAX_TIP_Z)
        lift = float(np.clip((HOME_Z - height) / .014, -1., 1.))
        if target is None:
            return np.array((0., 0., lift))
        error = observation.buttons[target] - tip
        distance = np.linalg.norm(error)
        if distance < .9:
            return np.array((0., 0., -1.))
        normal = effects @ effects.T + .16 * np.eye(2)
        steer = effects.T @ np.linalg.solve(normal, .87 * error)
        steer /= max(1., np.max(np.abs(steer) / XY_SATURATION))
        # Bring the wrist down while still refining the center of the cap.
        wrist = -1. if distance < 1.95 else lift
        return np.array((*steer, wrist))

    def command(self, capture, now, arm_image, game_image):
        panel = decode_travel(arm_image)
        self.arm_reads += 1
        self._track_motor(panel.tip, capture)
        if game_image is not None:
            self.game_reads += 1
            screen = PongScreen.read(game_image)
            for _ in range(SCREEN_TICKS):
                self.binding.advance(self.last_visible)
            if screen.ball is not None and screen.ball[0] > .10:
                self.binding.observe(screen.paddle_y, capture)
            deepest = int(np.argmax(panel.depth))
            self.last_visible = deepest if panel.depth[deepest] >= .0025 else None
            down = self.binding.down_button
            sign = 0 if self.last_visible is None or down is None else (1 if down == self.last_visible else -1)
            self.pressed += .35 * (sign - self.pressed)
            age = (now - capture) / NATIVE_HZ if self.compensate else 0.
            self.desired = self._paddle_direction(screen, capture, age)
        else:
            self.discarded_game_reads += 1
            self.last_visible = None
        target = self.binding.resolve(self.desired)
        command = self._wrist_and_steering(panel, capture, now, target)
        self.issued_commands.append((now, command))
        return tuple(command)

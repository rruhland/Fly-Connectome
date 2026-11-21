"""Play native Pong through camera-directed presses on a three-joint arm panel."""

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import math
import time

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong
from return_metrics import ReturnCounter


NATIVE_HZ = 120
SCREEN_TICKS = 4
SCREEN_SIZE = 192
DEFAULT_FIRST_TO = 5
DEFAULT_TICKS = 35 * NATIVE_HZ


@lru_cache(maxsize=16)
def _raster_axis(size):
    return (np.arange(size, dtype=np.float32) + .5) / size


def screen_frame(game, index=0, size=SCREEN_SIZE):
    """Render the binary native camera with the same float32 pixel predicates."""
    axis = _raster_axis(size)
    physics = game.config
    paddle_rows = (float(game.body.position[index]), float(game.opponent[index]))
    ball_x, ball_y = game.ball[index].tolist()
    ball_radius = max(physics.ball_radius, .5 / size)
    image = np.zeros((size, size), dtype=np.bool_)
    for x, y, x_radius, y_radius in (
        (physics.paddle_x, paddle_rows[0], .0125, physics.paddle_half),
        (1 - physics.paddle_x, paddle_rows[1], .0125, physics.paddle_half),
        (ball_x, ball_y, ball_radius, ball_radius),
    ):
        columns = np.flatnonzero(abs(axis - x) <= x_radius)
        rows = np.flatnonzero(abs(axis - y) <= y_radius)
        if len(rows) and len(columns):
            image[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1] = True
    return image


@dataclass(frozen=True)
class MatchCase:
    pong_seed: int
    panel_seed: int
    wiring_seed: int
    controller_seed: int


class VisualButtonBinding:
    """Infer which visible button drives screen-down from signed optical motion.

    A short two-stage trace of visible button occupation handles delayed paddle
    response. Its sign is compared to measured paddle displacement over a
    sliding screen window. Old evidence decays so swapped wiring can be relearned.
    """

    def __init__(self, guess, *, learn=True):
        if guess not in (0, 1):
            raise ValueError('binary visible button guess required')
        self.guess = guess
        self.learn = learn
        self.down_button = None
        self.evidence = 0.
        self.events = []
        self.samples = 0
        self.reset_trace()

    def reset_trace(self):
        self._contact_trace = 0.
        self._motion_trace = 0.
        self._displacement = 0.
        self._history = deque(maxlen=6)

    def advance(self, visual_button):
        signed_occupation = 0 if visual_button is None else 1 - 2 * visual_button
        self._contact_trace += .10 * (signed_occupation - self._contact_trace)
        self._motion_trace += .035 * (.74 * self._contact_trace - self._motion_trace)
        self._displacement += self._motion_trace / NATIVE_HZ

    def observe(self, paddle, tick):
        self._history.append((paddle, self._displacement))
        if not self.learn or len(self._history) < 3:
            return
        old_paddle, old_trace = self._history[0]
        displacement = paddle - old_paddle
        trace = self._displacement - old_trace
        if abs(trace) < .004 or abs(displacement) < .003:
            return
        # Border clipping can attenuate traces. A clipped window's direction
        # remains informative while the paddle is observably moving.
        agreement = max(-1., min(1., math.copysign(1., trace) * displacement / abs(trace)))
        self.evidence = .65 * self.evidence + .35 * agreement
        self.samples += 1
        choice = 0 if self.evidence >= .46 else 1 if self.evidence <= -.46 else None
        if choice is not None and choice != self.down_button:
            self.down_button = choice
            self.events.append(dict(tick=tick, down_button=choice, evidence=self.evidence))

    def resolve(self, direction):
        if direction == 0:
            return None
        down_button = self.guess if self.down_button is None else self.down_button
        return down_button if direction > 0 else 1 - down_button


@dataclass(frozen=True)
class ArmDecision:
    joint_command: tuple[float, float, float]
    desired_direction: int
    target: int | None
    visual_button: int | None
    inferred_down_button: int | None
    probing: bool


class ArmPongController:
    """Use only two raster feeds, issued commands and the natural point score."""

    def __init__(self, policy, motor, seed, *, learn_binding=True, initial_probe=True):
        self.policy = policy
        self.motor = motor
        self._seeds = np.random.default_rng(seed)
        self.binding = VisualButtonBinding(int(self._seeds.integers(2)), learn=learn_binding)
        self.initial_probe = initial_probe
        self.desired_direction = 0
        self._last_screen = None
        self._probe_button = self.binding.guess
        self._probe_start = 0
        self._probe_ticks = 0
        self._visual_button = None
        self._visual_candidate = None

    @property
    def probe_ticks(self):
        return self._probe_ticks

    def begin_point(self):
        self.policy.begin(int(self._seeds.integers(2**32)))
        self.binding.reset_trace()
        self._last_screen = None
        self.desired_direction = 0

    def finish_point(self, score):
        self.policy.finish(score)

    def aim(self, screen, tick):
        return self.policy.action(screen, tick)

    def steer_target(self, panel, screen, tick, desired_direction, target, inferred_down_button):
        return target

    def occupied_button(self, panel):
        from pong_button_panel import VISUAL_BUTTON_RADIUS, VISUAL_RELEASE_RADIUS

        errors = panel.buttons - panel.tip
        distances = np.einsum('ij,ij->i', errors, errors)
        active = self._visual_button
        if active is not None and distances[active] <= VISUAL_RELEASE_RADIUS**2:
            return active
        near = int(distances.argmin())
        candidate = near if distances[near] <= VISUAL_BUTTON_RADIUS**2 else None
        self._visual_button = candidate if candidate == self._visual_candidate else None
        self._visual_candidate = candidate
        return self._visual_button

    def command(self, arm_frame, screen, tick):
        from pong_button_panel import decode_panel

        panel = decode_panel(arm_frame)
        visual_button = self.occupied_button(panel)
        self.binding.advance(visual_button)
        if screen is not None:
            # The binary paddle and ball can share pixels near the left
            # edge. Use only samples with independently visible entities.
            if screen.ball is not None and screen.ball[0] > .10:
                self.binding.observe(screen.paddle_y, tick)
            self.desired_direction = int(self.aim(screen, tick))
            self._last_screen = screen
        target = self.binding.resolve(self.desired_direction)
        probing = self.initial_probe and self.binding.learn and self.binding.down_button is None
        if probing:
            if tick - self._probe_start >= 36:
                self._probe_button = 1 - self._probe_button
                self._probe_start = tick
            target = self._probe_button
            self._probe_ticks += 1
        else:
            target = self.steer_target(panel, self._last_screen, tick, self.desired_direction,
                                       target, self.binding.down_button)
        commands = self.motor.act(arm_frame, target, observation=panel)
        return ArmDecision(tuple(commands), self.desired_direction, target, visual_button,
                           self.binding.down_button, probing)


def _point_result(counter, start, end, score, last_touch, initial):
    return dict(start_tick=start, end_tick=end, physics_ticks=end - start,
                score=int(score), completed=bool(score), own_returns=counter.returns,
                own_opportunities=counter.opportunities, incoming=counter.first_incoming,
                maximum_streak=counter.maximum_streak, opponent_returns=counter.opponent_returns,
                opponent_opportunities=counter.opponent_opportunities,
                scored_after_own_return=bool(score > 0 and counter.returns),
                last_touch_score=bool(score > 0 and last_touch == 'own'),
                serve=initial)


def run_match(case, policy, motor, *, first_to=DEFAULT_FIRST_TO, max_ticks=DEFAULT_TICKS,
              opponent_speed=.45, panel_kwargs=None, contact_enabled=True, stationary=False,
              learn_binding=True, initial_probe=True, rewire_ticks=(), controller=None,
              callback=None, record=False):
    """Play a real first-to-N native match with a hard optional native clock.

    The callback receives public camera samples and separate evaluator telemetry.
    The same physical pose, motor memory, wiring belief and clock cross points.
    Contact disable and stationary controls still step the same game at 120 Hz.
    """
    from button_pong_skill import PongScreen
    from pong_button_panel import ButtonPanel

    if type(first_to) is not int or first_to < 1 or type(max_ticks) is not int or max_ticks < 1:
        raise ValueError('positive first-to score and native tick cap required')
    rewires = set(rewire_ticks)
    if any(type(tick) is not int or tick < 1 or tick >= max_ticks for tick in rewires):
        raise ValueError('rewires must be scheduled inside the match clock')
    started = time.monotonic()
    game = Pong([case.pong_seed], config=Physics(opponent_speed=opponent_speed))
    panel = ButtonPanel(case.panel_seed, wiring_seed=case.wiring_seed, tick_seconds=1/NATIVE_HZ,
                        **({} if panel_kwargs is None else panel_kwargs))
    controller = controller if controller is not None else ArmPongController(
        policy, motor, case.controller_seed, learn_binding=learn_binding, initial_probe=initial_probe)
    commands = {drive: torch.tensor([-float(drive)]) for drive in (-1, 0, 1)}
    arm_actions, physical_drives = sha256(), sha256()
    points, contacts, rewirings, return_ticks, trajectory = [], [], [], [], []
    count = ReturnCounter()
    points_won = points_lost = 0
    start_tick = 0
    start_screen = tuple(game.ball[0].tolist() + game.ball_velocity[0].tolist())
    last_touch = None
    screen_samples = joint_actions = physical_motion_ticks = intended_motion_ticks = 0
    last_active = None
    initial_down = panel.wiring.index(1)
    controller.begin_point()
    for tick in range(max_ticks):
        if tick in rewires:
            panel.rewire()
            rewirings.append(dict(tick=tick, down_button=panel.wiring.index(1)))
        camera_tick = (tick - start_tick) % SCREEN_TICKS == 0
        screen_raster = screen_frame(game) if camera_tick else None
        screen = PongScreen.read(screen_raster) if camera_tick else None
        screen_samples += int(camera_tick)
        arm_frame = panel.frame()
        if stationary:
            decision = ArmDecision((0., 0., 0.), 0, None, None, None, False)
        else:
            decision = controller.command(arm_frame, screen, tick)
            joint_actions += int(any(decision.joint_command))
        panel_drive = panel.step(decision.joint_command)
        drive = panel_drive if contact_enabled and not stationary else 0
        arm_actions.update(np.asarray(decision.joint_command, dtype='<f8').tobytes())
        physical_drives.update(bytes((drive + 1,)))
        physical_motion_ticks += int(drive != 0)
        intended_motion_ticks += int(decision.desired_direction != 0)
        active = panel.active_button
        if active != last_active:
            contacts.append(dict(tick=tick + 1, button=active, drive=int(panel_drive)))
            last_active = active
        before_own = count.opportunities
        before_opponent = count.opponent_returns
        outcome = count.step(game, game.step, commands[drive], auto_reset=False)
        own_hit = int(outcome.hits[0])
        score = int(outcome.scores[0])
        if own_hit:
            last_touch = 'own'
            return_ticks.append(tick + 1)
        elif count.opponent_returns > before_opponent:
            last_touch = 'opponent'
        if callback is not None or record:
            row = dict(tick=tick, physics_seconds=(tick + 1) / NATIVE_HZ,
                       action=list(decision.joint_command), desired_direction=decision.desired_direction,
                       target=decision.target, visual_button=decision.visual_button,
                       inferred_down_button=decision.inferred_down_button, probing=decision.probing,
                       active_button=active, panel_drive=int(panel_drive), drive=int(drive),
                       hit=own_hit, opportunity=count.opportunities > before_own,
                       score=score, point_start=tick == start_tick, camera_tick=camera_tick)
            if record:
                trajectory.append(row)
            if callback is not None:
                callback(dict(row, screen=screen_raster, arm=arm_frame, wall_seconds=time.monotonic() - started))
        if score or tick + 1 == max_ticks:
            points.append(_point_result(count, start_tick, tick + 1, score, last_touch, start_screen))
            controller.finish_point(score)
            points_won += int(score > 0)
            points_lost += int(score < 0)
            if max(points_won, points_lost) >= first_to or tick + 1 == max_ticks:
                break
            game.reset(torch.ones(1, dtype=torch.bool))
            start_tick = tick + 1
            start_screen = tuple(game.ball[0].tolist() + game.ball_velocity[0].tolist())
            last_touch = None
            count = ReturnCounter()
            controller.begin_point()

    first_incoming = next((point['incoming'] for point in points if point['incoming'] is not None), None)
    first_point = next((point for point in points if point['incoming'] is not None), None)
    result = dict(case=asdict(case), first_to=first_to, max_ticks=max_ticks, native_hz=NATIVE_HZ,
                  screen_ticks=SCREEN_TICKS, opponent_speed=opponent_speed, panel_gear=panel.gear,
                  ticks=tick + 1, seconds=(tick + 1) / NATIVE_HZ,
                  wall_seconds=time.monotonic() - started, initial_down_button=initial_down,
                  rewirings=rewirings, binding_events=controller.binding.events,
                  binding_samples=controller.binding.samples, probe_ticks=controller.probe_ticks,
                  arm_motor_probe_ticks=len(getattr(motor, 'probe_ticks', ())),
                  screen_samples=screen_samples, joint_motion_commands=joint_actions,
                  physical_motion_ticks=physical_motion_ticks, intended_motion_ticks=intended_motion_ticks,
                  first_incoming=first_incoming, first_incoming_match_tick=None if first_point is None
                  else first_point['start_tick'] + first_incoming['tick'],
                  points_won=points_won, points_lost=points_lost,
                  winner='own' if points_won >= first_to else 'opponent' if points_lost >= first_to else None,
                  own_opportunities=sum(point['own_opportunities'] for point in points),
                  own_returns=sum(point['own_returns'] for point in points),
                  scored_after_own_return=sum(point['scored_after_own_return'] for point in points),
                  last_touch_scores=sum(point['last_touch_score'] for point in points),
                  completed_points=points_won + points_lost, points=points, contacts=contacts,
                  return_ticks=return_ticks,
                  arm_actions_sha256=arm_actions.hexdigest(), drives_sha256=physical_drives.hexdigest())
    if record:
        result['trajectory'] = trajectory
    return result

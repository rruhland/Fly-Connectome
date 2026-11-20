"""Visual interception and public goal learning for air hockey."""

from dataclasses import asdict, dataclass
import math
import random

import numpy as np

from air_hockey import (
    ACTIONS, HEIGHT, HZ, MALLET_RADIUS, OPPONENT_COLOR, PIXELS_PER_UNIT, PLAYER_COLOR,
    PUCK_COLOR, PUCK_RADIUS, WIDTH,
)
from arcade_wall_learn import WallFlight, WallMemory, project_crossing


BOUNDS = (PUCK_RADIUS, HEIGHT - PUCK_RADIUS)
STRIKE_PLANE = .35
REACH = MALLET_RADIUS + PUCK_RADIUS
# Target at the visible goal center, either edge, or the mirrored image of an
# edge beyond a side rail. Values are learned from the resulting public score.
AIMS = ((0., 0), (-.085, 0), (.085, 0), (-.075, -1), (.075, 1))
CONTEXTS = 9
SCHEMA = 'air-hockey-goal-memory-v1'


@dataclass(frozen=True)
class HockeyImage:
    player: tuple[float, float] | None
    rival: tuple[float, float] | None
    puck: tuple[float, float] | None


def read_camera(frame):
    """Locate the three visible colors. All positions come from image pixels."""
    red = frame[:, :, 0]

    def center(color):
        y, x = np.nonzero(red == color[0])
        if not len(x):
            return None
        return (float(x.mean() + .5) / PIXELS_PER_UNIT,
                float(y.mean() + .5) / PIXELS_PER_UNIT)

    return HockeyImage(center(PLAYER_COLOR), center(OPPONENT_COLOR), center(PUCK_COLOR))


class GoalMemory:
    """Learn the result of visible striking choices in similar approaches."""

    def __init__(self, *, evidence=None, goals=None):
        shape = (CONTEXTS, len(AIMS))
        self.evidence = np.zeros(shape, dtype=int) if evidence is None else np.asarray(evidence, dtype=int).copy()
        self.goals = np.zeros(shape, dtype=float) if goals is None else np.asarray(goals, dtype=float).copy()
        if (self.evidence.shape != shape or self.goals.shape != shape or np.any(self.evidence < 0)
                or not np.isfinite(self.goals).all() or np.any(abs(self.goals) > self.evidence)):
            raise ValueError('invalid goal memory')

    def to_state(self):
        return {'schema': SCHEMA, 'evidence': self.evidence.tolist(), 'goals': self.goals.tolist()}

    @classmethod
    def from_state(cls, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('unknown goal memory schema')
        return cls(evidence=state['evidence'], goals=state['goals'])

    def choose(self, context, rng, explore):
        n = self.evidence[context] + .45 * self.evidence.sum(axis=0)
        score = self.goals[context] + .45 * self.goals.sum(axis=0)
        value = score / (n + 4.)
        if explore and rng.random() < explore:
            return rng.choices(range(len(AIMS)), weights=1 / np.sqrt(n + 1.))[0]
        contenders = np.flatnonzero(value == value.max())
        return int(rng.choice(contenders))

    def observe(self, context, arm, reward):
        self.evidence[context, arm] += 1
        self.goals[context, arm] += reward


@dataclass
class Aim:
    arm: int
    context: int
    frame: int
    contact_frame: int | None = None


def _sign(value, dead_zone):
    return 1 if value > dead_zone else -1 if value < -dead_zone else 0


class HockeyPlayer:
    """Plan an interception from RGB history and update choices on public goals."""

    def __init__(self, wall, goals=None, *, seed=0, learn_wall=False, learn_goals=False,
                 explore=0., aim=True, fixed_center=False):
        self.wall = wall
        self.goals = GoalMemory() if goals is None else goals
        self.flight = WallFlight(wall, BOUNDS, position_noise=.00165)
        self.learn_wall = learn_wall
        self.learn_goals = learn_goals
        self.explore = explore
        self.aim = aim
        self.fixed_center = fixed_center
        self.rng = random.Random(seed)
        self.image = None
        self.frame = 0
        self.shots = []
        self.plan = self.last_contact = None
        self.last_incoming = None
        self.bounces = []
        self.credited = None
        self.rival_context = None

    def observe(self, frame, reward=0, terminated=False, truncated=False):
        self.frame += 1
        self.image = read_camera(frame)
        if self.rival_context is None and self.image.rival is not None:
            self.rival_context = 1 + _sign(self.image.rival[1] - HEIGHT / 2, .037)
        bounce = self.flight.observe(self.image.puck, elapsed=1 / HZ, learn=self.learn_wall)
        if bounce is not None:
            self.bounces.append(asdict(bounce))
        velocity = self.flight.velocity
        if velocity is not None and self.image.player is not None:
            if velocity[0] < -.15 and self.image.puck[0] < .56:
                if math.dist(self.image.puck, self.image.player) < REACH + .048:
                    self.last_incoming = self.frame
            if (abs(velocity[0]) <= .15 and self.image.puck[0] < .52
                    and math.dist(self.image.puck, self.image.player) < REACH + .028):
                self.last_incoming = self.frame
            if (velocity[0] > .19 and self.last_incoming is not None
                    and self.frame - self.last_incoming <= 8 and self.plan is not None):
                self.plan.contact_frame = self.frame
                self.last_contact, self.plan = self.plan, None
                self.last_incoming = None
            if (velocity[0] < -.16 and self.last_contact is not None and self.image.rival is not None
                    and math.dist(self.image.puck, self.image.rival) < REACH + .048):
                self._credit(0)
        if terminated or truncated:
            self._credit(reward)

    def _credit(self, reward):
        if self.learn_goals and self.last_contact is not None:
            self.goals.observe(self.last_contact.context, self.last_contact.arm, reward)
            self.credited = asdict(self.last_contact)
        self.last_contact = None

    def action(self):
        if self.image.player is None or self.image.puck is None:
            return 0
        x, y = self.image.puck
        player_x, player_y = self.image.player
        velocity = self.flight.velocity
        vx, vy = velocity if velocity is not None else (0., 0.)
        target_x, target_y = .24, HEIGHT / 2
        if vx < -.10:
            target_x, target_y = self._intercept(x, y, vx, vy, player_x)
        elif self.frame > 6 and x < .52 and abs(vx) <= .10:
            target_x, target_y = self._approach_slow_puck(x, y, player_x, player_y)
        elif vx > .10 and self.plan is not None and self.frame - self.plan.frame > 12:
            self.plan = None
        return ACTIONS.index((_sign(target_x - player_x, .009),
                              _sign(target_y - player_y, .0075)))

    def _new_aim(self, crossing):
        lateral = min(2, max(0, int(crossing / (HEIGHT / 3))))
        rival = 1 if self.rival_context is None else self.rival_context
        context = 3 * rival + lateral
        arm = 0 if self.fixed_center else self.goals.choose(context, self.rng, self.explore)
        self.plan = Aim(arm, context, self.frame)
        self.shots.append(self.plan)

    def _normal(self, crossing, plane, vx, vy, lunge):
        arm = 0 if self.plan is None else self.plan.arm
        offset, bank = AIMS[arm]
        lane = HEIGHT / 2 + offset
        if bank:
            rail = BOUNDS[0] if bank < 0 else BOUNDS[1]
            lane = rail - (lane - rail) / max(.35, self.wall.coefficient)
        desired_angle = math.atan2(lane - crossing, WIDTH - plane)
        if not self.aim:
            return 0.
        desired_x, desired_y = math.cos(desired_angle), math.sin(desired_angle)
        low, high = -.85, .85
        for _ in range(8):
            angle = (low + high) / 2
            nx, ny = math.cos(angle), math.sin(angle)
            impulse = -1.95 * ((vx - lunge) * nx + vy * ny)
            out_x, out_y = vx + impulse * nx, vy + impulse * ny
            if out_y * desired_x > out_x * desired_y:
                high = angle
            else:
                low = angle
        return (low + high) / 2

    def _approach_slow_puck(self, x, y, player_x, player_y):
        if self.plan is None:
            self._new_aim(y)
        normal = self._normal(y, x, 0., 0., .98)
        lateral = y - REACH * math.sin(normal)
        back = x - REACH * math.cos(normal) - .008
        if player_x > x:
            return max(MALLET_RADIUS, back), y + (.12 if y < HEIGHT / 2 else -.12)
        if abs(player_y - lateral) < .016:
            return min(WIDTH / 2 - MALLET_RADIUS, x + .032), lateral
        return max(MALLET_RADIUS, back), lateral

    def _intercept(self, x, y, vx, vy, player_x):
        # Stay behind the point of impact. A late recovery uses a deeper
        # interception plane before it elects to fall back to the own goal.
        plane = min(STRIKE_PLANE, max(.12, x - .014))
        if x < .12 and player_x > x:
            return max(MALLET_RADIUS, x - .065), y
        crossing = project_crossing((x, y), (vx, vy), plane, BOUNDS, self.wall.coefficient)
        if crossing is None:
            crossing = y
        # Estimate each side of the arrival ray, including possible bank shots.
        next_y = project_crossing((x, y), (vx, vy), plane - .015, BOUNDS, self.wall.coefficient)
        terminal_vy = vy if next_y is None else (next_y - crossing) * -vx / .015
        selection_time = (x - plane) / -vx
        if self.plan is None:
            if selection_time < .025 or self.last_contact is None or self.frame - self.last_contact.contact_frame > 10:
                self._new_aim(crossing)
        normal = self._normal(crossing, plane, vx, terminal_vy, .98)
        target_x = plane - REACH * math.cos(normal)
        target_y = min(HEIGHT - MALLET_RADIUS, max(MALLET_RADIUS, crossing - REACH * math.sin(normal)))
        player_y = self.image.player[1]
        if .004 < x - plane < .022 and abs(player_y - target_y) < .014 and self.aim:
            target_x += .027
        return target_x, target_y

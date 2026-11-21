"""Learn return depth from stereo departures, table bounces and point reward."""
from dataclasses import asdict, dataclass, field
from math import log, sqrt

import numpy as np

from .tt5_player import BALL_RADIUS, DRAG, GRAVITY, NET_CLEARANCE, TABLE_END, TABLE_HALF_WIDTH
from .tt9_precontact_registration import RegistrationPlayer
from .tt11_stroke import VisualReply, _visible_bounce
from .tt12_online_timing import optical_role


DEPTHS = (.76, .27, 1.16)
ANGLES = np.linspace(-.02, .72, 51)
NORMALS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
TANGENTS = np.column_stack((-NORMALS[:, 1], NORMALS[:, 0]))


def depth_pitch(position, incoming, speed, restitution, correction, depth, anchor):
    """Change the landing target while keeping the established contact plane."""
    relative = incoming[1:] - (speed, 0.)
    impulses = ((1 + restitution) * (NORMALS @ relative)[:, None] * NORMALS
                + .23 * (TANGENTS @ relative)[:, None] * TANGENTS)
    out, rise = (incoming[1:] - impulses + correction).T
    out = np.maximum(.02, out)
    travel = np.array([-position[1], depth - position[1]])[:, None]
    loss = DRAG * out
    times = np.expm1(DRAG * travel) / loss
    decay = -np.expm1(-loss * times) / loss
    height = position[2] + (rise + GRAVITY / loss) * decay - GRAVITY / loss * times
    cost = abs(height[1] - BALL_RADIUS) + 2.5 * np.maximum(0., NET_CLEARANCE - height[0])
    return float(np.clip(ANGLES[np.argmin(cost)], max(-.02, anchor - .14), min(.72, anchor + .14)))


def visible_landing(trace, contact):
    """Use a visible rebound or an outbound sighting close to the table surface."""
    bounce = _visible_bounce(trace, contact)
    if bounce is not None:
        return dict(bounce=bounce, evidence='rebound')
    points = [(row['tick'], row['ball']) for row in trace if row['ball'] is not None
              and contact + 3 <= row['tick'] <= contact + 33]
    for i, (tick, ball) in enumerate(points):
        x, y, z = ball
        if not .06 < y < TABLE_END + .035 or abs(x) >= TABLE_HALF_WIDTH or not -.02 < z < .070:
            continue
        before = []
        for t, point in points[:i]:
            dt = tick - t
            if (0 < dt <= 12 and abs(x - point[0]) < .056 + .002 * dt
                    and .026 < (y - point[1]) / dt < .205
                    and .022 < (point[2] - z) / dt < .145):
                before.append((t, point))
        if len(before) < 2 or tick - before[-1][0] > 8:
            continue
        # Two pre-landing sightings must also connect to one another. This
        # excludes lone reflections near the table that follow a real deep arc.
        for (t, early), (u, late) in zip(before, before[1:]):
            if (.023 < (late[1] - early[1]) / (u - t) < .19
                    and -.025 < (early[2] - late[2]) / (u - t) < .12
                    and late[2] > .13):
                bounce = dict(tick=tick, ball=ball, before=[u, late], after=None)
                return dict(bounce=bounce, evidence='surface')
    return None


@dataclass
class PlacementMemory:
    observations: list = field(default_factory=list)

    def state(self):
        return asdict(self)

    def values(self, use_optics=True):
        direct, placement = [], []
        measured = [(row['depth'], row['win']) for row in self.observations
                    if row['depth'] is not None and row['win'] is not None]
        for action, target in enumerate(DEPTHS):
            observed = [row for row in self.observations if row['action'] == action]
            wins = [row['win'] for row in observed if row['win'] is not None]
            direct.append((sum(wins) + 1.5) / (len(wins) + 3))
            # Seed the expected landing distribution with one nominal shot.
            # Missing sightings give weak evidence against reliable placement.
            landings = [target, *(row['depth'] for row in observed if row['depth'] is not None)]
            expected = []
            for depth in landings:
                weights = [max(0., 1. - abs(depth - at) / .42) for at, _ in measured]
                expected.append((1. + sum(w * win for w, (_, win) in zip(weights, measured, strict=True))) /
                                (2. + sum(weights)))
            placement.append(sum(expected) / (len(expected) + .30 * (len(observed) + 1 - len(expected))))
        return [(.55 * v + .45 * p) if use_optics else v for v, p in zip(direct, placement, strict=True)]

    def choose(self, policy):
        if policy in ('fixed', 'center'):
            return 0
        counts = [sum(row['action'] == i for row in self.observations) for i in range(len(DEPTHS))]
        if policy == 'online' and min(counts) < 4:
            return counts.index(min(counts))
        values = self.values(use_optics=policy != 'reward')
        if policy == 'online':
            values = [value + .10 * sqrt(log(1 + sum(counts)) / (1 + n)) for value, n in zip(values, counts, strict=True)]
        return max(range(len(DEPTHS)), key=lambda i: values[i])

    def observe(self, choices, reward):
        for index, shot in enumerate(choices):
            win = int(reward > 0) if reward and index == len(choices) - 1 else None
            landing = shot['landing']
            depth = None if landing is None else landing['bounce']['ball'][1]
            self.observations.append(dict(action=shot['action'], depth=depth, win=win))
        self.observations[:] = self.observations[-144:]


class PlacementPlayer(RegistrationPlayer):
    """Keep the registered toss controller and vary only receiving pitch."""

    placement_policy = 'fixed'

    def reset(self):
        super().reset()
        self.visual_role = None
        self.reply_observer = VisualReply()
        self.shots = []
        self.approach_seen = None

    def _stroke_goals(self, estimate):
        tick = self.scene.tick
        current = self.scene.tracker.ball
        if self.visual_role is None:
            self.visual_role = optical_role(current, tick)
        if self.serve_plan is not None and current is not None and current.observed:
            self.reply_observer.observe(tick, current.position, self.serve_plan[0])
        receiving = (self.visual_role == 'receive' or self.reply_observer.tick is not None)
        self.approach_seen = estimate if receiving else None
        return super()._stroke_goals(estimate)

    def _strike_pitch(self, position, incoming):
        anchor = super()._strike_pitch(position, incoming)
        if self.approach_seen is None or not self._wrist_available():
            return anchor
        seen, velocity = self.approach_seen
        tick = self.scene.tick
        shot = self.shots[-1] if self.shots and tick < self.shots[-1]['contact'] + 24 else None
        if shot is None and -1.17 < seen[1] < -.45 and velocity[1] < -.038:
            action = self.placement.choose(self.placement_policy)
            shot = dict(tick=tick, action=action, contact=None, last_pitch=float(anchor))
            self.shots.append(shot)
        if shot is None:
            return anchor
        horizon, _, _ = self.flight.crossing(seen, velocity, position[1], self._surface_ratio())
        if shot['contact'] is None or 0 <= horizon < 3:
            shot['contact'] = float(tick + horizon)
        if shot['action'] == 0:
            return anchor
        pitch = depth_pitch(position, incoming, self.stroke_speed, self.impact_restitution,
                            self.memory.correction(), DEPTHS[shot['action']], anchor)
        shot['last_pitch'] = pitch
        return pitch

    def finish(self, reward):
        details = super().finish(reward)
        choices = self.shots
        for shot in choices:
            shot['landing'] = visible_landing(self.trace, shot['contact'])
        if self.placement_policy == 'online':
            self.placement.observe(choices, reward)
        return dict(details, return_placement=dict(role=self.visual_role, reply=self.reply_observer.tick,
                                                  choices=choices))

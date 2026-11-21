"""Keep a camera-only return estimate through a short wrist sightline gap."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .tt19_factory import HOME, RATE, FactoryModel
from .tt19_model_rollout import observed_ball
from .tt19_shoot import HORIZON
from .tt20_curriculum import CLAMP_SHIFT
from .tt20_mono_flight import MonoFlight


@dataclass(frozen=True)
class ReplyEstimate:
    position: np.ndarray
    velocity: np.ndarray
    observed_tick: int


class ReplyTrack:
    """Isolate an inbound arc only after seeing the previous stroke travel out."""

    def __init__(self, law, bounds, *, coasting_frames=8):
        self.law = law
        self.bounds = bounds
        self.coasting_frames = coasting_frames
        self.arc = MonoFlight(bounds=bounds, law=law)
        self.depths = deque(maxlen=15)
        self.images = deque(maxlen=12)
        self.plan_number = 0
        self.outgoing = self.turn = self.latest = None

    def _reset_stroke(self, number):
        self.plan_number = number
        self.depths.clear()
        self.images.clear()
        self.arc.reset()
        self.outgoing = self.turn = self.latest = None

    def _reversal(self, tick):
        if len(self.depths) < 4 or self.depths[-1][0] != tick:
            return None
        times, depth = np.asarray(self.depths).T
        slope = np.diff(depth) / np.diff(times)
        if self.outgoing is None:
            forward = (times >= tick - 7) & (depth > .20)
            if np.count_nonzero(forward) >= 3 and np.median(slope[-3:]) > .025:
                self.outgoing = tick
            return None
        peak = int(np.argmax(depth))
        if (times[peak] < self.outgoing - 2 or depth[peak] < .47 or peak >= len(depth) - 3
                or depth[peak] - depth[-1] < .16 or np.median(slope[-3:]) > -.024):
            return None
        return int(times[peak])

    def _fit(self, sample):
        tick, camera, pixel, size, sigma = sample
        estimate = self.arc.observe(tick, camera, pixel, size, size_sigma=sigma)
        if estimate is None:
            return
        position, uncertainty = estimate
        speed = self.arc.velocity
        if (-.215 < speed[1] < -.025 and abs(speed[0]) < .052 and abs(speed[2]) < .20
                and .004 < position[2] < 1.15 and max(uncertainty) < .145
                and abs(position[0]) < self.bounds[0] + .12):
            self.latest = ReplyEstimate(position, speed, tick)

    def update(self, tick, stream, tracker, position, plan_number):
        """Return the current reply arc or a short continuation during a gap."""
        if plan_number != self.plan_number:
            self._reset_stroke(plan_number)
        if not plan_number:
            return None

        camera = tracker.fits.get(stream)
        blob = tracker.detections['ball'].get(stream)
        sample = (tick, camera, blob, tracker.raw_ball, tracker.raw_ball_sigma)
        self.images.append(sample)
        if position is not None and abs(position[0]) < .62 and -.08 < position[2] < 1.15:
            self.depths.append((tick, position[1]))

        if self.turn is None:
            self.turn = self._reversal(tick)
            if self.turn is None:
                return None
            for earlier in self.images:
                if earlier[0] > self.turn:
                    self._fit(earlier)
        else:
            self._fit(sample)

        state = self.latest
        if state is None:
            return None
        elapsed = tick - state.observed_tick
        if elapsed == 0:
            return state
        if position is not None:
            return None
        if not 1 <= elapsed <= self.coasting_frames:
            return None
        path, velocities = self.law.trajectory(state.position, state.velocity, elapsed, bounds=self.bounds)
        estimate, velocity = path[-1], velocities[-1]
        if not (-.85 < estimate[1] < .23 and -.008 < estimate[2] < .80):
            return None
        if camera is not None and blob is not None:
            size = max(3., camera.focal * .020 / max(.05, camera.project(estimate)[2]))
            if np.linalg.norm(camera.project(estimate)[:2] - blob.center) > 2. * size:
                return None
        return ReplyEstimate(estimate, velocity, state.observed_tick)


def early_reply(player, tick, estimate):
    """Prepare a physically predicted legal shot from a qualified incoming arc."""
    if (estimate is None or not -.24 <= estimate.position[1] < .40 or not player.cleared
            or player.plan is not None or tick - player.ends < 5 or np.max(abs(player.movement)) > .002):
        return None
    choice = player.choose((estimate.position, estimate.velocity))
    if choice is None or not choice['legal'] or min(choice['robustness']) > 2.5:
        return None
    index, duration = choice['index'], choice['duration']
    player.plans.append(dict(tick=tick, reply_lead=True, **choice))
    player.plan = (tick, index, duration)
    player.cleared = False
    action = player.tapes[index, 0]
    player.movement += action
    return np.r_[action, 0.]


class ReplyBlock:
    """Solve possible settled blocks in the public arm model and rank impacts."""

    def __init__(self, bounds):
        self.bounds = bounds
        self.arm = FactoryModel()
        self.arm.base += CLAMP_SHIFT

    def _programs(self, state, law):
        points, speeds = law.trajectory(state.position, state.velocity, HORIZON, bounds=self.bounds)
        times = np.flatnonzero((points[:, 1] < -.79) & (points[:, 1] > -1.34)
                               & (points[:, 2] > .046) & (points[:, 2] < .44)
                               & (speeds[:, 1] < -.026) & (np.arange(HORIZON + 1) >= 4))
        tapes, durations = [], []
        for frame in times:
            position, velocity = points[frame], speeds[frame]
            yaw = float(np.clip(-.32 * position[0] - 6.0 * velocity[0], -.36, .36))
            for tilt in (-.18, -.024, .14, .30, .46):
                for yaw_offset in (-.08, .06):
                    actual_yaw = yaw + yaw_offset
                    normal = np.array((np.sin(actual_yaw) * np.cos(tilt),
                                       np.cos(actual_yaw) * np.cos(tilt), np.sin(tilt)))
                    center = position - .025 * normal
                    q = self.arm.solve(center, normal, HOME)
                    spot, face, _, _ = self.arm.pose(q)
                    if np.linalg.norm(spot - center) > .035 or face @ normal < .92:
                        continue
                    shift = (q - HOME) / RATE[:5]
                    for settling in (1, 2):
                        lead = int(frame) - settling
                        if np.max(abs(shift)) > lead:
                            continue
                        tape = np.zeros((HORIZON, 5))
                        tape[:lead] = shift / lead
                        tapes.append(tape)
                        durations.append(min(HORIZON, int(frame) + 3))
        return np.asarray(tapes), np.asarray(durations)

    def command(self, player, tick, estimate):
        if (estimate is None or not -.55 < estimate.position[1] < .43 or not player.cleared
                or player.plan is not None or tick - player.ends < 5 or np.max(abs(player.movement)) > .002):
            return None
        tapes, durations = self._programs(estimate, player.law)
        if not len(tapes):
            return None
        observation = observed_ball(player.history, (estimate.position, estimate.velocity), player.law)
        rows = player.sim.predict(tapes, durations, observation, np.zeros((len(tapes), 9)))
        legal = sorted((i for i, row in enumerate(rows) if row['legal']), key=lambda i: rows[i]['cost'])[:6]
        if not legal:
            return None
        changes = np.tile([[-.010, -.025, .012, -.025, -.10, .06, 25., 0., 0.],
                           [.010, .025, -.012, .025, .10, -.06, -25., 0., 0.]], (len(legal), 1))
        alternatives = player.sim.predict(np.repeat(tapes[legal], 2, axis=0), np.repeat(durations[legal], 2),
                                          observation, changes)
        stress = np.asarray([row['cost'] for row in alternatives]).reshape(len(legal), 2)
        selected = int(np.argmin([rows[i]['cost'] + .22 * stress[j].mean() for j, i in enumerate(legal)]))
        pick = legal[selected]
        index = len(player.tapes)
        duration = int(durations[pick])
        player.plans.append(dict(rows[pick], tick=tick, index=index, duration=duration,
                                 reply_block=True, alternatives=stress[selected].tolist(), candidates=len(tapes),
                                 estimate=estimate.position.tolist()))
        player.tapes = np.concatenate((player.tapes, tapes[[pick]]))
        player.plan = (tick, index, duration)
        player.cleared = False
        action = tapes[pick, 0]
        player.movement += action
        return np.r_[action, 0.]

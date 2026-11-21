"""Rank factory SO-101 strokes in an independent, RGB-initialized MuJoCo model."""

from dataclasses import dataclass

from mujoco import rollout
import numpy as np

from .tt4_arena import (BALL_INERTIA, BALL_MASS, BALL_RADIUS, MAGNUS_COEFFICIENT, NET_HEIGHT,
                        SOLVER_SUBSTEPS, SPIN_DECAY, TABLE_HEIGHT, mujoco)
from .tt19_catalog import Player, candidate_costs
from .tt19_flight import optical_start
from .tt19_shoot import COURT, DISK_RADIUS, HALF_THICKNESS, HORIZON
from .tt19_so101 import SO101Arena


LOOKAHEAD = 50
ORIGIN = np.array((0., 0., TABLE_HEIGHT))
SPIN = np.array((46., 0., 0.))


@dataclass
class BallObservation:
    center: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray
    bounced: bool


def observed_ball(history, state, law):
    """Estimate limited spin from image curvature on the latest smooth arc."""
    position, speed = map(np.asarray, state)
    tick = history[-1][0]
    recent = [(time, np.asarray(point)) for time, point in history[-8:]
              if point is not None and tick - time <= 7]
    spin = SPIN.copy()
    final = position[1] < -.46
    bounced = final and speed[2] > .017 and position[2] < .21
    if final and len(recent) >= 3:
        bounced |= any(point[2] < .080 and point[1] < -.30 for _, point in recent[:-1])
    if len(recent) >= 5:
        times, points = zip(*recent)
        xyz = np.asarray(points)
        if np.min(xyz[:, 2]) > .085 or np.all(np.diff(xyz[:, 2]) > 0.):
            ticks = (np.asarray(times) - times[-1]) / 30.
            basis = np.column_stack((np.ones(len(ticks)), ticks, .5 * ticks**2))
            origin, velocity, acceleration = np.linalg.lstsq(basis, xyz, rcond=None)[0]
            residual = np.linalg.norm(basis @ np.stack((origin, velocity, acceleration)) - xyz, axis=1)
            if max(residual) < .011 and np.linalg.norm(velocity) > .8:
                aero = acceleration + np.array((0., 0., law.gravity * 900.))
                aero += law.drag * np.linalg.norm(velocity) * velocity
                estimated = np.cross(velocity, aero) / (MAGNUS_COEFFICIENT * (velocity @ velocity))
                spin = .70 * spin + .30 * np.clip(estimated, [-120, -90, -90], [140, 90, 90])
    return BallObservation(position, speed * 30., spin, bool(bounced))


class DrySimulator:
    """Use the known station dynamics in a fresh model with no opponent contact."""

    def __init__(self, law, *, arena=None):
        self.arena = SO101Arena(venue='studio', detail='preview', strict_endline=False) if arena is None else arena
        self.model, self.data = self.arena.model, self.arena.data
        rival = self.model.geom('rival_racket')
        rival.contype[0] = rival.conaffinity[0] = 0
        self.arena.ball[:] = (1.7, 2., .6)
        self.arena.velocity[:] = 0.
        self.timestep = self.arena.frameskip * SOLVER_SUBSTEPS
        mujoco.mj_step(self.model, self.data, nstep=8 * self.timestep)
        self.state_flags = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.controls = int(mujoco.mjtState.mjSTATE_CTRL) | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        self.base = np.empty(mujoco.mj_stateSize(self.model, self.state_flags))
        mujoco.mj_getState(self.model, self.data, self.base, self.state_flags)
        self.base[0] = 0.
        self.home = self.data.ctrl.copy()
        self.n = self.model.nu
        self.force_id = self.n + 6 * self.arena._ball_body
        self.ball_p = 1 + self.arena._ball_joint
        self.ball_v = 1 + self.model.nq + self.arena._ball_dof
        self.axes = self.arena._arm_ids[1][2][:5]
        self.increments = self.arena._joint_increments[self.axes]
        self.low = self.arena._joint_low[self.axes]
        self.high = self.arena._joint_high[self.axes]
        self.drag = law.drag

    def predict(self, tape, duration, observation, offsets):
        batch = len(tape)
        size = self.n + 6 * self.model.nbody
        actual = np.broadcast_to(self.base, (batch, len(self.base))).copy()
        actual[:, self.ball_p:self.ball_p + 3] = observation.center + offsets[:, :3] + ORIGIN
        actual[:, self.ball_p + 3:self.ball_p + 7] = [1., 0., 0., 0.]
        actual[:, self.ball_v:self.ball_v + 3] = observation.velocity + offsets[:, 3:6]
        actual[:, self.ball_v + 3:self.ball_v + 6] = observation.spin + offsets[:, 6:9]
        trace = [actual[:, None, :]]
        progress = np.zeros((batch, len(self.axes)))
        controls = np.zeros((batch, self.timestep, size))
        micros = np.arange(1, self.timestep + 1)[None, :, None]
        drive = np.broadcast_to(self.home, (batch, self.n)).copy()
        used = self.ball_v
        for tick in range(LOOKAHEAD):
            rate = np.where((tick < duration)[:, None], tape[:, min(tick, HORIZON - 1)],
                            -np.clip(progress, -1., 1.))
            progress += rate
            target = drive[:, None, self.axes] + micros * (rate * self.increments)[:, None, :]
            target = np.clip(target, self.low, self.high)
            controls[:, :, :self.n] = self.home
            controls[:, :, self.axes] = target
            drive[:, self.axes] = target[:, -1]
            # Update drag and Magnus force twice per video frame. The compiled
            # model resolves contact and updates angular velocity in each chunk.
            half = self.timestep // 2
            for start in (0, half):
                q = actual[:, self.ball_p + 3:self.ball_p + 7]
                local = actual[:, used + 3:used + 6]
                axis = np.cross(q[:, 1:], local)
                spin = local + 2 * (q[:, :1] * axis + np.cross(q[:, 1:], axis))
                speed = actual[:, used:used + 3]
                force = -BALL_MASS * self.drag * np.linalg.norm(speed, axis=1, keepdims=True) * speed
                force += BALL_MASS * MAGNUS_COEFFICIENT * np.cross(spin, speed)
                controls[:, start:start + half, self.force_id:self.force_id + 3] = force[:, None, :]
                controls[:, start:start + half, self.force_id + 3:self.force_id + 6] = (
                    -BALL_INERTIA * SPIN_DECAY * spin)[:, None, :]
                chunk = np.ascontiguousarray(controls[:, start:start + half])
                states, _ = rollout.rollout(self.model, self.data, actual, chunk, control_spec=self.controls)
                actual = np.ascontiguousarray(states[:, -1])
                trace.append(states)
        paths = np.concatenate(trace, axis=1)
        points = paths[..., self.ball_p:self.ball_p + 3] - ORIGIN
        speed = paths[..., self.ball_v:self.ball_v + 3]
        return physical_ranking(points, speed, observation.bounced, self.timestep)

    def close(self):
        self.arena.close()


def physical_ranking(position, velocity, already_bounced, steps_per_frame):
    """Score solver paths by near bounce, paddle reversal, net and far bounce."""
    before, after = position[:, :-1], position[:, 1:]
    incoming, outgoing = velocity[:, :-1], velocity[:, 1:]
    width, depth = COURT
    rollouts = []
    reverse = (incoming[..., 1] <= 0.) & (outgoing[..., 1] > 0.) & (before[..., 1] < -.70)
    bounce = (incoming[..., 2] <= 0.) & (outgoing[..., 2] > 0.) & (before[..., 2] < BALL_RADIUS + .010)
    bounce &= (abs(before[..., 0]) <= width + .009) & (abs(before[..., 1]) <= depth + .009)
    for row, contact in enumerate(reverse):
        flips = np.flatnonzero(contact)
        if not flips.size:
            rollouts.append(dict(cost=14., hit=False, legal=False, eta=None, net=None, landing=None))
            continue
        strike = int(flips[0])
        near = already_bounced or np.any(bounce[row, :strike] & (before[row, :strike, 1] < 0.))
        p = position[row, strike + 1:]
        v = velocity[row, strike + 1:]
        over = np.flatnonzero((p[:-1, 1] < 0.) & (p[1:, 1] >= 0.))
        if over.size:
            k = int(over[0])
            mix = -p[k, 1] / (p[k + 1, 1] - p[k, 1])
            net = float(p[k, 2] + mix * (p[k + 1, 2] - p[k, 2]))
        else:
            net = None
        falling = (p[:-1, 2] >= BALL_RADIUS) & (p[1:, 2] < BALL_RADIUS) & (v[1:, 2] < 0.)
        first = np.flatnonzero(falling)
        landing = (p[first[0], :2] if first.size else p[np.argmin(abs(p[:, 2] - BALL_RADIUS)), :2]).tolist()
        far = np.flatnonzero(bounce[row, strike + 1:] & (before[row, strike + 1:, 1] > 0.))
        own = np.flatnonzero(bounce[row, strike + 1:] & (before[row, strike + 1:, 1] < 0.))
        legal = bool(near and far.size and (not own.size or min(far) < min(own)) and net is not None)
        if legal:
            landing = before[row, strike + 1 + far[0], :2].tolist()
        x, y = landing
        penalty = (0. if legal else 3.2) + (0. if near else 2.2)
        penalty += .38 * abs(y - .5) + .30 * abs(x)
        penalty += 4. * (max(0., .025 - y) + max(0., y - depth) + max(0., abs(x) - width))
        penalty += 6.5 * (max(0., NET_HEIGHT + BALL_RADIUS + .008 - net) if net is not None else .20)
        rollouts.append(dict(cost=float(penalty), hit=True, legal=legal, eta=strike / steps_per_frame,
                             net=net, landing=landing))
    return rollouts


def reachable_tapes(poses, points):
    """Conservatively reject sweeps too far from every eligible ball segment."""
    ball_low = np.minimum(points[:-1], points[1:])
    ball_high = np.maximum(points[:-1], points[1:])
    ticks = np.flatnonzero((ball_low[:, 1] < -.78) & (ball_high[:, 2] > .03))
    if not len(ticks):
        return np.empty(0, dtype=int)
    radius = np.hypot(DISK_RADIUS, HALF_THICKNESS) + BALL_RADIUS + .001001
    # Both centers move linearly inside each segment in the exact disk test.
    # Box overlap is necessary for contact at any of its four sample times.
    possible = np.ones((len(ticks), len(poses)), dtype=bool)
    for axis in range(3):
        start = poses[:, ticks, axis].T
        end = poses[:, ticks+1, axis].T
        possible &= np.maximum(start, end) >= ball_low[ticks, axis, None] - radius
        possible &= np.minimum(start, end) <= ball_high[ticks, axis, None] + radius
    return np.flatnonzero(np.any(possible, axis=0))


class RolloutPlayer(Player):
    """Keep the catalogue command protocol and replace the receiving shot ranker."""

    def __init__(self, catalog, law, response, *, count=32):
        super().__init__(catalog, law, response, 'late')
        self.count = count
        self.sim = DrySimulator(law)
        self.evaluated = 0

    def choose(self, estimate):
        points, speeds = self.law.trajectory(*estimate, HORIZON, bounds=COURT)
        reachable = reachable_tapes(self.poses, points)
        if not len(reachable):
            return None
        rows = [candidate_costs(self.poses[batch], self.law, self.response, points, speeds, False)
                for batch in np.array_split(reachable, (len(reachable) + 255) // 256)]
        costs, eta, hit, _, _, _, _, ball = map(np.concatenate, zip(*rows))
        potential = np.flatnonzero(hit & (ball[:, 1] < -.78) & (ball[:, 2] > .03))
        if not len(potential):
            return None
        # Keep nominally promising as well as geometrically useful options.
        by_landing = potential[np.argsort(costs[potential])]
        by_depth = potential[np.argsort(abs(ball[potential, 1] + 1.02))]
        shortlist = np.unique(np.r_[by_landing[:self.count * 3 // 4], by_depth[:self.count // 4]])
        duration = np.clip(np.ceil(eta[shortlist]).astype(int) + 2, 2, HORIZON)
        landing_cost = costs[shortlist]
        shortlist = reachable[shortlist]
        ball = observed_ball(self.history, estimate, self.law)
        trials = self.sim.predict(self.tapes[shortlist], duration, ball, np.zeros((len(shortlist), 9)))
        self.evaluated += len(trials)
        best = np.argsort([row['cost'] for row in trials])[:min(6, len(trials))]
        perturbations = np.array([[-.007, .009, -.008, -.025, -.06, .09, -34., 12., 0.],
                                  [.007, -.009, .008, .025, .06, -.09, 34., -12., 0.]])
        probes = self.sim.predict(np.repeat(self.tapes[shortlist[best]], 2, axis=0), np.repeat(duration[best], 2),
                                  ball, np.tile(perturbations, (len(best), 1)))
        self.evaluated += len(probes)
        scenario = np.column_stack(([trials[i]['cost'] for i in best],
                                    np.asarray([probe['cost'] for probe in probes]).reshape((-1, 2))))
        objective = .62 * scenario[:, 0] + .26 * scenario.mean(axis=1) + .12 * scenario.max(axis=1)
        pick = int(np.argmin(objective))
        chosen = best[pick]
        detail = trials[chosen]
        if not detail['hit']:
            return None
        return dict(detail, index=int(shortlist[chosen]), duration=int(duration[chosen]),
                    approximate_spin=ball.spin.tolist(), estimate=ball.center.tolist(),
                    candidates=len(shortlist), robustness=scenario[pick].tolist(),
                    landing_prior_cost=float(landing_cost[chosen]))

    def command(self, tick, xyz, server, *, estimated_velocity=None):
        self.history.append((tick, xyz))
        if self.plan is not None:
            start, index, until = self.plan
            if tick - start < until:
                action = self.tapes[index, tick - start]
                self.movement += action
                return np.r_[action, 0.]
            self.plan = None
            self.ends = tick
        if np.max(abs(self.movement)) > .002:
            action = -np.clip(self.movement, -1., 1.)
            self.movement += action
            return np.r_[action, 0.]
        if xyz is not None and xyz[1] > .10:
            self.cleared = True
        if self.plans and (tick - self.ends < 5 or not self.cleared):
            return np.zeros(6)
        estimate = ((xyz, estimated_velocity) if estimated_velocity is not None and xyz is not None
                    else optical_start(self.history, self.law.gravity))
        if estimate is None or xyz is None:
            return np.zeros(6)
        p, velocity = estimate
        if velocity[1] >= -.031 or not -.81 < p[1] < -.24 or tick < 8:
            return np.zeros(6)
        detail = self.choose(estimate)
        if detail is None:
            return np.zeros(6)
        if not detail['legal'] and min(detail['robustness']) > 2.8 and p[1] > -.51:
            return np.zeros(6)
        index, duration = detail['index'], detail['duration']
        self.plans.append(dict(tick=tick, **detail))
        self.plan = (tick, index, duration)
        self.cleared = False
        action = self.tapes[index, 0]
        self.movement += action
        return np.r_[action, 0.]

    def close(self):
        self.sim.close()

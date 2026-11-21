"""Physically control the rival arm with a separate MuJoCo trajectory search."""

from itertools import product

import numpy as np
from mujoco import rollout

from .tt4_arena import (AIR_DRAG, BALL_INERTIA, BALL_MASS, BALL_RADIUS, GRAVITY, HOME, LINK_LENGTH,
                        MAGNUS_COEFFICIENT, NET_HEIGHT, SOLVER_SUBSTEPS, SPIN_DECAY, TABLE_BOUNCE,
                        TABLE_FORWARD, TABLE_HEIGHT, _racket_loft, mujoco)


DEPTHS = (.032, .090, .155)
PACES = (-.28, .28, .76, 1.23)
LEADS = (.025, .062, .099)
LOFTS = (-.13, .005, .14)


def aim_angles(base, x, depth, height, pitch):
    forward = base - depth
    radius = np.hypot(x, forward)
    vertical = height - .96
    elbow = np.arccos(np.clip((radius**2 + vertical**2) / (2 * LINK_LENGTH**2) - 1, -.99, .99))
    shoulder = np.arctan2(vertical, radius) - elbow / 2
    return np.stack((np.arctan2(-x, forward), shoulder, elbow, np.broadcast_to(pitch, elbow.shape)), axis=-1)


def forecast(arena, depths):
    """Estimate reachable contact windows on the next incoming arc."""
    point = arena.ball.copy()
    speed = arena.velocity.copy()
    spin = arena.spin
    court = arena.table_half_length
    top = TABLE_HEIGHT + BALL_RADIUS
    dt = arena.control_dt / 5
    hits = {}
    for index in range(175):
        if speed[1] < .12 or point[2] < TABLE_HEIGHT - .04:
            break
        force = -AIR_DRAG * np.linalg.norm(speed) * speed + MAGNUS_COEFFICIENT * np.cross(spin, speed)
        force[2] -= GRAVITY
        nxt = point + dt * (speed + .5 * dt * force)
        velocity = speed + dt * force
        if (point[2] >= top > nxt[2] and abs(nxt[0]) < arena.table_half_width
                and abs(nxt[1]) < court):
            nxt[2] = 2 * top - nxt[2]
            velocity *= (TABLE_FORWARD, TABLE_FORWARD, -TABLE_BOUNCE)
        for depth in depths:
            if point[1] <= depth < nxt[1]:
                fraction = (depth - point[1]) / (nxt[1] - point[1])
                center = point + fraction * (nxt - point)
                if TABLE_HEIGHT + .031 < center[2] < 1.49:
                    hits[depth] = ((index + fraction) * dt, center, speed + fraction * (velocity - speed))
        if nxt[1] > max(depths):
            break
        point, speed = nxt, velocity
    return hits


def trial_quality(paths, position, velocity, court, width, already_bounced):
    """Rank observed dry bounces after a physical far-end reversal."""
    pos = paths[:, :, position:position + 3]
    vel = paths[:, :, velocity:velocity + 3]
    previous, current = pos[:, :-1], pos[:, 1:]
    before, after = vel[:, :-1], vel[:, 1:]
    tabletop = TABLE_HEIGHT + BALL_RADIUS
    bounce = ((before[..., 2] < 0.) & (after[..., 2] > 0.) & (previous[..., 2] <= tabletop + .014)
              & (previous[..., 2] >= TABLE_HEIGHT - .028) & (abs(previous[..., 0]) <= width + .010)
              & (abs(previous[..., 1]) <= court + .010))
    reversal = ((before[..., 1] > .22) & (after[..., 1] < -.34) & (previous[..., 1] > court - .035)
                & (previous[..., 2] > TABLE_HEIGHT + .027))
    results = []
    for index, hits in enumerate(reversal):
        choices = np.flatnonzero(hits)
        if not len(choices):
            results.append(None)
            continue
        struck = choices[0]
        if not already_bounced and not np.any(bounce[index, :struck] & (previous[index, :struck, 1] > 0.)):
            results.append(None)
            continue
        after_hit = pos[index, struck + 1:]
        over = np.flatnonzero((after_hit[:-1, 1] >= 0.) & (after_hit[1:, 1] < 0.))
        landed = np.flatnonzero(bounce[index, struck + 1:])
        if not len(over) or not len(landed):
            results.append(None)
            continue
        split = over[0]
        a, b = after_hit[split:split + 2]
        net = a[2] + a[1] / (a[1] - b[1]) * (b[2] - a[2])
        x, y, _ = previous[index, struck + 1 + landed[0]]
        if y >= -.034 or net < TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS - .004:
            results.append(None)
            continue
        # Keep the next shot central and inside the learner's receiving depth.
        cost = 1.7 * abs(x) + .85 * abs(y + .63 * court)
        cost += 2.5 * max(0., TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS + .030 - net)
        cost += .22 * max(0., net - TABLE_HEIGHT - .29)
        results.append(dict(cost=float(cost), landing=[float(x), float(y)],
                            net_m=float(net - TABLE_HEIGHT), strike_micro=int(struck)))
    return results


class RivalSearch:
    """Choose joint-rate commands using a separate solver and private rival state."""

    def __init__(self, arena, inherited):
        self.arena = arena
        self.inherited = inherited
        self.model = arena.model
        self.dry = mujoco.MjData(self.model)
        self.micro = arena.frameskip * SOLVER_SUBSTEPS
        self.axes = arena._arm_ids[-1][2]
        self.increment = arena._joint_increments[self.axes]
        self.position = 1 + arena._ball_joint
        self.velocity = 1 + self.model.nq + arena._ball_dof
        self.state = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.controls = int(mujoco.mjtState.mjSTATE_CTRL) | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        self.force = self.model.nu + 6 * arena._ball_body
        self.buffer = np.empty(mujoco.mj_stateSize(self.model, self.state))
        self.reset()

    def reset(self):
        self.decisions = []
        self.new_shot()

    def new_shot(self):
        self.tape = None
        self.start = -1
        self.attempts = 0

    def command(self, bias, target_depth, scale):
        arena = self.arena
        if arena._last_hitter == -1 or arena.velocity[1] <= .22:
            return self.inherited(-1, bias, target_depth, scale)
        x, y, z = arena.ball
        if self.tape is not None and arena.steps - self.start < len(self.tape):
            return self.tape[arena.steps - self.start]
        gate = -.055 if self.attempts == 0 else .36
        if self.attempts < 2 and gate <= y < arena.table_half_length + .04:
            self.attempts += 1
            self.tape, record = self.plan()
            self.start = arena.steps
            self.decisions.append(dict(tick=arena.steps, ball=arena.ball.tolist(), **record))
            if self.tape is not None:
                return self.tape[0]
        if y < gate:
            depths = (arena.table_half_length + .09,)
            hit = forecast(arena, depths).get(depths[0])
            if hit is not None:
                _, point, _ = hit
                goal = aim_angles(arena.rival_base_depth, point[0], depths[0] + .070, point[2], .19)
                return arena._command_to(-1, goal, .92)
        return self.inherited(-1, bias, target_depth, max(scale, .94))

    def plan(self):
        arena = self.arena
        court = arena.table_half_length
        frames = 48
        timings = np.arange(frames) * arena.control_dt
        hits = forecast(arena, tuple(court + depth for depth in DEPTHS))
        goals, params = [], []
        for depth, (when, center, speed) in hits.items():
            for pace, lead, loft in product(PACES, LEADS, LOFTS):
                lateral = (-.45 * np.sign(center[0])
                           if abs(center[0]) > .19 and lead == LEADS[-1] else 0.)
                pitch = float(np.clip(_racket_loft(-speed[1], speed[2], center[2], depth + .64 * court,
                                                   pace) + loft, -.49, .79))
                shifted = np.clip(pace * (timings - when + lead), -.14, .15)
                depth_goal = depth + (BALL_RADIUS + .006) * np.cos(pitch) - shifted
                height = center[2] - (BALL_RADIUS + .006) * np.sin(pitch)
                x = center[0] + np.clip(lateral * (timings - when + lead), -.06, .06)
                goals.append(aim_angles(arena.rival_base_depth, x, depth_goal, height, pitch))
                params.append((depth, pace, lead, pitch, lateral))
        if not goals:
            return None, dict(candidates=0, predicted_legal=0)
        planned, paths = self.predict(np.asarray(goals))
        rows = trial_quality(paths, self.position, self.velocity, court, arena.table_half_width, arena._received_bounce)
        valid = [i for i, row in enumerate(rows) if row is not None]
        if not valid:
            return None, dict(candidates=len(rows), predicted_legal=0)
        best = min(valid, key=lambda index: rows[index]['cost'])
        return planned[best], dict(candidates=len(rows), predicted_legal=len(valid), selected=params[best], **rows[best])

    def predict(self, goals):
        arena = self.arena
        count, frames, _ = goals.shape
        data = arena.data
        mujoco.mj_getState(self.model, data, self.buffer, self.state)
        state = np.broadcast_to(self.buffer, (count, len(self.buffer))).copy()
        traces = [state[:, None]]
        requests = np.broadcast_to(data.ctrl[self.axes], (count, len(self.axes))).copy()
        low, high = arena._joint_low[self.axes], arena._joint_high[self.axes]
        n = self.model.nu
        control = np.zeros((count, self.micro, n + 6 * self.model.nbody))
        control[:, :, :n] = data.ctrl
        micros = np.arange(1, self.micro + 1)[None, :, None]
        rates = []
        for target in np.moveaxis(goals, 1, 0):
            rate = np.clip((np.clip(target, low, high) - requests) / (self.micro * self.increment), -1., 1.)
            targets = np.clip(requests[:, None, :] + micros * (rate * self.increment)[:, None, :], low, high)
            requests = targets[:, -1]
            rates.append(rate)
            control[:, :, self.axes] = targets
            q = state[:, self.position + 3:self.position + 7]
            motion = state[:, self.velocity:self.velocity + 3]
            local = state[:, self.velocity + 3:self.velocity + 6]
            twist = np.cross(q[:, 1:], local)
            spin = local + 2 * (q[:, :1] * twist + np.cross(q[:, 1:], twist))
            force = BALL_MASS * (-AIR_DRAG * np.linalg.norm(motion, axis=1, keepdims=True) * motion
                                 + MAGNUS_COEFFICIENT * np.cross(spin, motion))
            control[:, :, self.force:self.force + 3] = force[:, None, :]
            control[:, :, self.force + 3:self.force + 6] = (-BALL_INERTIA * SPIN_DECAY * spin)[:, None, :]
            solved, _ = rollout.rollout(self.model, self.dry, state, control, control_spec=self.controls)
            state = np.ascontiguousarray(solved[:, -1])
            traces.append(solved)
        return np.stack(rates, axis=1), np.concatenate(traces, axis=1)


def install_opponent(arena):
    """Install stronger receiving and preserve the arena's real launch and referee."""
    inherited = arena._aim
    response = arena._choose_rival_response
    reset = arena.reset
    rival = RivalSearch(arena, inherited)

    def opening(*args, **kwargs):
        rival.reset()
        return reset(*args, **kwargs)

    def choose():
        response()
        arena._rival_attempt = True
        rival.new_shot()

    def aim(side, bias=0., target_depth=.96, scale=1.):
        return rival.command(bias, target_depth, scale) if side == -1 else inherited(side, bias, target_depth, scale)

    arena.reset = opening
    arena._choose_rival_response = choose
    arena._aim = aim
    return rival

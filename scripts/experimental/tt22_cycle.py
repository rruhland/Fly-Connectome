"""Privileged two-arm physical controller targeting a recurrent rally envelope."""

from copy import copy, deepcopy
from itertools import product

import numpy as np

from .tt4_arena import (AIR_DRAG, BALL_INERTIA, BALL_MASS, BALL_RADIUS, MAGNUS_COEFFICIENT,
                        NET_HEIGHT, SPIN_DECAY, TABLE_HEIGHT, mujoco)
from .tt19_factory import CONTACT_OFFSET, FactoryModel, moving_joint_rate
from .tt20_curriculum import CLAMP_SHIFT
from .tt20_fixture import TT20Arena
from .tt20_opponent import RivalSearch, aim_angles
from .tt21_coach_oracle import PHASE


ARENA = dict(venue='hall', detail='preview', strict_endline=True, mount='clamp', wrist_optics='outboard',
             role_palette='natural', marked=False, view_mode='fixed', width=320, height=240, max_steps=1400)


class World:
    """Keep stock contacts, aerodynamics, serve, and referee; accept both robot rate tapes."""

    def __init__(self):
        self.arena = TT20Arena(**ARENA)
        self.arena._aim = self._aim
        self.arena._choose_rival_response = self._rival_active
        self.far = np.zeros(4)

    def _aim(self, side, *_):
        if side != -1:
            raise ValueError('only the far rate controller is installed on the fixture')
        return self.far

    def _rival_active(self):
        self.arena._rival_attempt = True

    def reset(self, seed):
        self.arena.reset(seed=seed, server=-1, render=False)
        self._rival_active()

    def step(self, near, far):
        self.far = np.asarray(far, dtype=float)
        return self.arena.step(near, render=False)

    def close(self):
        self.arena.close()


class Flight:
    """Compute the upcoming bounce and passage without dry paddle interference."""

    def __init__(self, arena):
        self.model = copy(arena.model)
        self.data = mujoco.MjData(self.model)
        keep = {arena.model.geom(name).id for name in (*arena.table_panels, 'ball', 'floor', 'net')}
        off = list(set(range(arena.model.ngeom)) - keep)
        self.model.geom_contype[off] = 0
        self.model.geom_conaffinity[off] = 0
        self.live = arena

    def predict(self, side, micro, frames):
        arena = self.live
        data = self.data
        mujoco.mj_copyData(data, self.model, arena.data)
        q, v = arena._ball_joint, arena._ball_dof
        dof = data.qvel[v:v + 6]
        ball = data.qpos[q:q + 7]
        force = data.xfrc_applied[arena._ball_body]
        path, speeds = [ball[:3].copy()], [dof[:3].copy()]
        bounce = 0 if arena._received_bounce else None
        for i in range(1, micro * frames + 1):
            twist = np.cross(ball[4:7], dof[3:])
            spin = dof[3:] + 2 * (ball[3] * twist + np.cross(ball[4:7], twist))
            force[:3] = BALL_MASS * (-AIR_DRAG * np.linalg.norm(dof[:3]) * dof[:3]
                                      + MAGNUS_COEFFICIENT * np.cross(spin, dof[:3]))
            force[3:] = -BALL_INERTIA * SPIN_DECAY * spin
            vertical = dof[2]
            mujoco.mj_step(self.model, data)
            if (vertical < 0. < dof[2] and 0. < -side * ball[1] < .805
                    and abs(ball[0]) < .428 and ball[2] < TABLE_HEIGHT + .037):
                if bounce is not None:
                    break
                bounce = i
            if -side * ball[1] > 1.18 or ball[2] < TABLE_HEIGHT - .12 or side * dof[1] > -.16:
                break
            path.append(ball[:3].copy())
            speeds.append(dof[:3].copy())
        return np.asarray(path), np.asarray(speeds), bounce


def scores(path, p, v, side, already_bounced, anchor):
    """Score the next ball at the opposite receiver's finite capture aperture."""
    point, speed = path[..., p:p+3], path[..., v:v+3]
    prior, after = point[:, :-1], point[:, 1:]
    v0, v1 = speed[:, :-1], speed[:, 1:]
    depth = side * prior[..., 1]
    reverse = (side * v0[..., 1] <= -.18) & (side * v1[..., 1] >= .25) & (depth < -.71)
    bounce = (v0[..., 2] < 0.) & (v1[..., 2] > 0.) & (prior[..., 2] < TABLE_HEIGHT + .036)
    bounce &= (prior[..., 2] > TABLE_HEIGHT - .025) & (abs(prior[..., 0]) < .430) & (abs(depth) < .805)
    rows = []
    for i, possibilities in enumerate(reverse):
        hits = np.flatnonzero(possibilities)
        if not len(hits):
            continue
        strike = hits[0]
        if not already_bounced and not np.any(bounce[i, :strike] & (depth[i, :strike] < 0.)):
            continue
        crosses = np.flatnonzero((depth[i, strike:] < 0.) & (side * after[i, strike:, 1] >= 0.)) + strike
        landings = np.flatnonzero(bounce[i, strike:] & (depth[i, strike:] > 0.)) + strike
        if not len(crosses) or not len(landings) or landings[0] <= crosses[0]:
            continue
        cross, landing = crosses[0], landings[0]
        if not .12 < depth[i, landing] < .755 or np.any(bounce[i, strike:cross]):
            continue
        net = prior[i, cross, 2] - TABLE_HEIGHT
        if net < NET_HEIGHT + BALL_RADIUS - .008:
            continue
        # The dry partner stays still. Evaluate the ball just before any next
        # impact, with a short interpolation to the common reception plane.
        segment = point[i, landing+1:]
        flight = speed[i, landing+1:]
        signed = side * flight[:, 1]
        early = np.flatnonzero(signed < .15)
        if len(early):
            segment, flight = segment[:early[0]], flight[:early[0]]
        if not len(segment):
            continue
        valid = (side * segment[:, 1] >= .825) & (segment[:, 2] > TABLE_HEIGHT + .062)
        choices = np.flatnonzero(valid)
        if not len(choices):
            continue
        idx = choices[np.argmin(abs(side*segment[choices, 1] - .91))]
        xyz, vel = segment[idx], flight[idx]
        height = xyz[2] - TABLE_HEIGHT
        margin = max(0., abs(prior[i, landing, 0]) - .32) + max(0., depth[i, landing] - .71)
        cost = (6.2*abs(xyz[0]) + 3.3*abs(height-anchor[0]) + .29*abs(side*vel[1]-anchor[1])
                + .09*abs(vel[0]) + .09*abs(vel[2]-.5) + 2.7*abs(side*xyz[1]-.91)
                + 3.5*margin + 1.9*abs(depth[i, landing]-.49) + 2.5*max(0., .187-net))
        rows.append(dict(index=i, cost=float(cost), strike_micro=int(strike), net_m=float(net),
                         landing=prior[i, landing].tolist(), receive=(xyz-[0.,0.,TABLE_HEIGHT]).tolist(),
                         receive_velocity=vel.tolist()))
    return sorted(rows, key=lambda row: row['cost'])


class CyclePlanner:
    def __init__(self, world, *, height=.205, speed=2.60):
        self.world = world
        self.arena = world.arena
        self.flight = Flight(self.arena)
        self.dry = RivalSearch(self.arena, self.arena._aim)
        self.factory = FactoryModel()
        self.factory.base += CLAMP_SHIFT
        self.check = World()
        self.check_flight = Flight(self.check.arena)
        self.anchor = (height, speed)

    def _options(self, side, frames, *, wide=False):
        arena = self.arena
        positions, speeds, bounce = self.flight.predict(side, self.dry.micro, frames)
        if bounce is None:
            return [], []
        clock = np.arange(frames)
        height = positions[:, 2] - TABLE_HEIGHT
        depth = -side * positions[:, 1]
        points = np.flatnonzero((np.arange(len(height)) > bounce) & (depth > .758) & (depth < 1.12)
                                & (height > .044) & (height < .48) & (abs(positions[:, 0]) < .425))
        if not len(points):
            return [], []
        points = np.unique(points[np.round(np.linspace(0., len(points)-1, min(6, len(points)))).astype(int)])
        goals, notes = [], []
        if side == 1:
            reference = arena.data.ctrl[self.dry.axes].copy()
            for micro in points:
                xyz = positions[micro] - [0., 0., TABLE_HEIGHT]
                eta = micro / self.dry.micro
                vx, vy, _ = speeds[micro]
                yaw = np.clip(-.36 * vx / max(.50, -vy) - .42 * xyz[0], -.29, .29)
                yaw_offsets = (0.,) if not wide else (-.085, .085)
                for pitch, side_yaw in product((.02,.17,.33,.49,.66,.82), yaw_offsets):
                    angle = yaw + side_yaw
                    normal = np.array((np.sin(angle)*np.cos(pitch), np.cos(angle)*np.cos(pitch), np.sin(pitch)))
                    target = xyz - CONTACT_OFFSET*normal
                    q = self.factory.solve(target, normal, reference)
                    center, face, _, _ = self.factory.pose(q)
                    if np.linalg.norm(center-target) > .12 or face @ normal < .55:
                        continue
                    for pace, lead in product((-.009,.008,.025,.047), (1.4,3.7)):
                        rates = moving_joint_rate(self.factory, q, pace)
                        goals.append(q + np.clip(clock-eta+lead, -4.5, 4.5)[:, None] * rates)
                        notes.append(dict(impact_frame=float(eta), pitch=pitch, lead=lead, pace=pace))
        else:
            for micro in points:
                x, y, z = positions[micro]
                eta = micro / self.dry.micro
                vx = speeds[micro, 0]
                lateral = (0., np.clip(-vx*.50 - x, -.72, .72)) if wide else (0.,)
                for pitch, pace, lead, sway in product((.00,.16,.31,.46,.62,.76), (-.35,.20,.66,1.12),
                                                       (1.1,2.6), lateral):
                    motion = (clock-eta+lead) * arena.control_dt
                    goal = aim_angles(arena.rival_base_depth, x + np.clip(sway*motion, -.11, .11),
                                      y + CONTACT_OFFSET*np.cos(pitch)-np.clip(pace*motion, -.17,.17),
                                      z - CONTACT_OFFSET*np.sin(pitch), pitch)
                    goals.append(goal)
                    notes.append(dict(impact_frame=float(eta), pitch=pitch, lead=lead, pace=pace))
        return goals, notes

    def _validate(self, side, tape):
        trial, live = self.check, self.arena
        arena = trial.arena
        mujoco.mj_copyData(arena.data, arena.model, live.data)
        for key in PHASE:
            if hasattr(live, key):
                setattr(arena, key, deepcopy(getattr(live, key)))
        arena.events = list(live.events)
        trial._rival_active()
        for action in tape:
            near, far = (np.r_[action,0.], np.zeros(4)) if side == 1 else (np.zeros(6), action)
            _, _, done, capped, _ = trial.step(near, far)
            gained = arena.legal_returns > live.legal_returns if side == 1 else arena.rival_returns > live.rival_returns
            if gained:
                landing = next(e for e in reversed(arena.events) if e['kind']=='legal_return' and e['side']==side)
                loc = landing.get('location')
                positions, velocity, _ = self.check_flight.predict(-side, self.dry.micro, 18)
                depth = side*positions[:, 1]
                height = positions[:, 2] - TABLE_HEIGHT
                playable = (depth > .825) & (depth < 1.05) & (height > .075) & (height < .40)
                playable &= (abs(positions[:, 0]) < .33) & (side*velocity[:, 1] > 1.10)
                return bool(loc and .11 < side*loc[1] < .740 and np.any(playable))
            if done or capped:
                break
        return False

    def choose(self, side, *, wide=False):
        arena = self.arena
        self.dry.axes = arena._arm_ids[side][2][:5] if side == 1 else arena._arm_ids[side][2]
        self.dry.increment = arena._joint_increments[self.dry.axes]
        goals, parameters = self._options(side, 59, wide=wide)
        note = dict(tick=arena.steps, side=side, candidate_count=len(goals), solver_legal=0, exact_attempts=0)
        if not goals:
            return None, note
        tapes, paths = self.dry.predict(np.asarray(goals))
        ranking = scores(paths, self.dry.position, self.dry.velocity, side, arena._received_bounce, self.anchor)
        note['solver_legal'] = len(ranking)
        for candidate in ranking[:12]:
            index = candidate['index']
            note['exact_attempts'] += 1
            if self._validate(side, tapes[index]):
                return tapes[index], dict(note, chosen=candidate, swing=parameters[index])
        return None, note

    def close(self):
        self.check.close()


def rally(world, planner, seed, *, verbose=False):
    world.reset(seed)
    arena = world.arena
    decisions, actions = [], []
    sender, tape, begin, attempts, last_attempt = None, None, 0, 0, -6
    for tick in range(arena.max_steps):
        if sender != arena._last_hitter:
            sender = arena._last_hitter
            tape, attempts, last_attempt = None, 0, -6
        side = -sender
        if (tape is None and side * arena.velocity[1] < -.4 and tick - last_attempt >= 5 and attempts < 3
                and arena._racket_ids[sender] not in arena._previous_touches):
            tape, record = planner.choose(side, wide=attempts>0)
            decisions.append(record)
            if verbose:
                print('decision', seed, tick, side, record['candidate_count'], record['solver_legal'],
                      'ready' if tape is not None else 'miss', 'returns', arena.legal_returns, arena.rival_returns, flush=True)
            begin, last_attempt, attempts = tick, tick, attempts + 1
        near, far = np.zeros(6), np.zeros(4)
        if tape is not None and tick-begin < len(tape):
            if side == 1:
                near[:5] = tape[tick-begin]
            else:
                far[:] = tape[tick-begin]
        actions.append([near.tolist(), far.tolist()])
        _, score, ended, cap, _ = world.step(near, far)
        if ended or cap:
            break
    result = arena.point_result()
    legal = [event for event in arena.events if event['kind']=='legal_return']
    if any(a['side']==b['side'] for a,b in zip(legal, legal[1:])):
        raise ValueError('rally is not legally alternating')
    return dict(seed=seed, actions=actions, decisions=decisions, events=arena.events.copy(),
                result=result, reward=score, legal_alternating_returns=len(legal))

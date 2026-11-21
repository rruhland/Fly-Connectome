"""A physical post-bounce rival for sustained compact-court rallies."""

from copy import copy, deepcopy
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from .tt4_arena import (AIR_DRAG, ARM_AXES, BALL_INERTIA, BALL_MASS, BALL_RADIUS, HOME, JOINT_RATES,
                        MAGNUS_COEFFICIENT, NET_HEIGHT, SPIN_DECAY, TABLE_HEIGHT, _racket_loft, mujoco)
from .tt20_fixture import TT20Arena
from .tt20_opponent import RivalSearch, aim_angles
from .tt21_coach_oracle import PHASE


RALLY_STEPS = 1400
PLAN_FRAMES = 57
EXACT_CHECKS = 6
DESIGNS = qmc.Sobol(5, scramble=True, seed=14724).random_base2(5)
STEERING_DESIGNS = qmc.Sobol(6, scramble=True, seed=2214724).random_base2(5)
DEPTHS = (.635, .700, .775, .855, .940, 1.030)
ANCHOR = (0., -.91, .19, 2.2)
VARIED = ((-.10, -.89, .17, 2.), (.10, -.93, .21, 2.4),
          (.0, -.87, .20, 2.15), (-.06, -.96, .23, 2.6), (.06, -.91, .15, 2.05))


class RallyArena(TT20Arena):
    """The same clamped learner and court with an actively steerable rival wrist."""

    def __init__(self, *args, rival_reach=.040, max_steps=RALLY_STEPS, **settings):
        self.rival_reach = float(rival_reach)
        super().__init__(*args, max_steps=max_steps, **settings)

    def _model_document(self):
        import xml.etree.ElementTree as ET

        document = super()._model_document()
        base = document.find(".//body[@name='rival_base']")
        position = np.fromstring(base.get('pos'), sep=' ')
        self.rival_base_depth = float(position[1] - self.rival_reach)
        position[1] = self.rival_base_depth
        base.set('pos', ' '.join(map(str, position)))
        constraint = document.find('equality')
        constraint.remove(constraint.find("tendon[@tendon1='rival_counter_yaw']"))
        document.find(".//joint[@name='rival_yaw_link']").set('range', '-1.48 1.48')
        ET.SubElement(document.find('actuator'), 'position', name='rival_yaw_link', joint='rival_yaw_link',
                      kp='47', kv='1.8', forcerange='-4.2 4.2')
        return document

    def _arm_axes(self, side):
        if side == -1:
            return (*ARM_AXES, 'yaw_link'), ('pitch',), np.r_[JOINT_RATES, 9.]
        return super()._arm_axes(side)

    def _arm_home(self, side):
        home, passive = super()._arm_home(side)
        if side == -1:
            return np.r_[home, -home[0]], passive[:1]
        return home, passive

    def _command_to(self, side, goal, scale=1.):
        if side != -1:
            return super()._command_to(side, goal, scale)
        target = np.asarray(goal)
        if len(target) == 4:
            target = np.r_[target, -target[0]]
        axes = self._arm_ids[side][2]
        bounded = np.clip(target, self._joint_low[axes], self._joint_high[axes])
        increments = self._joint_increments[axes] * round(self.control_dt / self.model.opt.timestep)
        return np.clip((bounded - self.data.ctrl[axes]) / increments, -scale, scale)


def _air(data, ball, dof, body):
    position = data.qpos[ball:ball + 3]
    velocity = data.qvel[dof:dof + 3]
    quat = data.qpos[ball + 3:ball + 7]
    local = data.qvel[dof + 3:dof + 6]
    spin = np.empty(3)
    mujoco.mju_rotVecQuat(spin, local, quat)
    force = data.xfrc_applied[body]
    vx, vy, vz = velocity
    wx, wy, wz = spin
    drag = -BALL_MASS * AIR_DRAG * float(np.sqrt(velocity @ velocity))
    magnus = BALL_MASS * MAGNUS_COEFFICIENT
    force[:3] = (drag * vx + magnus * (wy * vz - wz * vy),
                 drag * vy + magnus * (wz * vx - wx * vz),
                 drag * vz + magnus * (wx * vy - wy * vx))
    force[3:] = -BALL_INERTIA * SPIN_DECAY * spin
    return position, velocity


class Incoming:
    """Follow real table contact without exposing any state to the playing pupil."""

    def __init__(self, arena):
        self.arena = arena
        self.model = copy(arena.model)
        self.data = mujoco.MjData(self.model)
        self.ball = arena._geom_ids['ball']
        self.surface = {arena.model.geom(name).id for name in arena.table_panels}
        kept = self.surface | {self.ball, arena._geom_ids['floor'], arena._geom_ids['net']}
        disabled = list(set(range(arena.model.ngeom)) - kept)
        self.model.geom_contype[disabled] = 0
        self.model.geom_conaffinity[disabled] = 0

    def windows(self, frames):
        arena, model, data = self.arena, self.model, self.data
        mujoco.mj_copyData(data, model, arena.data)
        depths = np.asarray(DEPTHS) + arena.table_half_length - .8
        step = model.opt.timestep
        limits = frames * round(arena.control_dt / step)
        found = []
        touched = set(arena._previous_touches).intersection(self.surface)
        bounced = arena._received_bounce
        for tick in range(limits):
            p, v = _air(data, arena._ball_joint, arena._ball_dof, arena._ball_body)
            before = p.copy()
            mujoco.mj_step(model, data)
            if p[2] < TABLE_HEIGHT - .045 or p[1] > depths[-1] + .03 or v[1] < .12:
                break
            contact = {int(c.geom[1] if c.geom[0] == self.ball else c.geom[0]) for c in data.contact
                       if c.efc_address >= 0 and self.ball in c.geom and abs(c.frame[2]) > .55}
            table = contact & self.surface
            if table - touched:
                if bounced or p[1] <= 0.:
                    break
                bounced = True
            touched = table
            if not bounced:
                continue
            for y in depths[(before[1] <= depths) & (depths < p[1])]:
                part = (y - before[1]) / (p[1] - before[1])
                ball = before + part * (p - before)
                radius = np.linalg.norm((ball[0], arena.rival_base_depth - ball[1], ball[2] - .96))
                if TABLE_HEIGHT + .035 < ball[2] < 1.52 and radius < .790:
                    found.append(((tick + part) * step, ball, v.copy()))
        return found


def _goals(arena, options, lane, frames):
    timeline = np.arange(frames) * arena.control_dt
    steering = len(arena._arm_ids[-1][2]) == 5
    design = STEERING_DESIGNS if steering else DESIGNS
    goals, parameters = [], []
    for hit_index, (time, ball, incoming) in enumerate(options):
        for sample in design:
            pace_bias, timing, loft, side, lift = sample[:5]
            pace = -.38 + 2.48 * pace_bias
            lead = .023 + .118 * timing
            base = _racket_loft(-incoming[1], incoming[2], ball[2], ball[1] + .72, pace)
            pitch = np.clip(base - .21 + .47 * loft, -.47, .90)
            lateral = np.clip(2.9 * (lane[0] - ball[0]), -.82, .82) + (side - .5) * 1.7
            rise = -.42 + .96 * lift
            heading = None
            if steering:
                desired_side_speed = (lane[0] - ball[0]) / .60
                aim = (desired_side_speed - incoming[0] - .20 * lateral) / (max(1.2, incoming[1]) + 3.)
                heading = float(np.clip(aim + .36 * (sample[5] - .50), -.40, .40))
            goals.append(_stroke(arena, timeline, time, ball, pace, lead, pitch, lateral, rise, heading))
            parameters.append((hit_index, pace, lead, float(pitch), float(lateral), rise)
                              if heading is None else (hit_index, pace, lead, float(pitch), float(lateral), rise, heading))
    return np.asarray(goals), parameters


def _stroke(arena, timeline, time, ball, pace, lead, pitch, lateral, rise, heading):
    phase = timeline - time + lead
    depth = ball[1] + (BALL_RADIUS + .006) * np.cos(pitch) - np.clip(pace * phase, -.17, .20)
    height = ball[2] - (BALL_RADIUS + .006) * np.sin(pitch) + np.clip(rise * phase, -.063, .063)
    lateral = ball[0] + np.clip(lateral * phase, -.158, .158)
    if heading is not None:
        lateral = lateral - .026 * np.sin(heading)
    goals = aim_angles(arena.rival_base_depth, lateral, depth, height, pitch)
    if heading is None:
        return goals
    return np.column_stack((goals, -goals[:, 0] - heading))


def _rank(arena, centers, speeds, lane):
    """Use the physical trajectories to rank a useful incoming window after landing."""
    position, velocity = centers[:, :-1], speeds[:, :-1]
    near = (abs(position[..., 0]) < arena.table_half_width + .008)
    near &= (abs(position[..., 1]) < arena.table_half_length + .006)
    near &= (position[..., 2] < TABLE_HEIGHT + BALL_RADIUS + .014)
    near &= (position[..., 2] > TABLE_HEIGHT - .018)
    bounce = near & (velocity[..., 2] < 0.) & (speeds[:, 1:, 2] > 0.)
    reverse = ((speeds[:, 3:, 1] < -.26) & (speeds[:, :-3, 1] > .18)
               & (centers[:, 2:-1, 1] > .50) & (centers[:, 2:-1, 2] > TABLE_HEIGHT + .024))
    crossing = (centers[:, :-1, 1] >= 0.) & (centers[:, 1:, 1] < 0.)
    target_x, target_y, target_z, target_v = lane
    records = []
    for index, hits in enumerate(reverse):
        hit = np.flatnonzero(hits)
        if not len(hit):
            continue
        strike = int(hit[0] + 2)
        prior = np.flatnonzero(bounce[index, :strike])
        if not arena._received_bounce and (len(prior) != 1 or position[index, prior[0], 1] <= 0.):
            continue
        passes = np.flatnonzero(crossing[index, strike:])
        lands = np.flatnonzero(bounce[index, strike:])
        if not len(passes) or not len(lands) or passes[0] >= lands[0]:
            continue
        net_index, land = strike + int(passes[0]), strike + int(lands[0])
        a, b = centers[index, net_index:net_index + 2]
        net = float(a[2] + a[1] * (b[2] - a[2]) / (a[1] - b[1]) - TABLE_HEIGHT)
        if position[index, land, 1] >= -.025 or net < NET_HEIGHT + BALL_RADIUS - .016:
            continue
        end = strike + int(lands[1]) if len(lands) > 1 else len(centers[index])
        sample = centers[index, land + 1:end]
        speed = speeds[index, land + 1:end]
        usable = ((abs(sample[:, 0]) < .38) & (sample[:, 1] > -1.14) & (sample[:, 1] < -.745)
                  & (sample[:, 2] > TABLE_HEIGHT + .05) & (sample[:, 2] < TABLE_HEIGHT + .47)
                  & (speed[:, 1] < -.45))
        use = np.flatnonzero(usable)
        if not len(use):
            cost = 8. + abs(position[index, land, 0]) + abs(position[index, land, 1] + .66)
            reach = None
        else:
            xyz = sample[use] - [0., 0., TABLE_HEIGHT]
            velocity_y = speed[use, 1]
            loss = (5.5 * abs(xyz[:, 0] - target_x) + 5.0 * abs(xyz[:, 1] - target_y)
                    + 4.0 * abs(xyz[:, 2] - target_z) + .42 * abs(velocity_y + target_v))
            picked = int(np.argmin(loss))
            cost = float(loss[picked]) + .10 * max(0., 1. - len(use) / 20.)
            reach = [*xyz[picked].tolist(), float(-velocity_y[picked])]
        cost += 2.4 * max(0., NET_HEIGHT + BALL_RADIUS + .048 - net)
        records.append(dict(index=index, cost=float(cost), strike_micro=strike,
                            landing=position[index, land, :2].tolist(), net=net, reachable=reach,
                            window_microsteps=len(use)))
    return sorted(records, key=lambda record: record['cost'])


class RallyOpponent(RivalSearch):
    """Replan a separate rate-limited rival arm on each actual learner strike."""

    def __init__(self, arena, inherited, lane, target_profile=None):
        self.lane = lane
        self.target_profile = target_profile
        super().__init__(arena, inherited)
        self.incoming = Incoming(arena)
        self.check = copy(arena)
        self.check.data = mujoco.MjData(arena.model)

    def new_shot(self):
        super().new_shot()
        self.stage = -1

    def target(self):
        if self.target_profile is not None:
            return self.target_profile
        if self.lane == 'anchor':
            return ANCHOR
        index = (self.arena.seed + self.arena.contacts - 1) % len(VARIED)
        return VARIED[index]

    def command(self, bias, target_depth, scale):
        arena = self.arena
        if arena._last_hitter == -1 or arena.velocity[1] <= .22:
            return arena._command_to(-1, HOME, .97)
        age = arena.steps - self.start
        if self.tape is not None and 0 <= age < len(self.tape):
            return self.tape[age]
        y = arena.ball[1]
        stage = int(arena._received_bounce) + int(y > .37) + int(y > -.43)
        separated = arena._racket_ids[1] not in arena._previous_touches
        if stage > self.stage and stage > 0 and y < 1.04 and separated:
            self.stage = stage
            self.tape, record = self.plan()
            self.start = arena.steps
            self.decisions.append(dict(tick=arena.steps, phase=stage, ball=arena.ball.tolist(), **record))
            if self.tape is not None:
                return self.tape[0]
        return arena._command_to(-1, HOME, .8)

    def plan(self):
        started = perf_counter()
        arena = self.arena
        frames = min(PLAN_FRAMES, arena.max_steps - arena.steps)
        options = self.incoming.windows(frames)
        record = dict(windows=len(options), candidates=0, predicted_legal=0, exact_trials=0, confirmed=False)
        if not options:
            return None, dict(record, wall_seconds=perf_counter() - started)
        lane = self.target()
        goals, params = _goals(arena, options, lane, frames)
        tapes, paths = self.predict(goals)
        ranked = self._rank_paths(paths, lane)
        record.update(candidates=len(goals), predicted_legal=len(ranked), target=list(lane))
        if ranked:
            chosen = self.confirm(ranked, tapes, record)
            if chosen is not None:
                return tapes[chosen['index']], dict(record, **chosen, selected=params[chosen['index']],
                                                    wall_seconds=perf_counter() - started)
        return None, dict(record, wall_seconds=perf_counter() - started)

    def _rank_paths(self, paths, lane):
        return _rank(self.arena, paths[..., self.position:self.position+3],
                     paths[..., self.velocity:self.velocity+3], lane)

    def confirm(self, ranked, tapes, record):
        arena, check = self.arena, self.check
        fields = {key: getattr(arena, key) for key in PHASE if hasattr(arena, key)}
        steering = len(self.axes) == 5
        best = None
        target = np.asarray(self.target())
        weights = np.array([5.8, 4.9, 4., .42])
        for candidate in ranked[:EXACT_CHECKS]:
            record['exact_trials'] += 1
            mujoco.mj_copyData(check.data, arena.model, arena.data)
            for key, value in fields.items():
                setattr(check, key, deepcopy(value))
            check.events = []
            check._rival_attempt = True
            tape = tapes[candidate['index']]
            check._aim = lambda *args: tape[check.steps - fields['steps']]
            check._choose_rival_response = lambda: None
            landing, landing_tick, choice, cost = None, None, None, 9.
            for _ in tape:
                _, _, done, capped, _ = check.step(np.zeros(arena.action_size), render=False)
                if check.rival_returns > arena.rival_returns:
                    if landing is None:
                        landing, landing_tick = check.ball.tolist(), check.steps
                    if not steering:
                        record['confirmed'] = True
                        return dict(candidate, exact_landing=landing, exact_tick=landing_tick)
                    x, y, z = check.ball - [0., 0., TABLE_HEIGHT]
                    vy = -check.velocity[1]
                    if -.1 + target[1] < y < -.72 and .055 < z < .46 and abs(x) < .40 and vy > .45:
                        sample = np.asarray((x, y, z, vy))
                        error = float(weights @ np.abs(sample - target))
                        if error < cost:
                            choice, cost = sample.tolist(), error
                    if check._last_hitter == 1 or y <= -1.10:
                        break
                if done or capped:
                    break
            if landing is not None and (best is None or cost < best['exact_cost']):
                best = dict(candidate, exact_landing=landing, exact_tick=landing_tick,
                            exact_corridor=choice, exact_cost=cost)
                if cost < .34:
                    break
        record['confirmed'] = best is not None
        return best


def install_rally_opponent(arena, lane='anchor', *, target_profile=None):
    """Install the practice rival without changing ball state or referee rules."""
    if lane not in ('anchor', 'varied'):
        raise ValueError('expected anchor or varied rally targets')
    if target_profile is not None:
        target_profile = tuple(map(float, target_profile))
        if len(target_profile) != 4 or not all(np.isfinite(target_profile)):
            raise ValueError('expected a finite x, depth, height and incoming speed')
    inherited = arena._aim
    response = arena._choose_rival_response
    reset = arena.reset
    opponent = RallyOpponent(arena, inherited, lane, target_profile)

    def opening(*args, **kwargs):
        opponent.reset()
        return reset(*args, **kwargs)

    def choose():
        response()
        arena._rival_attempt = True
        opponent.new_shot()

    def aim(side, bias=0., target_depth=.96, scale=1.):
        if side != -1:
            return inherited(side, bias, target_depth, scale)
        return opponent.command(bias, target_depth, scale)

    arena.reset = opening
    arena._choose_rival_response = choose
    arena._aim = aim
    return opponent

"""A fixed-lane physical blocking drill and its independent reference pose."""

from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from .tt4_arena import HOME as FAR_HOME, TABLE_HEIGHT, _racket_loft, mujoco
from .tt19_factory import FactoryModel, HOME, RATE, moving_joint_rate
from .tt19_flight import FlightLaw
from .tt19_so101 import AXES
from .tt20_curriculum import CLAMP_SHIFT
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import RivalSearch
from .tt21_coach_oracle import PHASE
from .tt22_opponent import RallyArena, _rank, _stroke, install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt22_rally import face
from .tt22_rally_trial import CONFIG, GentleDraws


DEVELOP = 92347591
OPENING_GRID = ((1.48, .72, .90), (1.59, .72, .90), (1.26, .72, .90), (1.48, .80, .90))
SETTLED_GRID = ((1.76, .95, .82, 1.), (1.76, .95, .86, .46), (1.56, .95, .78, 1.),
                (1.59, .72, .90, 1.), (1.66, .99, .90, 1.), (1.37, .72, .90, 1.))
PRACTICE_SPREAD = .12
PREP_TICKS = 56
FRAME = np.array([0., 0., TABLE_HEIGHT])
SEARCH_FRAMES = 71
FAST_DESIGN = qmc.Sobol(6, scramble=True, seed=473951).random_base2(6)


class Draws(GentleDraws):
    """Choose the initial release through the ordinary two-bounce generator."""

    def __init__(self, rng, spread, opening):
        pace, self.far, self.near = opening[:3]
        self.spin_scale = opening[3] if len(opening) == 4 else 1.
        super().__init__(rng, spread, pace)

    def uniform(self, low, high):
        value = super().uniform(low, high)
        if np.ndim(low) == 0 and low == .58:
            return self.far + value - .65
        if np.ndim(low) == 0 and low == .69:
            return self.near + value - .79
        if np.ndim(low) == 1 and low[0] == 55.:
            return np.asarray(value) * self.spin_scale
        return value


class BlockArena(RallyArena):
    def __init__(self, opening, spread=PRACTICE_SPREAD, *, release_seed=None):
        self.opening, self.spread, self.release_seed = opening, spread, release_seed
        super().__init__(**CONFIG)

    def _opening(self):
        if self.server == 1:
            return super()._opening()
        original = self._rng
        release = original if self.release_seed is None else np.random.default_rng(
            np.random.SeedSequence((self.release_seed, 0, 4)))
        self._rng = Draws(release, self.spread, self.opening)
        try:
            super()._opening()
        finally:
            self._rng = original


def requests(goal):
    """Approach one motor posture from ordinary home using bounded rates."""
    current = HOME.copy()
    while True:
        rate = np.clip((goal - current) / RATE[:5], -1., 1.)
        current += RATE[:5] * rate
        yield np.r_[rate, 0.]


def prepare(arena, seed, goal):
    """Settle through real motor requests before the only release of the point."""
    arena.reset(seed=seed, server=-1, render=False)
    q, v = arena._ball_joint, arena._ball_dof
    position = arena.data.qpos[q:q+7].copy()
    velocity = arena.data.qvel[v:v+6].copy()
    # No service physics run until the pre-point preparation is finished.
    arena.ball[:] = (1.55, 1.8, .4)
    arena.data.qvel[v:v+6] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    axes = arena._arm_ids[1][2]
    micro = round(arena.control_dt / arena.model.opt.timestep)
    low, high = arena._joint_low[axes], arena._joint_high[axes]
    tape = []
    planned = requests(np.asarray(goal))
    for _ in range(PREP_TICKS):
        request = next(planned)
        tape.append(request.tolist())
        delta = request * arena._joint_increments[axes]
        for _ in range(micro):
            arena.data.ctrl[axes] = np.clip(arena.data.ctrl[axes] + delta, low, high)
            mujoco.mj_step(arena.model, arena.data)
    arena.data.qpos[q:q+7] = position
    arena.data.qvel[v:v+6] = velocity
    arena.data.qacc_warmstart[v:v+6] = 0.
    arena.data.xfrc_applied[arena._ball_body] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    return tape


def first_return(arena, seed, goal, *, settled=False):
    """One official return opportunity. Stop at credit or referee fault."""
    if settled:
        warmup = prepare(arena, seed, goal)
        goal_requests = requests(HOME)
    else:
        arena.reset(seed=seed, server=-1, render=False)
        warmup = []
        goal_requests = requests(np.asarray(goal))
    arena._choose_rival_response = lambda: setattr(arena, '_rival_attempt', False)
    joint_speeds = [int(arena.model.joint('learner_' + key).dofadr[0]) for key in AXES[:5]]
    actions, impact = [], None
    for action in goal_requests:
        before = arena.contacts
        speed = arena.data.qvel[joint_speeds].copy()
        actions.append(action.tolist())
        _, reward, done, timeout, _ = arena.step(action, render=False)
        if arena.contacts > before and impact is None:
            impact = dict(tick=arena.steps, max_joint_speed=float(np.max(abs(speed))), request=action.tolist())
        if arena.legal_returns or done or timeout:
            break
    return dict(seed=seed, goal=np.asarray(goal).tolist(), preparation=warmup, actions=actions, impact=impact, reward=reward,
                legal=bool(arena.legal_returns), outcome=arena.point_result(), events=arena.events.copy())


def reference_candidates(settled=False):
    factory = FactoryModel()
    factory.base += CLAMP_SHIFT
    depths = (-.90, -.95, -1.00, -1.052) if settled else (-.85, -.90, -.95, -1.00)
    heights = (.107, .137, .170, .204, .239) if settled else (.085, .119, .154, .199, .24)
    pitches = (.19, .31, .43, .55, .67, .79, .91) if settled else (.19, .31, .43, .55, .67, .79)
    for depth, height, pitch in product(depths, heights, pitches):
        target = np.array([0., depth, height])
        normal = face(pitch, 0.)
        joints = factory.solve(target, normal, HOME)
        reached, fitted, *_ = factory.pose(joints)
        if np.linalg.norm(reached - target) < .067 and normal @ fitted > .8:
            yield dict(joints=joints.tolist(), designed=[0., depth, height, pitch])


def screen(*, settled=False):
    """Geometry may prepare the independent reference. The learner never reads it."""
    evaluated = []
    goals = list(reference_candidates(settled))
    openings = SETTLED_GRID if settled else OPENING_GRID
    for opening in openings:
        arena = BlockArena(opening)
        try:
            for reference in goals:
                row = first_return(arena, DEVELOP, reference['joints'], settled=settled)
                if row['legal']:
                    landed = next(e for e in row['events'] if e['kind'] == 'legal_return' and e['side'] == 1)
                    impact = next(e for e in row['events'] if e['kind'] == 'contact' and e['side'] == 1)
                    speed = row['impact']['max_joint_speed']
                    merit = (abs(landed['ball'][1] - .59) + 1.2 * abs(landed['ball'][0])
                             + .5 * min(1., speed) + .35 * max(0., 2.85 + impact['incoming'][1]))
                    evaluated.append(dict(reference, opening=opening, score=float(merit), trace=row))
        finally:
            arena.close()
    return dict(candidates=len(goals) * len(openings), valid=sorted(evaluated, key=lambda row: row['score']))


def play_block(arena, seed, goal, tape, rival, *, settled=False, pulse=None):
    """Play the full uninterrupted point using one prepared near posture."""
    if settled:
        warmup = prepare(arena, seed, goal)
        goal_requests = requests(HOME)
    else:
        arena.reset(seed=seed, server=-1, render=False)
        warmup = []
        goal_requests = requests(np.asarray(goal))
    tape.steps.clear()
    dofs = [int(arena.model.joint('learner_' + key).dofadr[0]) for key in AXES[:5]]
    actions, contacts, optical = [], [], []
    camera = None if pulse is None else NaturalViewTransport('fixed', lens_fov=arena.lens_intrinsics(),
                                                            law=FlightLaw(), flight=False, associate=True).tracker()
    driver = None if pulse is None else PulseDriver(pulse)
    for tick, action in enumerate(goal_requests):
        if camera is not None:
            camera.update(arena.render_cameras(), tick=tick)
            ball = camera.ball
            position = None if ball is None or not ball.observed else list(map(float, ball.position))
            optical.append(position)
            action = driver.command(tick, None if position is None else position[1])
        before = arena.contacts
        speed = arena.data.qvel[dofs].copy()
        blade = arena.paddle.copy()
        actions.append(action.tolist())
        _, reward, done, timeout, _ = arena.step(action, render=False)
        if arena.contacts > before:
            contacts.append(dict(tick=arena.steps, velocity=speed.tolist(), request=action.tolist(),
                                 blade_move_m=float(np.linalg.norm(arena.paddle - blade))))
        if done or timeout:
            break
    return dict(seed=seed, goal=np.asarray(goal).tolist(), preparation=warmup, actions=actions,
                rival_actions=tape.tape(len(actions)), contacts=contacts,
                optical=optical, rgb=None if camera is None else camera.digest.hexdigest(),
                pulses=[] if driver is None else driver.triggers, decisions=rival.decisions.copy(), reward=reward,
                outcome=arena.point_result(), events=arena.events.copy())


def simple_reference(reference, seed=DEVELOP):
    arena = BlockArena(reference['opening'])
    _, depth, height, pitch = reference['designed']
    target = (0., depth + .026 * np.cos(pitch), height + .026 * np.sin(pitch), 3.32, 0., .55)
    coach = install_rally_opponent(arena, target_profile=target)
    tape = FarTape(arena)
    try:
        return play_block(arena, seed, reference['joints'], tape, coach)
    finally:
        arena.close()


class PulseDriver:
    """One short actuator pulse after a visible inbound crossing, then restore posture."""

    def __init__(self, pulse, *, armed=False):
        self.pulse = pulse
        self.armed = armed
        self.previous = None
        self.start = None
        self.triggers = []

    def command(self, tick, y):
        if y is not None:
            if self.previous is not None:
                time, previous = self.previous
                motion = (y - previous) / (tick - time)
                if y > .07 and motion > .026:
                    self.armed = True
                if self.armed and y < self.pulse['cue'] and motion < -.028:
                    self.armed = False
                    self.start = tick
                    self.triggers.append(tick)
            self.previous = tick, y
        if self.start is None:
            return np.zeros(6)
        age = tick - self.start
        if age < 3:
            return np.asarray(self.pulse['rates'])
        if 7 <= age < 10:
            return -np.asarray(self.pulse['rates'])
        return np.zeros(6)


def reference_pulse(goal, cue, pace):
    factory = FactoryModel()
    factory.base += CLAMP_SHIFT
    velocity = moving_joint_rate(factory, np.asarray(goal), pace)
    return dict(cue=cue, requested_pace=pace, rates=np.r_[velocity / RATE[:5], 0.].tolist())


class ReferenceShadow:
    """A separate arena with one independently prepared, stationary learner arm."""

    def __init__(self, opponent, goal, pulse):
        self.opponent = opponent
        self.pulse = pulse
        original = opponent.arena
        arena = self.arena = BlockArena(original.opening, original.spread, release_seed=original.release_seed)
        arena.reset(seed=DEVELOP, server=-1, render=False)
        arena.ball[:] = (1.55, 1.8, .4)
        arena.velocity[:] = 0.
        axes = arena._arm_ids[1][2]
        micro = opponent.micro
        action = requests(np.asarray(goal))
        for tick in range(70):
            request = next(action)
            arena.data.ctrl[axes] = np.clip(arena.data.ctrl[axes] + request * micro * arena._joint_increments[axes],
                                            arena._joint_low[axes], arena._joint_high[axes])
            mujoco.mj_step(arena.model, arena.data, nstep=micro)
        paddle_joint = arena.model.joint('learner_paddle_free')
        q, d = int(paddle_joint.qposadr[0]), int(paddle_joint.dofadr[0])
        self.qpos = np.r_[arena._arm_ids[1][0], np.arange(q, q + 7)]
        self.qvel = np.r_[[int(arena.model.joint('learner_' + key).dofadr[0]) for key in AXES], np.arange(d, d + 6)]
        self.axes = axes
        self.position = arena.data.qpos[self.qpos].copy()
        self.velocity = arena.data.qvel[self.qvel].copy()
        self.targets = arena.data.ctrl[axes].copy()
        self.blade = arena.paddle.copy() - FRAME
        self.normal = -arena.data.geom_xmat[arena._racket_ids[1]].reshape(3, 3)[:, 2].copy()
        self.search = RivalSearch(arena, arena._aim)

    def restore(self):
        original, arena = self.opponent.arena, self.arena
        mujoco.mj_copyData(arena.data, arena.model, original.data)
        # The opponent's reference pose is independent of the live pupil.
        arena.data.qpos[self.qpos] = self.position
        arena.data.qvel[self.qvel] = self.velocity
        arena.data.qacc_warmstart[self.qvel] = 0.
        arena.data.ctrl[self.axes] = self.targets
        for key in PHASE:
            if hasattr(original, key):
                setattr(arena, key, deepcopy(getattr(original, key)))
        arena.events = []
        arena._rival_attempt = True
        arena._choose_rival_response = lambda: None
        mujoco.mj_forward(arena.model, arena.data)

    def confirm(self, tape):
        self.restore()
        arena = self.arena
        initial_rival, initial_near = arena.rival_returns, arena.legal_returns
        begin = arena.steps

        def command(*args):
            elapsed = arena.steps - begin
            if elapsed < len(tape) and arena.rival_returns == initial_rival:
                return tape[elapsed]
            return arena._command_to(-1, FAR_HOME, .86)

        arena._aim = command
        legal = None
        driver = None if self.pulse is None else PulseDriver(self.pulse, armed=True)
        for _ in range(min(len(tape) + 28, arena.max_steps - arena.steps)):
            near = np.zeros(arena.action_size) if driver is None else driver.command(arena.steps, arena.ball[1])
            _, _, done, capped, _ = arena.step(near, render=False)
            if legal is None and arena.rival_returns > initial_rival:
                legal = next(e for e in arena.events if e['kind'] == 'legal_return' and e['side'] == -1)
            if arena.legal_returns > initial_near:
                receive = next(e for e in arena.events if e['kind'] == 'legal_return' and e['side'] == 1)
                return legal, receive
            if done or capped:
                break
        return legal, None

    def close(self):
        self.arena.close()


def _strong_goals(arena, options, target, frames):
    timeline = np.arange(frames) * arena.control_dt
    goals, params = [], []
    for hit_index, (time, ball, incoming) in enumerate(options):
        for pace_bias, phase, loft, side, lift, steer in FAST_DESIGN:
            pace = .25 + 3.1 * pace_bias
            lead = .018 + .115 * phase
            base = _racket_loft(-incoming[1], incoming[2], ball[2], ball[1] + .70, pace)
            pitch = float(np.clip(base - .28 + .54 * loft, -.59, .88))
            lateral = float(np.clip(2.4 * (target[0] - ball[0]), -.72, .72) + (side - .50) * 1.6)
            rise = float(-.50 + 1.15 * lift)
            correction = ((target[0] - ball[0]) / .5 - incoming[0] - .18 * lateral) / (max(1.2, incoming[1]) + 4.)
            yaw = float(np.clip(correction + .35 * (steer - .50), -.43, .43))
            goals.append(_stroke(arena, timeline, time, ball, pace, lead, pitch, lateral, rise, yaw))
            params.append([hit_index, float(pace), float(lead), pitch, lateral, rise, yaw])
    return np.asarray(goals), params


def _reaction_bonus(positions, velocity, receive_tick):
    shifted = np.flatnonzero((velocity[:-3, 1] < -.25) & (velocity[3:, 1] > .30)
                            & (positions[2:-1, 1] < -.73) & (np.arange(len(positions) - 3) > receive_tick))
    if not len(shifted):
        return 0.
    after = int(shifted[0] + 3)
    position, speed = positions[after:], velocity[after:]
    returned = np.flatnonzero((speed[:-1, 2] < 0.) & (speed[1:, 2] > 0.) & (position[:-1, 1] > .035)
                              & (position[:-1, 1] < .82) & (abs(position[:-1, 0]) < .427)
                              & (position[:-1, 2] < TABLE_HEIGHT + .044))
    return 10. if len(returned) else 3.


class BlockCoach:
    def __init__(self, arena, reference, *, pace=3.40, rise=1.25, pulse=None):
        self.rival = install_rally_opponent(arena)
        self.shadow = ReferenceShadow(self.rival, reference, pulse)
        x, y, z = self.shadow.blade + .025 * self.shadow.normal
        self.target = (float(x), float(y), float(z), pace, 0., rise)
        self.rival.plan = self.plan

    @property
    def decisions(self):
        return self.rival.decisions

    def plan(self):
        began = perf_counter()
        original, shadow = self.rival.arena, self.shadow
        frames = min(SEARCH_FRAMES, original.max_steps - original.steps)
        windows = self.rival.incoming.windows(frames)
        stats = dict(legal=0, checked=0, compatible=False, target=list(self.target), candidates=0)
        if not windows:
            return None, dict(stats, wall_seconds=perf_counter() - began)
        goals, params = _strong_goals(original, windows, self.target, frames)
        shadow.restore()
        tapes, paths = shadow.search.predict(goals)
        p, v = shadow.search.position, shadow.search.velocity
        xyz, speeds = paths[..., p:p+3], paths[..., v:v+3]
        ranked = _rank(shadow.arena, xyz, speeds, self.target)
        stats.update(legal=len(ranked), candidates=len(goals))
        for row in ranked:
            index = row['index']
            row['block_bonus'] = _reaction_bonus(xyz[index], speeds[index], row['strike_micro'])
        ranked.sort(key=lambda row: row['cost'] - row['block_bonus'])
        chosen = None
        for candidate in ranked[:18]:
            stats['checked'] += 1
            index = candidate['index']
            reply, block = shadow.confirm(tapes[index])
            if reply is None:
                continue
            option = (tapes[index], dict(candidate, parameters=params[index], checked_reply=reply, checked_block=block))
            if chosen is None or block is not None:
                chosen = option
            if block is not None:
                stats['compatible'] = True
                break
        if chosen is None:
            return None, dict(stats, wall_seconds=perf_counter() - began)
        return chosen[0], dict(stats, **chosen[1], wall_seconds=perf_counter() - began)

    def close(self):
        self.shadow.close()

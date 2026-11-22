"""Carry a duration-selected commitment gate from native Pong to SO-101."""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from itertools import product

import numpy as np

from .tt3_numeric import NumericMemory
from .tt3_pong import ACTIONS, CAMERA_SECONDS, _native_rallies
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_factory import RATE
from .tt22_duration_pong import CAP, CHECKPOINT as LATERAL, OPPONENT_SPEED, ROLE_MODEL, OpticalDurationAgent, cases
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_match import CHECKPOINT as PRACTICE, summarize
from .tt22_opponent import RallyArena
from .tt22_ordinary_game import DIR as TT22, FrozenMemory, StandardMatch
from .tt22_rally_trial import CONFIG


DIR = ROOT/'artifacts/tt23'
PONG = DIR/'timing-pong.json.gz'
TARGET = DIR/'timing-live.json.gz'
OPTIONS = tuple((a, b) for b, a in product((.5, 1.), (.65, 1., 1.4, 2.)))
PERMUTE = 5
TRAIN, HELD = 1441000000, 1481000000
PONG_POINTS = 16
TARGET_SEED = 1372569301
TARGET_POINTS = 12


def ready(rule, eta, travel, lag):
    if rule is None:
        return True
    travel_weight, lag_weight = rule
    return eta <= travel_weight * travel + lag_weight * lag


class ProbeResponse(ShorterDwell):
    """Read steady speed and delay from the existing visual actuator probe."""

    def observe(self, binding, preceding_action):
        already_fit = self.axis is not None
        super().observe(binding, preceding_action)
        if already_fit or self.axis is None:
            return
        _, integrals, position = zip(*self.history[self.identity], strict=True)
        negative, _, positive = map(ACTIONS.index, self.effects)
        x = np.asarray(integrals)[..., positive] - np.asarray(integrals)[..., negative]
        x -= x.mean(axis=0)
        y = np.asarray(position) @ self.axis
        y -= y.mean()
        speed = (y @ x) / (np.square(x).sum(axis=0) + 1e-8)
        residual = np.square(y[:, None] - x * speed).sum(axis=0)
        index = int(np.argmin(residual))
        self.speed = float(speed[index] * CAMERA_SECONDS)
        self.lag = float((1. / self.alpha[index, :, 0] - 1.).sum() + .5)
        if self.speed <= 0.:
            raise ValueError('visual probe measured nonpositive speed along the fitted axis')


class TimingPong(OpticalDurationAgent):
    def __init__(self, roles, memory, case, rule):
        self.rule = rule
        super().__init__(roles, memory, case.exploration, option=0, motor_factory=ProbeResponse)

    def begin_point(self):
        super().begin_point()
        self.committed = False
        self.waited = 0
        self.approaches = 0

    def choose(self):
        if self.motor.axis is None:
            return self.motor.probe()
        if self.actor is None:
            return self.motor.effects[1]

        axis = self.motor.axis
        position = np.asarray(self.actor.position)
        actor_drift = 6. * (self.actor_history[-1] - self.actor_history[0]) / max(1, len(self.actor_history) - 1)
        recovered = float((self.motor.ready - position - actor_drift) @ axis)
        if len(self.mover_history) < 2:
            return self.motor.action(recovered)

        normal = np.array((-axis[1], axis[0]))
        moving = self.mover_history[-1]
        velocity = (moving - self.mover_history[0]) / (len(self.mover_history) - 1)
        gap = float((moving - self.motor.ready) @ normal)
        rate = float(velocity @ normal)
        if abs(rate) < 1e-6:
            return self.motor.effects[1]
        if gap * rate >= 0.:
            self.committed = False
            return self.motor.action(recovered)

        goal = self._goal(position)
        travel = abs(float((goal - position) @ axis)) / self.motor.speed
        if not self.committed:
            if not ready(self.rule, -gap / rate, travel, self.motor.lag):
                self.waited += 1
                return self.motor.action(recovered)
            self.approaches += 1
            self.committed = True
        return self.motor.action(float((goal - position - actor_drift) @ axis))


def source_play(starts, rules):
    roles = RoleModel(**_read(ROLE_MODEL))
    memory = NumericMemory.restore(_read(LATERAL))
    scheduled = [(case, name, rule) for case in starts for name, rule in rules.items()]
    groups = {name: [] for name in rules}
    for start in range(0, len(scheduled), 48):
        batch = scheduled[start:start + 48]
        agents = [TimingPong(roles, memory, case, rule) for case, _, rule in batch]
        run = _native_rallies([case for case, _, _ in batch], agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for row, (_, name, _), agent in zip(run, batch, agents, strict=True):
            groups[name].append(dict(row, waited=agent.waited, approaches=agent.approaches,
                                     speed=getattr(agent.motor, 'speed', None), lag=getattr(agent.motor, 'lag', None)))
    return groups


def pong_summary(rows):
    return dict(seconds=float(np.mean([row['ticks'] for row in rows])) / 120.,
                cap=sum(row['ticks'] == CAP for row in rows),
                returns=sum(len([contact for contact in row['contacts'] if contact['returned']]) for row in rows),
                waited=sum(row['waited'] for row in rows), approaches=sum(row['approaches'] for row in rows))


def source():
    import torch

    torch.set_num_threads(1)
    candidates = {str(i): option for i, option in enumerate(OPTIONS)}
    practice = cases(PONG_POINTS, TRAIN)
    train = source_play(practice, candidates)
    ticks = [sum(row['ticks'] for row in train[str(i)]) for i in range(len(candidates))]
    selected = int(np.argmax(ticks))
    permutation = np.roll(ticks, PERMUTE).tolist()
    permuted = int(np.argmax(permutation))
    frozen = dict(pong=candidates[str(selected)], permuted=candidates[str(permuted)], eager=None)
    result = dict(schema='tt23-pong-timing-1', candidates=candidates, duration_ticks=ticks,
                  permuted_duration_ticks=permutation, selected=selected, permuted=permuted, frozen=frozen,
                  role_sha256=_hash(ROLE_MODEL), lateral_sha256=_hash(LATERAL), max_native_ticks=CAP,
                  opponent_speed=OPPONENT_SPEED, training=train, training_cases=[asdict(case) for case in practice])
    _write(PONG, result)
    print('Pong training totals', ticks, 'frozen', frozen, flush=True)
    test = cases(PONG_POINTS, HELD)
    evaluate = source_play(test, frozen)
    replay = source_play(test[:1], frozen)
    if any(replay[name][0] != rows[0] for name, rows in evaluate.items()):
        raise ValueError('Pong controller replay changed a held match')
    result.update(held=evaluate, held_cases=[asdict(case) for case in test], verified=len(replay),
                  summary={name: pong_summary(rows) for name, rows in evaluate.items()})
    _write(PONG, result)
    print('Pong held', result['summary'], flush=True)


def nominal_response_lag(player):
    """Measure the fixed command twin's lag in its own control frames."""
    twin = player.sim.twin
    qpos = twin.arena._arm_ids[1][0][:5]
    delayed = []
    for axis in range(len(qpos)):
        for sign in (-1., 1.):
            twin.reset()
            for _ in range(22):
                drive = twin.data.ctrl[twin.axes[:5]]
                twin.advance(np.r_[np.clip((player.neutral - drive) / RATE[:5], -1., 1.), 0.])
            joint, actuator = qpos[axis], twin.axes[axis]
            static = twin.data.ctrl[actuator] - twin.data.qpos[joint]
            command = np.zeros(6)
            command[axis] = .35 * sign
            for frame in range(8):
                twin.advance(command)
                if frame >= 4:
                    gap = twin.data.ctrl[actuator] - twin.data.qpos[joint] - static
                    delayed.append(gap / (command[axis] * RATE[axis]))
    twin.reset()
    return float(np.median(delayed))


class TimingArm(InversePlayer):
    def __init__(self, law):
        super().__init__(law)
        self.rule = None
        self.lag = nominal_response_lag(self)
        self.qpos = self.sim.twin.arena._arm_ids[1][0][:5]
        self.proposals = {}

    def reset(self):
        super().reset()
        self.gates = []

    def search(self, estimate, observation, target):
        # Shared prefixes often ask the deterministic stock planner exactly
        # the same expensive question. Cache its proposal before applying the gate.
        state = self.sim.twin.state()
        history = tuple((tick, None if point is None else tuple(point)) for tick, point in self.history[-21:])
        key = (history, tuple(estimate[0]), tuple(estimate[1]), tuple(observation.center),
               tuple(observation.velocity), tuple(observation.spin), observation.bounced,
               state.physics.tobytes(), state.targets.tobytes())
        if key not in self.proposals:
            self.proposals[key] = super().search(estimate, observation, target)
        choice = deepcopy(self.proposals[key])
        if choice is None:
            return None
        tape, shot = choice
        eta = shot['prediction']['eta']
        frame = int(eta)
        if frame >= len(tape):
            # The dry rollout unwinds accumulated commands after the tape.
            unwind = eta - len(tape)
            displacement = tape.sum(axis=0)
            displacement -= np.clip(displacement, -unwind, unwind)
        else:
            displacement = tape[:frame].sum(axis=0) + (eta - frame) * tape[frame]
        joint = self.sim.twin.data.qpos[self.qpos]
        travel = float(np.max(np.abs((target - joint) / RATE[:5] + displacement)))
        allow = ready(self.rule, eta, travel, self.lag)
        self.gates.append(dict(tick=self.history[-1][0], eta=eta, travel=travel, allow=allow))
        return choice if allow else None


def replay_points(branches):
    arena = RallyArena(**CONFIG)
    original_aim, original_direct = arena._aim, arena._command_to
    try:
        for rows in branches.values():
            for row in rows:
                arena._aim, arena._command_to = original_aim, original_direct
                arena.reset(seed=row['seed'], server=row['role'], render=False)
                for index, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), start=1):
                    rates = np.asarray(far)
                    arena._aim = arena._command_to = lambda *args, rates=rates: rates
                    _, reward, done, capped, _ = arena.step(near, render=False)
                    if bool(done or capped) != (index == len(row['actions'])):
                        raise ValueError(f"point length changed at {row['seed']}:{index}")
                if (reward != row['reward'] or arena.point_result() != row['outcome']
                        or json.dumps(arena.events) != json.dumps(row['events'])):
                    raise ValueError(f"referee replay changed at {row['seed']}")
    finally:
        arena.close()
    return {name: len(rows) for name, rows in branches.items()}


def paired(branches):
    eager = branches['eager']
    results = {}
    for name, rows in branches.items():
        first_difference, prefix = [], []
        for row, control in zip(rows, eager, strict=True):
            limit = min(len(row['actions']), len(control['actions']))
            tick = next((i for i in range(limit) if row['actions'][i] != control['actions'][i]), None)
            prefix_limit = limit if tick is None else tick + 1
            if row['rival_actions'][:prefix_limit] != control['rival_actions'][:prefix_limit]:
                raise ValueError(f"rival diverged before the gate intervened on {row['seed']}")
            optical = [tuple(sample) for sample in row['optical'] if sample[0] < prefix_limit]
            if optical != [tuple(sample) for sample in control['optical'] if sample[0] < prefix_limit]:
                raise ValueError(f"optics diverged before the gate intervened on {row['seed']}")
            first_difference.append(tick)
            prefix.append(prefix_limit)
        base = summarize(rows)
        base.update(total_alternations=sum(base['lengths']), learner_points=sum(row['reward'] > 0. for row in rows),
                    deferrals=sum(not gate['allow'] for row in rows for gate in row['gates']),
                    first_different_command=first_difference, shared_prefix_frames=prefix)
        results[name] = base
    return results


def target():
    checkpoint = _read(PONG)
    if 'verified' not in checkpoint:
        raise ValueError('complete the held Pong comparison before running the frozen target branches')
    frozen = checkpoint['frozen']
    profile = _read(TT22/'tt22_opponent-learned.json.gz')['fit']['profile']
    practice = _read(PRACTICE)
    memory = FrozenMemory(practice['evidence'])
    seeds = np.random.default_rng(TARGET_SEED).integers(2**31, size=TARGET_POINTS).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds) & {row['seed'] for row in practice['practice']}:
        raise ValueError('target point seeds must be unique and absent from target practice')
    roles = (-1, -1, 1, 1) * (TARGET_POINTS // 4)
    station = StandardMatch(profile)
    station.player.close()
    station.player = player = TimingArm(station.optics.law)
    metadata = dict(schema='tt23-so101-timing-1', source_sha256=_hash(PONG), practice_sha256=_hash(PRACTICE),
                    seed=TARGET_SEED, seeds=seeds, roles=list(roles), rule=frozen,
                    arm_lag=player.lag, arena=CONFIG, rival=profile)
    metadata = json.loads(json.dumps(metadata))
    result = _read(TARGET) if TARGET.exists() else dict(metadata=metadata, branches={name: [] for name in frozen})
    branches = result['branches']
    if result['metadata'] != metadata:
        raise ValueError('existing target receipt has different frozen settings')
    try:
        for i, (seed, role) in enumerate(zip(seeds, roles, strict=True)):
            player.proposals.clear()
            order = tuple(frozen)[i % len(frozen):] + tuple(frozen)[:i % len(frozen)]
            for name in order:
                rows = branches[name]
                if i < len(rows):
                    continue
                station.optics.registry.known.clear()
                station.service_optics.registry.known.clear()
                player.rule = frozen[name]
                row = station.play(seed, role, memory)
                rows.append(dict(row, gates=player.gates))
                _write(TARGET, result)
                print('SO-101', i + 1, name, row['longest'], row['reward'], len(row['actions']),
                      sum(not gate['allow'] for gate in player.gates), flush=True)
    finally:
        station.close()
    result.update(verified=replay_points(branches), summary=paired(branches))
    _write(TARGET, result)
    print('SO-101 held', result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('source', 'target', 'both'))
    phase = parser.parse_args().phase
    if phase in ('source', 'both'):
        source()
    if phase in ('target', 'both'):
        target()

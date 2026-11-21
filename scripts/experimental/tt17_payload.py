"""Carry a calibrated actuator map into an RGB gate-and-dock task."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_control import MOTOR_BUDGET
from .tt6_memory import load
from .tt10_target import TargetArena
from .tt12_motor_recall import rewire
from .tt14_gain_hardware import GainWiring
from .tt14_gain_trial import load_map
from .tt16_payload import BLOCK_FRAMES, MODEL_SCALE, PIXELS, PULSE_FRAMES, VIEW_SCALE, course, display, see


HOLDOUT_SEED = 761025341
DELIVERY_BLOCKS = 29
BEFORE = 2
AFTER = 2
SWITCH = BEFORE * DELIVERY_BLOCKS
GATE_RADIUS = .054
DOCK_RADIUS = .036
DOCK_SPEED = .019
DOCK_HOLD = 2
PROBES = 3
MODES = ('selective', 'reset_both', 'continuous')
CHANGES = ('motor', 'payload', 'sham')
BASELINES = ('signed', 'original', 'cold')
MOTOR_PENALTY = .75
OBJECT_PENALTY = (.05, .05, .07, .07, .07)


def scenario(seed, matrix):
    """Evaluator-only payload physics and route generation."""
    coupling, damping, _ = course(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3617]))
    turn = rng.choice((-1., 1.)) * rng.uniform(.88, 1.73)
    co, si = np.cos(turn), np.sin(turn)
    rotated = np.array([[co, -si], [si, co]]) @ coupling
    routes = []
    last = np.zeros(2)
    while len(routes) != BEFORE + AFTER:
        gate = rng.uniform(-.22, .22, 2)
        dock = rng.uniform(-.26, .26, 2)
        approach, departure = np.linalg.norm(gate - last), np.linalg.norm(dock - gate)
        if .14 < approach < .29 and .14 < departure < .28:
            routes.append([gate.tolist(), dock.tolist()])
            last = dock
    return dict(seed=seed, routes=routes, start_coupling=coupling.tolist(), changed_coupling=rotated.tolist(),
                damping=damping.tolist(), start_motor=matrix.tolist(), changed_motor=(.47 * matrix).tolist())


class Regression:
    """Maintain a prior and normal equations for one causal link."""

    def __init__(self, penalty, baseline):
        self.penalty = np.diag(penalty)
        self.baseline = baseline
        self.reset()

    def reset(self):
        self.gram = self.penalty.copy()
        self.cross = self.penalty @ self.baseline.T

    def observe(self, features, value):
        self.gram += np.outer(features, features)
        self.cross += np.outer(features, value)

    def fit(self):
        return np.linalg.solve(self.gram, self.cross).T


class ChangeMonitor:
    """Require repeated predictive errors in recent informative observations."""

    def __init__(self):
        self.window = deque(maxlen=5)

    def observe(self, predicted, actual, *, minimum, floor, scale):
        informative = float(np.linalg.norm(predicted))
        excess = float(np.linalg.norm(actual - predicted)) - floor - scale * informative
        self.window.append((max(informative, float(np.linalg.norm(actual))) >= minimum, excess))
        return sum(useful and above > 0 for useful, above in self.window) >= 3


class Player:
    """A single visual learner with separate predictions for tool and payload motion."""

    def __init__(self, prior, seed, mode):
        self.width = prior.shape[1]
        self.mode = mode
        self.motor = Regression(np.full(self.width, MOTOR_PENALTY), PULSE_FRAMES * prior / MODEL_SCALE)
        self.payload = Regression(OBJECT_PENALTY, np.column_stack((.5 * np.eye(2), np.zeros((2, 3)))))
        self.monitors = {name: ChangeMonitor() for name in ('motor', 'payload')}
        self.histories = {name: deque(maxlen=5) for name in self.monitors}
        self.rng = np.random.default_rng(seed)
        self.command = np.zeros(self.width)
        self.requested = self.command.copy()
        self.previous = None
        self.done = False
        self.tick = 0
        self.last_alarm = -12
        self.alarms = []

    def learn(self, tool, last_tool, puck, last_puck, last_velocity):
        velocity = puck - last_puck
        values = {'motor': (self.command, tool - last_tool), 'payload': (np.r_[last_velocity, tool], velocity)}
        flags = []
        parameters = {'motor': dict(minimum=.31, floor=.145, scale=.20),
                      'payload': dict(minimum=.29, floor=.145, scale=.16)}
        for name, (features, actual) in values.items():
            model = getattr(self, name)
            predicted = model.fit() @ features
            monitor = self.monitors[name]
            alarm = monitor.observe(predicted, actual, **parameters[name])
            if alarm:
                flags.append(name)
            self.histories[name].append((features, actual))
            model.observe(features, actual)
        if flags and self.tick >= 20 and self.tick - self.last_alarm >= 7:
            self.last_alarm = self.tick
            self.alarms.append(dict(block=self.tick, links=flags))
            reset = flags if self.mode == 'selective' else tuple(values) if self.mode == 'reset_both' else ()
            for name in reset:
                model = getattr(self, name)
                model.reset()
                for features, value in list(self.histories[name])[-3:]:
                    model.observe(features, value)
            for monitor in self.monitors.values():
                monitor.window.clear()
        return velocity

    def act(self, markers, reward):
        tool, puck, aim = np.asarray(markers)
        puck, aim = puck[:2], aim[:2]
        velocity = np.zeros(2)
        if self.previous is not None:
            last_tool, last_puck, last_velocity, last_aim = self.previous
            self.done = (self.done or bool(reward)) and np.array_equal(aim, last_aim)
            velocity = self.learn(tool, last_tool, puck, last_puck, last_velocity)
        self.previous = tool, puck, velocity, aim

        motor, payload = self.motor.fit(), self.payload.fit()
        wanted = np.clip(.33 * (aim - puck), -.84, .84)
        change = wanted - payload @ np.r_[velocity, tool]
        effect = payload[:, 2:] @ motor
        ridge = .012 * motor.T @ motor + .016 * np.eye(self.width)
        command = np.linalg.solve(effect.T @ effect + ridge, effect.T @ change - .012 * motor.T @ tool)
        exploration = self.rng.uniform(-1., 1., self.width)
        if self.tick < PROBES:
            command = .63 * exploration
        elif self.tick - self.last_alarm < 5:
            command += .20 * exploration
        elif not self.done:
            command += max(.05, .28 / np.sqrt(self.tick - 1)) * exploration
        lower = np.maximum(-.82, (-MOTOR_BUDGET - self.requested) / PULSE_FRAMES)
        upper = np.minimum(.82, (MOTOR_BUDGET - self.requested) / PULSE_FRAMES)
        self.command = np.clip(command, lower, upper)
        self.requested += PULSE_FRAMES * self.command
        self.tick += 1
        return self.command


def crossed(start, stop, gate):
    span = stop - start
    fraction = np.clip((gate - start) @ span / max(span @ span, 1e-12), 0., 1.)
    return np.linalg.norm(start + fraction * span - gate) <= GATE_RADIUS


def play(arena, wiring_seed, world, perturb, *, player=None, tape=None, deliveries=BEFORE + AFTER):
    """Advance MuJoCo and then the separate payload model on each action block."""
    initial = GainWiring(arena, wiring_seed, world['start_motor'])
    changed = GainWiring(arena, wiring_seed, world['changed_motor']) if perturb == 'motor' else initial
    initial.reset(seed=world['seed'], render=False)
    first, alternative = [np.asarray(world[key]) for key in ('start_coupling', 'changed_coupling')]
    damping = np.asarray(world['damping'])
    zero = np.zeros(arena.action_size)
    puck, velocity = np.zeros((2, 2))
    reward = 0
    rows = []
    for i, (gate, dock) in enumerate(np.asarray(world['routes'])[:deliveries]):
        wiring = initial if i < BEFORE else changed
        coupling = alternative if perturb == 'payload' and i >= BEFORE else first
        through, delivered, hold = False, False, 0
        for _ in range(DELIVERY_BLOCKS):
            goal = dock if through else gate
            marks = see(display(arena.paddle - arena.origin, puck, goal))
            command = player.act(marks, reward) if tape is None else np.asarray(tape[len(rows)])
            if not np.isfinite(command).all() or np.max(abs(command)) > 1.:
                raise ValueError('invalid command')
            for tick in range(BLOCK_FRAMES):
                wiring.step(command if tick < PULSE_FRAMES else zero, render=False)
            tool = arena.paddle - arena.origin
            velocity = damping @ velocity + coupling @ tool
            following = puck + velocity
            pass_gate = not through and bool(crossed(puck, following, gate))
            puck = following
            distance = float(np.linalg.norm(puck - goal))
            inside = through and distance <= DOCK_RADIUS and np.linalg.norm(velocity) <= DOCK_SPEED
            hold = hold + 1 if inside else 0
            reward = int(hold >= DOCK_HOLD and not delivered)
            delivered |= bool(reward)
            through |= pass_gate
            rows.append(dict(u=command.tolist(), rgb=marks.tolist(), tool=tool.tolist(), puck=puck.tolist(),
                             distance=distance, gate=pass_gate, reward=reward))
    return rows


def summarize(rows):
    gates, docks = [], []
    for start in range(0, len(rows), DELIVERY_BLOCKS):
        block = rows[start:start + DELIVERY_BLOCKS]
        gates.append(next(((i + 1) * BLOCK_FRAMES for i, row in enumerate(block) if row['gate']), None))
        docks.append(next(((i + 1) * BLOCK_FRAMES for i, row in enumerate(block) if row['reward']), None))
    view_end = (PIXELS // 2 - 2) / (VIEW_SCALE * MODEL_SCALE)
    edge = sum(bool(np.max(np.abs(row['rgb'])) >= view_end) for row in rows)
    return dict(gates=gates, docks=docks, edge_blocks=edge,
                requested_rate_frames=PULSE_FRAMES * sum(sum(map(abs, row['u'])) for row in rows))


def verified(arena, seed, world, change, prior, learner_seed, mode, deliveries):
    player = Player(prior, learner_seed, mode)
    rows = play(arena, seed, world, change, player=player, deliveries=deliveries)
    if play(arena, seed, world, change, tape=[r['u'] for r in rows], deliveries=deliveries) != rows:
        raise ValueError('physical, RGB or score replay diverged')
    repeated = Player(prior, learner_seed, mode)
    reward = 0
    for row in rows:
        if repeated.act(row['rgb'], reward).tolist() != row['u']:
            raise ValueError('public-observation policy replay diverged')
        reward = row['reward']
    if repeated.alarms != player.alarms:
        raise ValueError('change decisions differ on replay')
    return dict(summary=summarize(rows), alarms=player.alarms, blocks=rows)


def run(setup_path, checkpoint, output, count, master_seed, *, signed_path=None, pilot=False):
    scene = _read(setup_path)
    archive = _read(checkpoint)
    optical = archive.get('tt17_motor', {})
    setup_digest = _hash(setup_path)
    if optical and optical['setup_sha256'] != setup_digest:
        raise ValueError('the corrected checkpoint uses a different physical setup')
    body, correction = load_map(checkpoint)
    home = np.zeros(body.action_size)
    corrected = body.jacobian(home)[:3] @ correction
    library = load(checkpoint)[0]
    source = library.models[archive['gain_memory']['maps'][0]['model']]
    original = source.jacobian(home)[:3]
    key = optical.get('optical_fit', {}).get('signed')
    signed = original if key is None else rewire(source, *key).jacobian(home)[:3]
    if signed_path is not None:
        signed_optical = _read(signed_path).get('tt17_motor', {})
        if signed_optical and signed_optical['setup_sha256'] != setup_digest:
            raise ValueError('the signed checkpoint uses a different physical setup')
        reference, signed_transform = load_map(signed_path)
        if reference.action_size != body.action_size or not np.array_equal(signed_transform, np.eye(body.action_size)):
            raise ValueError('expected a signed body without a continuous correction')
        signed = reference.jacobian(np.zeros(reference.action_size))[:3]
    priors = dict(corrected=corrected, signed=signed, original=original, cold=np.zeros_like(signed))
    wiring_seed = scene['wiring'] if 'wiring' in scene else scene['arena']['seed']
    arena = TargetArena(**dict(scene['arena'], render_mode='projected'))
    seeds = np.random.SeedSequence([master_seed, wiring_seed, 3717]).spawn(count)
    episodes = []
    try:
        for child in seeds:
            world_seed, learner_seed = map(int, child.generate_state(2))
            world = scenario(world_seed, np.asarray(scene['matrix']))
            follow = {change: {mode: verified(arena, wiring_seed, world, change, priors['corrected'],
                                             learner_seed, mode, BEFORE + AFTER) for mode in MODES}
                      for change in CHANGES}
            prime = {'corrected': dict(summary=summarize(follow['sham']['continuous']['blocks'][:SWITCH]))}
            for name in BASELINES:
                prime[name] = verified(arena, wiring_seed, world, 'sham', priors[name], learner_seed,
                                       'continuous', BEFORE)
            if not all(follow['sham']['continuous']['blocks'][:PROBES] == prime[name]['blocks'][:PROBES]
                       for name in BASELINES):
                raise ValueError('matched exploratory pulses differ')
            for mode in MODES:
                if any(follow[change][mode]['blocks'][:SWITCH] != follow['sham'][mode]['blocks'][:SWITCH]
                       for change in ('motor', 'payload')):
                    raise ValueError('hidden treatment changed the common physical prefix')
            episodes.append(dict(world_seed=world_seed, learner_seed=learner_seed, world=world, prime=prime,
                                 follow=follow))
            print(output.stem, world_seed, {name: item['summary']['docks'] for name, item in prime.items()},
                  {change: {k: r['summary']['docks'][BEFORE:] for k, r in modes.items()}
                   for change, modes in follow.items()}, flush=True)
    finally:
        arena.close()
    _write(output, dict(schema='tt17-payload-v1', setup=str(setup_path), setup_sha256=setup_digest,
                        checkpoint=str(checkpoint), checkpoint_sha256=_hash(checkpoint),
                        signed_checkpoint=None if signed_path is None else str(signed_path),
                        signed_sha256=None if signed_path is None else _hash(signed_path),
                        optical=optical,
                        implementation_sha256=_hash(Path(__file__)), wiring=wiring_seed, master_seed=master_seed,
                        control_dt=arena.control_dt, priors={k: v.tolist() for k, v in priors.items()},
                        budget=dict(delivery_blocks=DELIVERY_BLOCKS, block_frames=BLOCK_FRAMES, switch=SWITCH,
                                    initially_matched_probes=PROBES), pilot=pilot, replayed=True, episodes=episodes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('setup', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--signed', type=Path)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--seed', type=int, default=HOLDOUT_SEED)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    run(args.setup, args.checkpoint, args.output, args.count, args.seed, signed_path=args.signed, pilot=args.pilot)

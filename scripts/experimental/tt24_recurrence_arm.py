"""Use a native-earned local mover witness during matched short arm calibration."""

import argparse
from collections import deque
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_trial import play, verify_replay
from .tt24_motor import MotorEffects, MotorPolicy, optical_sightings
from .tt24_recurrence import MODES, OUTPUT, WEIGHTS, contrast, effect_features, slope


RECEIPT = OUTPUT / 'recurrence-arm.json.gz'
REPLICATION = OUTPUT / 'recurrence-arm-replication.json.gz'
EMBODIMENT = OUTPUT / 'motor-rate-target-only.json.gz'
SCHEDULE = 1678521930
REPLICATION_SEED = 88323519
PRACTICE = 10
HELD = 12
EXPLORATION = 1.4
PULSE_FRAMES = 9
VISION_LIMIT = 35
PRACTICE_LIMIT = 96
CREW = (*MODES, 'warm')


def target_effects(body, axes):
    state = MotorEffects(axes)
    state.readout = np.asarray(body['motor_effects'])
    state.precision = np.asarray(body['motor_precision'])
    state.samples = int(body['motor_samples'])
    return state


class FixedWitness:
    """Register a single local perturbation using only the fixed camera."""

    def __init__(self, dt):
        self.dt = dt
        self.recent = deque(maxlen=5)
        self.cue = None
        self.samples = []
        self.prefix = sha256()
        self.post_frames = 0

    def update(self, tick, rgb):
        ball, blade = optical_sightings(rgb)
        if blade is None or ball is None:
            row = (np.full(2, np.nan), np.full(2, np.nan), np.nan)
        else:
            row = (blade.center, ball.center, blade.half_span)
        if self.cue is not None:
            if self.cue['lo'] <= tick < self.cue['hi']:
                self.samples.append(row)
            return
        self.prefix.update(rgb)
        self.recent.append(row)
        if len(self.recent) != self.recent.maxlen:
            return
        actor, mover, spans = (np.asarray(values) for values in zip(*self.recent, strict=True))
        if not np.isfinite(np.r_[actor.ravel(), mover.ravel(), spans]).all():
            return
        vel = slope(mover, np.arange(len(mover), dtype=float))
        gap = mover[-1] - actor[-1]
        expected = -(gap @ vel) / max(1.e-8, vel @ vel)
        scale = np.median(spans)
        if not (5 <= expected <= 23 and np.linalg.norm(gap) < 5. * scale):
            return
        predicted = tick + int(np.ceil(expected))
        lo = max(tick + 1, predicted - 2)
        hi = predicted + 7
        self.cue = dict(tick=tick, lo=lo, hi=hi, eta=float(expected), scale=float(scale),
                        actor=actor.tolist(), mover=mover.tolist(), prefix_sha256=self.prefix.hexdigest())

    def extract(self):
        if self.cue is None or len(self.samples) < self.cue['hi'] - self.cue['lo']:
            return None
        actor, mover, _ = (np.asarray(values) for values in zip(*self.samples, strict=True))
        result = effect_features(self.cue['actor'], self.cue['mover'], actor, mover,
                                 scale=self.cue['scale'], dt=self.dt)
        return None if result is None else result.tolist()


def pulse(arena, launch, body, basis, perturbation, sign, regime):
    options = dict(readout=body['readout'], retinal=body['retinal'])
    controller = MotorPolicy(basis, target_effects(body, arena.action_size), **options)
    tracker = FixedWitness(arena.dt)
    obs = arena.reset(seed=launch, split='practice', regime=regime)
    done, reward = False, 0.
    commands, pulse_frames = [], 0
    while not done and len(commands) < PRACTICE_LIMIT:
        tick = len(commands)
        will_observe = tick < VISION_LIMIT if tracker.cue is None else tick < tracker.cue['hi']
        if will_observe:
            tracker.update(tick, arena.render_external())
        enabled = tracker.cue is not None and tick - tracker.cue['tick'] < PULSE_FRAMES
        if enabled:
            pulse_frames += 1
        controller.weights = basis + sign * perturbation if enabled else basis
        sampled = tick % body['stride'] == 0
        if sampled:
            obs = obs | {'rgb': obs['rgb'][::2, ::2]}
        command = controller.command(obs, sampled=sampled)
        commands.append(command.tolist())
        obs, r, done = arena.step(command, render=(tick + 1) % body['stride'] == 0)
        reward += r
    local = tracker.extract()
    runtime = len(commands) * arena.dt
    return dict(seed=launch, sign=sign, pulse_frames=pulse_frames, cue=tracker.cue, optical=local,
                public_score=float(reward), seconds=runtime, censored=not done, actions=commands,
                referee=arena.judge_metrics())


def source_effects(pair, source):
    a, b = [branch['optical'] for branch in pair]
    if a is None or b is None:
        support = 0.
    else:
        stretch = np.max(np.abs((np.asarray(a) - b) / source['scale']))
        support = float(np.exp(-.5 * max(stretch - 3.5, 0.)**2))
    return support, {mode: support * contrast(a, b, source, mode) for mode in MODES}


def verify_pair(plus, minus):
    starts = plus['cue'], minus['cue']
    if starts[0] != starts[1]:
        raise ValueError('paired arm pulses had different fixed-camera prefixes')
    stop = len(plus['actions']) if starts[0] is None else starts[0]['tick']
    if plus['actions'][:stop] != minus['actions'][:stop]:
        raise ValueError('paired arm pulses had different pre-intervention commands')


def fit(practice, basis):
    directions = np.array([row['direction'] for row in practice]).reshape(len(practice), -1)
    gram = directions @ directions.T + .48 * np.eye(len(directions))
    gradients = {}
    for mode in MODES:
        delta = np.array([row['public_delta'] + .65 * row['source_delta'][mode] for row in practice])
        gradients[mode] = (directions.T @ np.linalg.solve(gram, delta)).reshape(basis.shape)
    scale = max(.12, np.linalg.norm(gradients['cold']))
    return {mode: basis + .85 * np.tanh(value / scale) for mode, value in gradients.items()} | {'warm': basis}


def summarized(rows):
    modes = {}
    for mode in CREW:
        rebounds = [row['rebounds'] for row in rows[mode]]
        modes[mode] = dict(rebounds=rebounds, mean=float(np.mean(rebounds)), best=max(rebounds),
                           seconds=float(np.mean([row['elapsed_s'] for row in rows[mode]])),
                           five=sum(value >= 5 for value in rebounds))
    rng = np.random.default_rng(991025411)
    pick = rng.integers(len(rows['pong']), size=(4096, len(rows['pong'])))
    pair = {}
    for mode in ('cold', 'shuffled', 'warm'):
        delta = np.asarray(modes['pong']['rebounds']) - modes[mode]['rebounds']
        pair[mode] = dict(mean=float(delta.mean()), higher=int(sum(delta > 0)), lower=int(sum(delta < 0)),
                          interval=np.quantile(delta[pick].mean(axis=1), (.025, .975)).tolist())
    return dict(modes=modes, pong_minus=pair)


def evaluate(held, policies, body, report, path, began, *, arm='stock', regime='standard'):
    arena = BounceArena(view='wrist', width=64, height=48, arm=arm)
    try:
        for i, launch in enumerate(held):
            for mode in CREW if i % 2 == 0 else CREW[::-1]:
                if len(report['held'][mode]) > i:
                    continue
                policy = np.asarray(policies[mode])
                row = play(arena, launch, policy, target_effects(body, arena.action_size), split='held',
                           stride=body['stride'], readout=body['readout'], retinal=body['retinal'],
                           keep_actions=True, regime=regime)
                report['held'][mode].append(row)
                _write(path, report)
                print('held', launch, mode, row['rebounds'], round(perf_counter() - began, 1), flush=True)
        for rows in report['held'].values():
            for row in rows:
                verify_replay(arena, row)
    finally:
        arena.close()
    report['summary'] = summarized(report['held'])
    report['replayed'] = sum(len(rows) for rows in report['held'].values())
    report['seconds'] = perf_counter() - began
    _write(path, report)
    print(report['summary'], flush=True)


def fresh_body(template):
    basis = np.zeros_like(template['weights'], dtype=float)
    effects = MotorEffects(len(basis))
    return template | dict(weights=basis.tolist(), motor_effects=effects.readout.tolist(),
                           motor_precision=effects.precision.tolist(), motor_samples=0, total_initial_attempts=0)


def run(source_path, embodiment_path, path, *, arm='stock', regime='standard', schedule=SCHEDULE, untrained=False):
    began = perf_counter()
    source = _read(source_path)
    if not source['qualification']['passed']:
        raise ValueError('local source failed held native qualification')
    body = _read(embodiment_path)
    if body['mode'] != 'cold' or body['width'] != 64 or body['height'] != 48:
        raise ValueError('expected the shared 64x48 cold direct-policy warm start')
    if untrained:
        body = fresh_body(body)
    rng = np.random.default_rng(schedule)
    seeds = rng.choice(2147483646, PRACTICE + HELD, replace=False).tolist()
    train, held = seeds[:PRACTICE], seeds[PRACTICE:]
    known = set(body['practice_seeds']) | {row['seed'] for rows in body['held'].values() for row in rows}
    if set(seeds) & known:
        raise ValueError('short calibration overlaps shared embodiment experience')
    metadata = dict(schema='tt24-local-recurrence-arm-1', source_sha256=_hash(source_path),
                    embodiment_sha256=_hash(embodiment_path), embodiment=embodiment_path.name,
                    prior_target_attempts=body['total_initial_attempts'],
                    previous_motor_observations=body['motor_samples'], practice=train, held=held, schedule=schedule,
                    fixed_calibration_camera=[128, 96], inherited_wrist_camera=[64, 48], pulse_frames=PULSE_FRAMES,
                    calibration_frame_cap=PRACTICE_LIMIT)
    if arm != 'stock' or regime != 'standard' or untrained:
        metadata.update(arm=arm, regime=regime, untrained=untrained)
    report = _read(path) if path.exists() else dict(metadata=metadata, practice=[], held={mode: [] for mode in CREW})
    if report['metadata'] != metadata:
        raise ValueError('paired calibration receipt has a different schedule or warm-start policy')
    basis = np.asarray(body['weights'])
    directions = rng.normal(size=(PRACTICE, len(basis), 1)) * rng.normal(size=(PRACTICE, 1, basis.shape[1]))
    directions *= EXPLORATION / np.linalg.norm(directions, axis=(1, 2))[:, None, None]
    calibration = BounceArena(view='wrist', width=128, height=96, arm=arm)
    try:
        for i, (launch, direction) in enumerate(zip(train, directions, strict=True)):
            if i < len(report['practice']):
                continue
            signs = (1, -1) if i % 2 == 0 else (-1, 1)
            rows = {sign: pulse(calibration, launch, body, basis, direction, sign, regime) for sign in signs}
            plus, minus = (rows[sign] for sign in (1, -1))
            verify_pair(plus, minus)
            support, source_delta = source_effects((plus, minus), source)
            scores = [np.log1p(row['seconds']) + .15 * row['public_score'] for row in (plus, minus)]
            record = dict(direction=direction.tolist(), plus=plus, minus=minus, public_delta=float(scores[0] - scores[1]),
                          source_delta=source_delta, support=support)
            report['practice'].append(record)
            _write(path, report)
            print('pair', launch, plus['pulse_frames'], minus['pulse_frames'], round(support, 3),
                  {k: round(v, 3) for k, v in source_delta.items()}, round(scores[0] - scores[1], 3), flush=True)
    finally:
        calibration.close()
    policies = fit(report['practice'], basis)
    report['policies'] = {mode: policy.tolist() for mode, policy in policies.items()}
    report['optical_pairs'] = sum(row['support'] > 0. for row in report['practice'])
    _write(path, report)
    if report['optical_pairs'] < 3:
        print('insufficient paired fixed-camera windows', report['optical_pairs'], flush=True)
        return
    evaluate(held, policies, body, report, path, began, arm=arm, regime=regime)


def replicate(first_path, embodiment_path, path, count):
    began = perf_counter()
    first = _read(first_path)
    body = _read(embodiment_path)
    if first['metadata']['embodiment_sha256'] != _hash(embodiment_path) or first['replayed'] != 4 * HELD:
        raise ValueError('replication needs the original completed pulse calibration')
    held = np.random.default_rng(REPLICATION_SEED).choice(2147483646, count, replace=False).tolist()
    known = set(body['practice_seeds']) | {row['seed'] for rows in body['held'].values() for row in rows}
    known.update(first['metadata']['practice'] + first['metadata']['held'])
    if known.intersection(held):
        raise ValueError('replication starts overlap arm practice or earlier tests')
    metadata = dict(schema='tt24-local-recurrence-replication-1', calibration_sha256=_hash(first_path),
                    held=held, schedule=REPLICATION_SEED, previous_attempts=body['total_initial_attempts'])
    report = _read(path) if path.exists() else dict(metadata=metadata, held={mode: [] for mode in CREW})
    if report['metadata'] != metadata:
        raise ValueError('replication uses different weights, body, or launch order')
    evaluate(held, first['policies'], body, report, path, began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=WEIGHTS)
    parser.add_argument('--embodiment', type=Path, default=EMBODIMENT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replicate', type=Path, help='completed calibration to test on fresh starts')
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--arm', choices=('stock', 'alternate'), default='stock')
    parser.add_argument('--regime', choices=('standard', 'changed'), default='standard')
    parser.add_argument('--schedule', type=int, default=SCHEDULE)
    parser.add_argument('--untrained', action='store_true')
    args = parser.parse_args()
    if args.replicate:
        replicate(args.replicate, args.embodiment, args.output or REPLICATION, args.count)
    else:
        run(args.source, args.embodiment, args.output or RECEIPT, schedule=args.schedule,
            arm=args.arm, regime=args.regime, untrained=args.untrained)


if __name__ == '__main__':
    main()

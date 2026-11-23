"""Recall reward-trained motor residuals using public appearance and motion."""

import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena, mujoco
from .tt24_donor_adapt import CHECKPOINT, PLAN
from .tt24_donor_trial import DESIGNS, DonorBounce, Native, ROOT, _live
from .tt24_public_body import FUSED_OUTPUT, load_public


DATA = ROOT / 'context-residual-public.npz'
OUTPUT = ROOT / 'context-residual.json.gz'
SWITCHED = ROOT / 'context-residual-switch.json.gz'
SOURCE = ROOT / 'tt24_donor-refined.json.gz'
PREFIX = 40
SCHEDULE_SEED = 265830519
SAMPLES = 3
WINDOW = 10


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def records():
    plans = json.loads(PLAN.read_text())['groups']
    donor = read(SOURCE)
    archives = [read(ROOT / f'donor-adapt-{i}.json.gz') for i in range(len(plans))]
    data = []
    for index, cases in enumerate(donor['held_donor']):
        for element in cases[:4]:
            row = element['conditions']['clean']
            metrics = row['metrics']
            data.append(dict(group=index + 4, seed=metrics['seed'], actions=row['actions'], score=metrics['rebounds'],
                             run=metrics['longest_consecutive_rebounds']))
    for i, archive in enumerate(archives):
        for row in archive['outcomes']['learned']['practice']:
            data.append(dict(row, group=i))
    return plans, archives, data


def create(plans, group):
    options = dict(view='wrist', width=192, height=144, max_steps=360 if group < 4 else 300)
    if group >= 4:
        return DonorBounce(DESIGNS[group - 4], **options), 'changed'
    condition = plans[group]
    arena = BounceArena(**options)
    arena._model.opt.gravity[:] *= condition['gravity']
    arena._model.body_mass[arena._ball_body] *= condition['mass']
    arena._model.body_inertia[arena._ball_body] *= condition['mass']
    if condition['weak_drive']:
        arena._standard_gain *= .80
        arena._standard_bias *= .80
        arena._standard_force *= .80
    mujoco.mj_setConst(arena._model, arena._data)
    return arena, 'standard'


def appearance(obs):
    cells = [cell for half in np.array_split(obs['rgb'], 2, axis=0) for cell in np.array_split(half, 2, axis=1)]
    colors = np.concatenate([cell.mean(axis=(0, 1)) for cell in cells]) / 255.
    return np.r_[obs['joints'], colors]


def motor(obs):
    return np.r_[obs['joints'], obs['joint_velocity']].astype(float)


def sample(arena, regime, tape):
    obs = arena.reset(seed=tape['seed'], split='held', regime=regime)
    descriptor = appearance(obs)
    observer = load_public(FUSED_OUTPUT)
    trace, body, actions, usable = [], [], [], []
    for i, command in enumerate(tape['actions']):
        if i < PREFIX:
            previous = observer.gaps
            trace.append(observer.observe(obs))
            usable.append(previous == observer.gaps)
            body.append(motor(obs))
            actions.append(command)
        obs, _, done = arena.step(command, render=i + 1 < PREFIX)
        if done:
            break
    scores = arena.judge_metrics()
    if i + 1 != len(tape['actions']) or scores['rebounds'] != tape['score'] or scores['longest_consecutive_rebounds'] != tape['run']:
        raise ValueError(f'archived calibration replay changed the physical result on {tape["seed"]}')
    return dict(scene=descriptor, optic=np.asarray(trace), body=np.asarray(body), commands=np.asarray(actions),
                visible=np.asarray(usable), group=np.array(tape['group']), seed=np.array(tape['seed']))


def collect(path):
    plans, _, tapes = records()
    payload = {}
    for group in range(7):
        arena, condition = create(plans, group)
        try:
            for tape in tapes:
                if tape['group'] != group:
                    continue
                data = sample(arena, condition, tape)
                payload.update({f'{group}_{tape["seed"]}_{k}': value for k, value in data.items()})
                print('context source', group, tape['seed'], len(data['optic']), int(data['visible'].sum()), flush=True)
        finally:
            arena.close()
    np.savez_compressed(path, **payload)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        names = [key[:-6] for key in archive.files if key.endswith('_scene')]
        return [{key: archive[f'{name}_{key}'] for key in ('scene', 'optic', 'body', 'commands', 'visible', 'group', 'seed')}
                for name in names]


def flight(positions, seen):
    """Infer acceleration from a recent, visually supported arc segment."""
    p = np.asarray(positions)
    if len(p) != WINDOW:
        return None
    valid = np.asarray(seen, dtype=bool)
    if valid.sum() < WINDOW - 1:
        return None
    z = p[valid, 2]
    if min(z) < .052 or max(z) > .43:
        return None
    t = (np.arange(WINDOW) - (WINDOW - 1) / 2.)[valid] / 60.
    design = np.c_[np.ones(len(t)), t, .5 * t**2]
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    noise = np.mean((z - design @ coef)**2)**.5
    if noise > .007 or coef[2] > -2.0 or coef[2] < -25.:
        return None
    return np.array((coef[2], coef[1], noise))


def innovation(previous, earlier, origin, actions):
    return np.r_[1., np.concatenate(actions), previous[5:] / 4., earlier[5:] / 4.,
                 (previous[:5] - origin[:5]) * 5.]


def regress(x, y, strength, prior=None):
    regular = np.eye(x.shape[1]) * strength
    if prior is None:
        prior = np.zeros((x.shape[1], y.shape[1]))
        regular[0, 0] *= .05
    return np.linalg.solve(x.T @ x + regular, x.T @ y + regular @ prior)


class AtRest:
    """Distinguish stocked and older wrist setups from any first image and own encoders."""

    def __init__(self, items):
        fields = np.asarray([item['scene'] for item in items])
        labels = np.asarray([int(item['group'] >= 4) for item in items])
        self.centers = np.asarray([fields[labels == k].mean(axis=0) for k in range(2)])
        self.scale = np.maximum(np.sqrt(np.mean((fields - self.centers[labels])**2, axis=0)), .020)

    def recognized_donor(self, state):
        delta = (state - self.centers) / self.scale
        return np.mean(delta[1]**2) < np.mean(delta[0]**2)


class CalibratedLikelihood:
    """Keep motion-prediction residuals calibrated on earlier physical observations."""

    def __init__(self, items):
        self.rest = AtRest(items)
        self.gravity, self.encoder = [], []
        for group in range(4):
            snapshots, x, y = [], [], []
            for item in items:
                if item['group'] != group:
                    continue
                pos, seen, body, actions = (item[key] for key in ('optic', 'visible', 'body', 'commands'))
                for t in range(WINDOW, min(23, len(pos)), 2):
                    sample = flight(pos[t + 1 - WINDOW:t + 1, :3], seen[t + 1 - WINDOW:t + 1])
                    if sample is not None:
                        snapshots.append(sample)
                for t in range(3, len(actions)):
                    x.append(innovation(body[t-1], body[t-2], body[0], actions[t-3:t]))
                    y.append(body[t, 5:] / 4.)
            self.gravity.append(np.asarray(snapshots))
            self.encoder.append((np.asarray(x), np.asarray(y)))
        pooled = np.vstack(self.gravity)
        self.mean_gravity = np.array([np.median(q[:, 0]) if len(q) else np.median(pooled[:, 0])
                                      for q in self.gravity])
        self.gravity_scale = np.asarray([max(.8, 1.48 * np.median(abs(q[:, 0] - self.mean_gravity[i])))
                                        if len(q) > 4 else 2.5 for i, q in enumerate(self.gravity)])
        x, y = map(np.vstack, zip(*self.encoder, strict=True))
        shared = regress(x, y, .12)
        self.weights = np.asarray([regress(a, b, 1.4, shared) for a, b in self.encoder])
        self.encoder_scale = np.maximum(np.sqrt(np.mean((x @ shared - y)**2, axis=0)), .055)

    def flight_loss(self, measured):
        expected = max(.85, 90. * measured[2]) + self.gravity_scale
        return 1.7 * np.log1p(((self.mean_gravity - measured[0]) / expected)**2) + .08 * np.log(expected)

    def motor_loss(self, feature, velocity):
        predicted = feature @ self.weights
        deviation = (predicted - velocity[None, :]) / self.encoder_scale
        return np.mean(np.log1p(deviation**2), axis=1)


class Router:
    """Maintain a causal, fading belief over the previously acquired residuals."""

    def __init__(self, fitted, start):
        self.fitted = fitted
        self.donor = fitted.rest.recognized_donor(start)
        self.positions = deque(maxlen=WINDOW)
        self.visible = deque(maxlen=WINDOW)
        self.actions = deque(maxlen=3)
        self.body = deque(maxlen=2)
        self.origin = None
        self.initial = 0
        self.gravity = np.zeros(4)
        self.motor = np.zeros(4)
        self.count = 0
        self.evidence = []

    def update(self, state, valid, body):
        if self.origin is None:
            self.origin = body.copy()
        if not self.donor and len(self.actions) == 3 and len(self.body) == 2:
            x = innovation(self.body[-1], self.body[-2], self.origin, self.actions)
            prediction = self.fitted.motor_loss(x, body[5:] / 4.)
            self.motor = .90 * self.motor + .10 * (prediction - prediction.mean())
        self.positions.append(state[:3].copy())
        self.visible.append(valid)
        self.body.append(body)
        self.initial += 1
        if not self.donor and self.initial % 2 == 0:
            measured = flight(self.positions, self.visible)
            if measured is not None:
                loss = self.fitted.flight_loss(measured)
                self.gravity = .56 * self.gravity + .44 * (loss - loss.mean())
                self.count += 1
                self.evidence.append([self.initial, *measured.tolist()])

    def posterior(self):
        logits = -3.6 * (self.gravity + .42 * self.motor)
        weights = np.exp(logits - max(logits))
        return weights / weights.sum()


def calibration_audit(items):
    rows = []
    for i, item in enumerate(items):
        mix = CalibratedLikelihood([value for j, value in enumerate(items) if j != i])
        group = int(item['group'])
        router = Router(mix, item['scene'])
        snapshots = []
        for k in range(len(item['optic'])):
            router.update(item['optic'][k], item['visible'][k], item['body'][k])
            router.actions.append(item['commands'][k])
            if k in (9, 13, 17, 21, 27, 35):
                snapshots.append(dict(frames=k + 1, posterior=router.posterior().tolist(), observations=router.count,
                                      optical_only=int(np.argmin(router.gravity)),
                                      encoder_only=int(np.argmin(router.motor))))
        rows.append(dict(group=group, seed=int(item['seed']), expected_donor=group >= 4,
                         inferred_donor=bool(router.donor), probabilities=snapshots, visual_curvature=router.evidence))
    return rows


def confidence(audit):
    evidence = [(max(item['posterior']), int(np.argmax(item['posterior'])) == row['group'])
                for row in audit if row['group'] < 4 for item in row['probabilities']
                if item['frames'] <= 18 and item['observations']]
    for cut in np.linspace(.50, .92, 22):
        outcomes = [right for certainty, right in evidence if certainty >= cut]
        if len(outcomes) >= 8 and np.mean(outcomes) >= .85:
            return float(cut)
    return .94


def run(arena, regime, native, weights, residuals, group, start, fitted, mode, *, switch=None, certainty=.7):
    obs = arena.reset(seed=start, split='held', regime=regime)
    router = Router(fitted, appearance(obs))
    eyes = load_public(FUSED_OUTPUT)
    if mode == 'oracle':
        active = residuals[group].copy() if group < 4 else np.zeros_like(weights)
    elif mode == 'last':
        active = residuals[-1].copy()
    elif mode == 'global' or mode == 'appearance' and not router.donor:
        active = residuals.mean(axis=0)
    else:
        active = np.zeros_like(weights)
    issued, action = np.zeros(5), np.zeros(5)
    actions, events = [], []
    done = False
    retained = None
    original = arena._model.opt.gravity.copy()
    while not done:
        if switch is not None and len(actions) == switch[0]:
            arena._model.opt.gravity[:] = original * switch[2]
            if mode == 'oracle':
                active = residuals[switch[1]].copy()
        gaps = eyes.gaps
        state = np.ascontiguousarray(eyes.observe(obs), dtype=float)
        router.update(state, eyes.gaps == gaps, motor(obs))
        prior = router.posterior()
        if mode == 'motion' and not router.donor:
            active = .20 * active + .80 * np.tensordot(prior, residuals, axes=1)
        if mode == 'commit' and not router.donor and retained is None and router.count and max(prior) >= certainty:
            selected = int(np.argmax(prior))
            retained = [len(actions), selected, float(prior[selected])]
            active = residuals[selected].copy()
        candidate = np.ascontiguousarray(weights + active, dtype=float)
        native.command(state.ctypes.data, candidate.ctypes.data, issued.ctypes.data, action.ctypes.data)
        router.actions.append(action.copy())
        actions.append(action.tolist())
        if len(actions) in (1, 10, 14, 18, 24, 32, 48) or len(actions) % 60 == 0:
            events.append([len(actions), router.count, *prior.tolist()])
        obs, _, done = arena.step(action)
    scores = arena.judge_metrics()
    arena._model.opt.gravity[:] = original
    arena.reset(seed=start, split='held', regime=regime, render=False)
    for tick, command in enumerate(actions):
        if switch is not None and tick == switch[0]:
            arena._model.opt.gravity[:] = original * switch[2]
        _, _, done = arena.step(command, render=False)
        if bool(done) != (tick + 1 == len(actions)):
            raise ValueError(f'replayed context policy ended at a different tick on {start}')
    if digest_json(scores) != digest_json(arena.judge_metrics()):
        raise ValueError(f'replayed context policy changed physical events on {start}')
    arena._model.opt.gravity[:] = original
    return dict(group=group, seed=start, score=scores['rebounds'], run=scores['longest_consecutive_rebounds'],
                ticks=scores['steps'], donor_inferred=bool(router.donor), posterior=events,
                ball_curvature=router.evidence, retained=retained, actions=actions)


def summary(rows):
    def portion(index):
        vals = np.array([row['run'] for row in rows if index(row['group'])])
        return dict(drops=len(vals), total=int(vals.sum()), mean=float(vals.mean()), maximum=int(vals.max()),
                    ten=int((vals >= 10).sum()))
    return dict(target=portion(lambda group: group < 4), donor=portion(lambda group: group >= 4),
                group=[portion(lambda other: other == group) for group in range(7)])


def roster(seed):
    random = np.random.default_rng(seed)
    groups = np.array([4, 0, 2, 5, 1, 3, 6] * SAMPLES)
    for index in range(SAMPLES):
        i = index * 7
        if index:
            groups[i:i + 7] = random.permutation(groups[i:i + 7])
    return [(int(group), int(start)) for group, start in zip(groups, random.integers(1, 2**31 - 1, len(groups)), strict=True)]


def qualify(path, output):
    plans, archives, _ = records()
    with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
        initial = np.asarray(checkpoint['weights'])
    residual = np.asarray([item['outcomes']['learned']['weights'] for item in archives]) - initial
    items = load(path)
    model = CalibratedLikelihood(items)
    validation = calibration_audit(items)
    schedule = roster(SCHEDULE_SEED)
    if {int(item['seed']) for item in items} & {seed for _, seed in schedule}:
        raise ValueError('test seeds overlap archived calibration')
    result = dict(schema='tt24-context-residual-1', schedule=schedule, validation=validation,
                  source_target_attempts=32, donor_sources=12, stage='mixed hidden between-episode revisits',
                  historical_donor_observer='clean during source evaluation, common wrist during these comparisons',
                  mean_gravity=model.mean_gravity.tolist(), gravity_scale=model.gravity_scale.tolist(),
                  held={}, summary={})
    print('visual accelerations', model.mean_gravity.tolist(), 'scales', model.gravity_scale.tolist(),
          'segments', [len(row) for row in model.gravity], flush=True)

    def write():
        with gzip.open(output, 'wt') as stream:
            json.dump(result, stream, separators=(',', ':'))

    native = Native()
    arenas = [create(plans, group) for group in range(7)]
    try:
        for mode in ('motion', 'frozen', 'oracle', 'global', 'last', 'appearance'):
            rows = []
            for group, seed in schedule:
                arena, regime = arenas[group]
                row = run(arena, regime, native, initial, residual, group, seed, model, mode)
                rows.append(row)
                print('context held', mode, group, seed, row['run'], row['donor_inferred'], flush=True)
            result['held'][mode] = rows
            result['summary'][mode] = summary(rows)
            print(mode, result['summary'][mode], flush=True)
            write()
        result['physical_replays'] = sum(len(rows) for rows in result['held'].values())
        write()
    finally:
        for arena, _ in arenas:
            arena.close()
        native.working.cleanup()


def switch_check(path, output):
    plans, archives, _ = records()
    with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
        initial = np.asarray(checkpoint['weights'])
    residual = np.asarray([item['outcomes']['learned']['weights'] for item in archives]) - initial
    last_active = residual[np.flatnonzero(np.any(residual != 0., axis=(1, 2)))[-1]]
    items = load(path)
    model = CalibratedLikelihood(items)
    audit = calibration_audit(items)
    limit = confidence(audit)
    starts = np.random.default_rng(265830521).integers(1, 2**31 - 1, 4).tolist()
    native = Native()
    contexts = {group: create(plans, group) for group in range(7)}
    result = dict(schema='tt24-context-residual-checks-1', switch={}, original_geometry_input_diagnostic=[],
                  commit_threshold_from_source=limit, calibration_ablation=audit, same_roster_commit=[])
    try:
        for mode in ('motion', 'frozen', 'oracle'):
            rows = []
            for group, seed in zip((0, 2, 0, 2), starts, strict=True):
                alternate = 2 if group == 0 else 0
                ratio = plans[alternate]['gravity'] / plans[group]['gravity']
                arena, regime = contexts[group]
                row = run(arena, regime, native, initial, residual, group, seed, model, mode, switch=(24, alternate, ratio))
                rows.append(row | dict(change_tick=24, new_group=alternate, gravity_factor=ratio))
                print('context in-flight', mode, group, alternate, row['run'], flush=True)
            result['switch'][mode] = rows

        # This separate diagnostic preserves exactly the feedback available in
        # historical donor training. Only the residual gate is public here.
        for group, seed in roster(SCHEDULE_SEED):
            if group < 4:
                continue
            arena, regime = contexts[group]
            obs = arena.reset(seed=seed, split='held', regime=regime)
            recognized = model.rest.recognized_donor(appearance(obs))
            row = dict(seed=seed, group=group, public_gate_inferred_donor=bool(recognized))
            for mode, current in (('frozen', initial), ('public_gate', initial if recognized else initial + residual.mean(axis=0)),
                                  ('last_nonzero', initial + last_active), ('global', initial + residual.mean(axis=0))):
                game = _live(native, arena, current, seed, regime, 'clean')
                replay = _live(native, arena, current, seed, regime, 'replay', tape=game['actions'])
                if digest_json(game) != digest_json(replay):
                    raise ValueError('source-faithful donor diagnostic changed physical replay')
                row[mode] = game['metrics']['longest_consecutive_rebounds']
            result['original_geometry_input_diagnostic'].append(row)
        for group, seed in roster(SCHEDULE_SEED):
            if group >= 4:
                continue
            arena, regime = contexts[group]
            row = run(arena, regime, native, initial, residual, group, seed, model, 'commit', certainty=limit)
            result['same_roster_commit'].append(row)
            print('context hard selection', group, seed, row['run'], row['retained'], flush=True)
        with gzip.open(output, 'wt') as stream:
            json.dump(result, stream, separators=(',', ':'))
    finally:
        for arena, _ in contexts.values():
            arena.close()
        native.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'qualify', 'switch'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'collect':
        collect(args.data)
    elif args.stage == 'switch':
        switch_check(args.data, args.output or SWITCHED)
    else:
        qualify(args.data, args.output or OUTPUT)


if __name__ == '__main__':
    main()

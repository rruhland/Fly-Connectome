"""Acquire and qualify a visual-duration skill using native Pong RGB and elapsed time."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
import torch

from .tt3_pong import ACTIONS, CAMERA_SECONDS, CAMERA_TICKS, NATIVE_HZ, _VisualActionEffects
from .tt3_pong import _native_rallies, cases_from_seeds, summarize_pong
from .tt3_visual import Binding, Track
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_source import CHECKPOINT, FEATURES, MODES, VisualConsequence, basis, visual_features


OUTPUT = CHECKPOINT.parent
PRACTICE = OUTPUT / 'source-practice.json.gz'
HELD = OUTPUT / 'source-held.json.gz'
RETENTION = OUTPUT / 'source-retention.json.gz'
SURVIVAL_MARGIN = .54
OPPONENT_SPEED = .45
CAP = 1040
DT = CAMERA_SECONDS
SCHEDULES = {'practice': 1546719849, 'probe_held': 1076418512, 'validation': 694410638,
             'held': 1718914334, 'retention': 345170192}
VALIDATION_HORIZONS = (4, 7, 10)
EXPLORATION_ALIASES = np.array(ACTIONS)


class Scene:
    """Keep episodic component histories independent of screen orientation and action IDs."""

    def __init__(self):
        self.extended = None
        self.mover = None
        self.tick = -1
        self.signature = sha256()

    @staticmethod
    def regions(image):
        # The native presentation is a common two-color palette for all objects.
        # Background occupies the corners. The light value and RGB order vary.
        panel = image[..., 0]
        labels, _ = ndimage.label(panel != panel[0, 0])
        detections = []
        for y, x in ndimage.find_objects(labels):
            width, height = x.stop - x.start, y.stop - y.start
            center = np.array((x.start + x.stop, y.start + y.stop), float) / (2. * panel.shape[1])
            detections.append((center, max(width, height) / (2. * panel.shape[1]),
                               min(width, height) / max(width, height)))
        return detections

    def update(self, image):
        self.tick += 1
        records = self.regions(image)
        blades = [(p, scale) for p, scale, aspect in records if aspect < .45]
        compact = [p for p, _, aspect in records if aspect >= .45]
        if self.extended is None:
            self.extended = sorted(blades, key=lambda e: (-e[1], *e[0]))[:2]
        measured = []
        for p, scale in self.extended:
            chosen = min(range(len(blades)), key=lambda i: np.linalg.norm(blades[i][0] - p), default=None)
            if chosen is not None and np.linalg.norm(blades[chosen][0] - p) < scale:
                q, r = blades.pop(chosen)
                # A merged component can have a widened silhouette. Preserve its
                # typical extent for the next association.
                measured.append((q, .92 * scale + .08 * r, True, q - p))
            else:
                measured.append((p, scale, False, np.zeros(2)))
        self.extended = [(p, span) for p, span, _, _ in measured]
        if self.mover is None:
            mover = next(iter(compact), None)
        else:
            mover = min(compact, key=lambda p: np.linalg.norm(p - self.mover), default=None)
            if mover is not None and np.linalg.norm(mover - self.mover) > .135:
                mover = None
        if mover is not None:
            self.mover = mover
        pixels = np.array([[-1., -1.] if mover is None else mover, *(row[0] for row in measured)])
        self.signature.update(pixels.astype('<f4').tobytes())
        tracks = tuple(Track(i, tuple(p), tuple(v), visible, self.tick + 1)
                       for i, (p, _, visible, v) in enumerate(measured))
        return tracks, mover


class Collector:
    """Probe anonymous action effects, then sample motor commands or score their forecasts."""

    def __init__(self, seed, source=None, plant=None, horizon=7):
        self.rng = np.random.default_rng(seed)
        self.motor = _VisualActionEffects(np.random.default_rng(seed))
        self.scene = Scene()
        self.source, self.plant, self.horizon = source, plant, horizon
        self.views = []
        self.actions = []
        self.remaining = 0
        self.action = None
        self.decision_counts = np.zeros(len(ACTIONS), dtype=int)
        self.forecasts = 0

    def observe(self, frame, previous, _score):
        tracks, mover = self.scene.update(frame)
        self.views.append((tracks, np.full(2, np.nan) if mover is None else mover))
        self.motor.observe(Binding(None, None, tracks), previous)

    def path(self):
        index = self.motor.identity
        blade = np.full((len(self.views), 2), np.nan)
        for n, (tracks, _) in enumerate(self.views):
            if index is not None and index < len(tracks) and tracks[index].observed:
                blade[n] = tracks[index].position
        return blade, np.asarray([mover for _, mover in self.views])

    def choose(self):
        if self.motor.effects is None:
            action = self.motor.probe()
        elif self.source is None:
            if self.remaining == 0:
                self.action = int(self.rng.choice(EXPLORATION_ALIASES))
                self.remaining = int(self.rng.integers(8, 25))
            self.remaining -= 1
            action = self.action
        else:
            action = self.select()
        self.actions.append(action)
        if self.motor.effects is not None:
            self.decision_counts[self.motor.effects.index(action)] += 1
        return action

    def select(self):
        negative, neutral, positive = self.motor.effects
        if len(self.views) < 10:
            return neutral
        index = self.motor.identity
        recent = self.views[-6:]
        if not all(index < len(tracks) and tracks[index].observed for tracks, _ in recent):
            return neutral
        blade = np.array([tracks[index].position for tracks, _ in recent])
        mover = np.array([m for _, m in recent])
        if not np.isfinite(mover).all():
            return neutral
        span = self.scene.extended[index][1]
        cue = visual_features(blade, mover, dt=DT, scale=span)
        if not (.45 < cue[0] < 4.2 and cue[2] < 5.0 and cue[3] > .15):
            return neutral
        signed = np.array([self.motor.effects.index(a) - 1 for a in self.actions[-8:]])
        history = np.array((blade[-1] - blade[-5]) / span) @ self.motor.axis
        actor_forecast = forecast(self.plant, signed, history, self.horizon)
        blades = blade[-1] + actor_forecast[..., None] * span * self.motor.axis
        t = np.arange(5, dtype=float) - 2.
        velocity = t @ mover[-5:] / (t @ t)
        ball = mover[-1] + np.arange(1, self.horizon + 1)[:, None] * velocity
        both = np.broadcast_to(ball, blades.shape)
        actor_path = np.concatenate((np.broadcast_to(blade[-4:], (3, 4, 2)), blades), axis=1)
        mover_path = np.concatenate((np.broadcast_to(mover[-4:], (3, 4, 2)), both), axis=1)
        self.forecasts += 1
        scores = self.source.predict(actor_path, mover_path, dt=DT, scale=span)
        chosen = int(np.argmax(scores)) if np.max(scores) > scores[1] + .0025 else 1
        return (negative, neutral, positive)[chosen]


def launch_cases(name, count):
    rng = np.random.default_rng(SCHEDULES[name])
    return cases_from_seeds(rng.choice(2147483646, size=count, replace=False) + 1)


def batches(cases, size):
    items = iter(cases)
    while batch := list(islice(items, size)):
        yield batch


def actuator_samples(agent):
    """Identify a native-only action-effect map from observed motion under probes."""
    axis, effects = agent.motor.axis, agent.motor.effects
    if effects is None:
        return []
    blade, _ = agent.path()
    blade = blade @ axis / agent.scene.extended[agent.motor.identity][1]
    commands = np.array([effects.index(a) - 1 for a in agent.actions])
    samples = []
    for i in range(8, min(len(blade), len(commands)) - max(VALIDATION_HORIZONS), 3):
        past = blade[i - 4:i + 1]
        future = blade[i + 1:i + max(VALIDATION_HORIZONS) + 1]
        if not np.isfinite(np.r_[past, future]).all():
            continue
        # Discard clipped/stopped apparent motion from the local plant fit.
        if np.max(np.abs(future - past[-1])) > 1.6:
            continue
        samples.append([commands[i - 8:i].tolist(), float(past[-1] - past[0]),
                        commands[i:i + len(future)].tolist(), (future - past[-1]).tolist()])
    return samples


def cue_samples(agent, ticks):
    if agent.motor.effects is None:
        return []
    blade, mover = agent.path()
    if len(blade) < 5:
        return []
    span = agent.scene.extended[agent.motor.identity][1]
    indices = np.arange(10, len(blade) - 1, 3)
    windows = indices[:, None] + np.arange(-4, 1)
    features = visual_features(blade[windows], mover[windows], dt=DT, scale=span)
    time = DT * (np.arange(5) - 2.)
    speeds = np.linalg.norm(np.einsum('t,ntd->nd', time / (time @ time), mover[windows]), axis=-1)
    remaining = ticks / NATIVE_HZ - indices * DT
    qualified = (np.isfinite(features).all(axis=1) & (features[:, 0] >= .45) & (features[:, 0] <= 3.8)
                 & (features[:, 2] <= 5.0) & (features[:, 3] >= .18) & (speeds >= .005))
    indices, remaining, features, speeds = (a[qualified] for a in (indices, remaining, features, speeds))
    expected = features[:, 0] * span / speeds
    # The task supplies elapsed episode duration as the only outcome. Comparable
    # survival is duration beyond the estimated travel to the controlled object.
    labels = remaining - expected >= SURVIVAL_MARGIN
    return [dict(frame=int(i), features=f.tolist(), expected_seconds=float(t), remaining_seconds=float(r),
                 survival=bool(y)) for i, f, t, r, y in zip(indices, features, expected, remaining, labels)]


def observe(cases, batch_size, name):
    raw, plant = [], []
    for j, batch in enumerate(batches(cases, batch_size)):
        agents = [Collector(case.exploration) for case in batch]
        results = _native_rallies(batch, agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for row, agent in zip(results, agents, strict=True):
            samples = cue_samples(agent, row['ticks'])
            raw.append(dict(case=row['case'], ticks=row['ticks'], action_sha256=row['action_sha256'],
                            track_sha256=agent.scene.signature.hexdigest(),
                            found_effects=agent.motor.effects is not None, cues=samples))
            plant.extend(actuator_samples(agent))
        if j % 4 == 0:
            print(name, len(raw), sum(len(r['cues']) for r in raw), flush=True)
    return raw, plant


def response_row(past, history, future, horizon):
    return np.r_[1., history, past, np.asarray(future)[:horizon]]


def fit_plant(samples):
    result = []
    for j in range(1, max(VALIDATION_HORIZONS) + 1):
        x = np.array([response_row(p, v, u, j) for p, v, u, _ in samples])
        y = np.array([d[j - 1] for _, _, _, d in samples])
        result.append(np.linalg.solve(x.T @ x + .30 * np.eye(x.shape[1]), x.T @ y).tolist())
    return result


def forecast(plant, past, history, horizon):
    return np.array([[response_row(past, history, np.full(j, action), j) @ plant[j - 1]
                      for j in range(1, horizon + 1)] for action in ACTIONS])


def flatten(rows):
    cue = [(x['features'], x['survival'], x['frame']) for row in rows for x in row['cues']]
    features, labels, ticks = (np.asarray(v) for v in zip(*cue, strict=True))
    return features, labels, ticks


def fit_duration(raw):
    x, labels, ticks = flatten(raw)
    prior = float(labels.mean())
    mean = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), np.array((.4, .25, .4, .2, .25, .15, .16, .15)))
    design = basis((x - mean) / scale)
    regularizer = max(1., .003 * len(x)) * np.eye(design.shape[1])
    fitted = {}
    scramble = labels.copy()
    rng = np.random.default_rng(1681277503)
    # Keep elapsed-time and approach-range marginals. Remove associations
    # within each natural time/geometry stratum between tracks and continuation.
    strata = np.column_stack((np.minimum(ticks // 12, 8), x[:, 0].astype(int)))
    for stratum in np.unique(strata, axis=0):
        indices = np.flatnonzero((strata == stratum).all(axis=1))
        scramble[indices] = rng.permutation(labels[indices])
    for name, y in (('pong', labels), ('shuffled', scramble)):
        fitted[name] = np.linalg.solve(design.T @ design + regularizer, design.T @ y).tolist()
    fitted['cold'] = [prior] + [0.] * (design.shape[1] - 1)
    # Preserve a soft support fringe around observed approach features.
    core = np.quantile(x, (.015, .985), axis=0)
    margin = np.maximum((core[1] - core[0]) * .28, scale * .7)
    outer = core + np.array((-1., 1.))[:, None] * margin
    return dict(schema='tt24-visual-duration-source-1', features=list(FEATURES), examples=len(x),
                prior=prior, mean=mean.tolist(), scale=scale.tolist(), core=core.tolist(), outer=outer.tolist(),
                weights=fitted, positive=int(labels.sum()), shuffle_seed=1681277503)


def prediction_scores(checkpoint, rows):
    x, y, _ = flatten(rows)
    count = len(y)
    positives = int(y.sum())
    result = {}
    for mode in MODES:
        scored = VisualConsequence.from_checkpoint(mode, checkpoint).score_features(x, details=True)
        p, support = scored['score'], scored['support']
        auc = (rankdata(p)[y].sum() - positives * (positives + 1) / 2) / (positives * (count - positives))
        result[mode] = dict(auc=float(auc), brier=float(np.mean((p - y)**2)),
                            scored_examples=count, positive=positives,
                            mean_support=float(np.mean(support)), covered=float(np.mean(support > 0.)))
    return result


def closed_loop(cases, model, plant, batch_size, horizon):
    schedule = [(case, mode) for case in cases for mode in MODES]
    result = {mode: [] for mode in MODES}
    for batch in batches(schedule, batch_size):
        agents = [Collector(case.exploration, model[mode], plant, horizon) for case, mode in batch]
        rows = _native_rallies([case for case, _ in batch], agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for (_, mode), row, agent in zip(batch, rows, agents, strict=True):
            effects = None if agent.motor.effects is None else list(agent.motor.effects)
            result[mode].append(dict(row, track_sha256=agent.scene.signature.hexdigest(),
                                     alias_actions=agent.decision_counts.tolist(), forecasts=agent.forecasts,
                                     visual_effects=effects, relation_executed=bool(agent.forecasts)))
    return result


def summarize(rows):
    result = {}
    for mode, runs in rows.items():
        result[mode] = dict(summarize_pong(runs),
                            seconds=float(np.mean([r['ticks'] / NATIVE_HZ for r in runs])),
                            forecasts=sum(r['forecasts'] for r in runs),
                            nonneutral=sum(r['alias_actions'][0] + r['alias_actions'][2] for r in runs))
    duration = {mode: np.array([r['ticks'] for r in runs]) / NATIVE_HZ for mode, runs in rows.items()}
    paired = {}
    bootstrap = np.random.default_rng(1202464109).integers(len(duration['pong']), size=(4096, len(duration['pong'])))
    for mode in ('cold', 'shuffled'):
        delta = duration['pong'] - duration[mode]
        paired[mode] = dict(seconds=float(np.mean(delta)), longer=int((delta > 0).sum()),
                            shorter=int((delta < 0).sum()), tied=int((delta == 0).sum()),
                            bootstrap95=np.quantile(delta[bootstrap].mean(axis=1), (.025, .975)).tolist())
    return dict(modes=result, pong_minus=paired)


def qualify(args):
    began = perf_counter()
    sizes = dict(practice=args.train, probe_held=args.probe_held, validation=args.validation, held=args.held)
    schedule = {k: launch_cases(k, size) for k, size in sizes.items()}
    all_seeds = [c.serve for cases in schedule.values() for c in cases]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('native training and qualification schedules overlap')
    roster = {k: [asdict(case) for case in cases] for k, cases in schedule.items()}
    if PRACTICE.exists():
        practice = _read(PRACTICE)
        if practice['roster'] != roster:
            raise ValueError('existing native source uses a different schedule')
        checkpoint, hold = practice['provisional'], practice['probe_held']
    else:
        raw, responses = observe(schedule['practice'], args.batch, 'practice')
        hold, _ = observe(schedule['probe_held'], args.batch, 'probe_held')
        checkpoint = dict(fit_duration(raw), plant=fit_plant(responses), actuator_samples=len(responses))
        practice = dict(schema='tt24-source-practice-1', roster=roster, practice=raw, probe_held=hold,
                        provisional=checkpoint, survival_margin=SURVIVAL_MARGIN, cap=CAP,
                        camera_tick_stride=CAMERA_TICKS, seconds=perf_counter() - began)
        _write(PRACTICE, practice)
    checkpoint = dict(checkpoint, source_practice_sha256=_hash(PRACTICE))
    if CHECKPOINT.exists():
        if _read(CHECKPOINT) != checkpoint:
            raise ValueError('earlier frozen source weights differ from the fitted checkpoint')
    else:
        _write(CHECKPOINT, checkpoint)
    print('source frozen', _hash(CHECKPOINT), flush=True)
    model = {mode: VisualConsequence.from_checkpoint(mode, checkpoint) for mode in MODES}
    print('native random held prediction', prediction_scores(checkpoint, hold), flush=True)
    screen = {}
    for horizon in VALIDATION_HORIZONS:
        rows = closed_loop(schedule['validation'], model, checkpoint['plant'], args.batch, horizon)
        screen[horizon] = summarize(rows)
        print('validation', horizon, screen[horizon]['pong_minus'], flush=True)
    horizon = max(screen, key=lambda h: screen[h]['modes']['pong']['seconds'])
    rows = closed_loop(schedule['held'], model, checkpoint['plant'], args.batch, horizon)
    result = dict(schema='tt24-source-held-1', source_sha256=_hash(CHECKPOINT),
                  practice_sha256=_hash(PRACTICE), prediction=prediction_scores(checkpoint, hold),
                  validation=screen, horizon=horizon, rows=rows, summary=summarize(rows), seconds=perf_counter() - began)
    _write(HELD, result)
    print('held', result['summary'], flush=True)


def retention(args):
    checkpoint = _read(CHECKPOINT)
    qualification = _read(HELD)
    if qualification['source_sha256'] != _hash(CHECKPOINT):
        raise ValueError('retention requires the exact qualified visual weights')
    model = {mode: VisualConsequence.from_checkpoint(mode, checkpoint) for mode in MODES}
    schedule = launch_cases('retention', args.retention)
    other = _read(PRACTICE)['roster']
    if {case.serve for case in schedule} & {row['serve'] for cases in other.values() for row in cases}:
        raise ValueError('the retention course overlaps source preparation')
    rows = closed_loop(schedule, model, checkpoint['plant'], args.batch, qualification['horizon'])
    result = dict(schema='tt24-source-retention-1', source_sha256=_hash(CHECKPOINT),
                  schedule=[asdict(case) for case in schedule], summary=summarize(rows), rows=rows)
    if args.phase == 'retention-pre':
        _write(RETENTION, dict(before=result))
    else:
        archived = _read(RETENTION)
        original = archived['before']
        if original['schedule'] != result['schedule'] or original['source_sha256'] != result['source_sha256']:
            raise ValueError('the later course must reuse the original schedule and source bytes')
        matched = {mode: sum(a == b for a, b in zip(original['rows'][mode], rows[mode], strict=True)) for mode in MODES}
        if any(n != len(schedule) for n in matched.values()):
            raise ValueError('frozen native command, optics, or referee retention mismatch')
        result['downstream_receipts'] = {path.name: _hash(path) for path in args.evidence}
        _write(RETENTION, dict(before=original, after=result, matching=matched))
    print(args.phase, result['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('qualify', 'retention-pre', 'retention-post'))
    parser.add_argument('--train', type=int, default=576)
    parser.add_argument('--probe-held', type=int, default=120)
    parser.add_argument('--validation', type=int, default=54)
    parser.add_argument('--held', type=int, default=144)
    parser.add_argument('--retention', type=int, default=48)
    parser.add_argument('--batch', type=int, default=40)
    parser.add_argument('--evidence', type=Path, nargs='*', default=(), help='downstream receipts for the final revisit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'qualify':
        qualify(args)
    else:
        retention(args)


if __name__ == '__main__':
    main()

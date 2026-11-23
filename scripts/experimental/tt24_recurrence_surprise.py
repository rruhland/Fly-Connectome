"""Fit an RGB mover-divergence witness from random-time matched native pulses."""

import argparse
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.stats import rankdata
import torch

from .tt3_pong import CAMERA_SECONDS, NATIVE_HZ, _native_rallies, cases_from_seeds
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_recurrence import CAP, OUTPUT, PULSE, PulseCollector, effect_features, fit as fit_credit, qualification, slope
from .tt24_source_native import OPPONENT_SPEED, batches


RECEIPT = OUTPUT / 'recurrence-surprise-native.json.gz'
WEIGHTS = OUTPUT / 'recurrence-surprise-weights.json'
SCHEDULE = {'practice': 676855301, 'held': 1603111956}
EXTENDED_SCHEDULE = {'practice': 108424318, 'held': 1284610184}
GATE_CUTOFF = .50
SOURCE_OBSERVATIONS = 43


class RandomPulse(PulseCollector):
    """Deliver a scheduled pulse once the anonymous optical actor can be tracked."""

    def __init__(self, seed, pulse, pulse_frames, witness_frames):
        super().__init__(seed, pulse, pulse_frames)
        self.witness_frames = witness_frames
        self.wait = int(self.rng.integers(0, 9))
        self.eligible = None

    def qualify(self):
        if self.eligible is None:
            self.eligible = len(self.actions)
        if len(self.actions) - self.eligible < self.wait:
            return None
        identity = self.motor.identity
        recent = self.views[-5:]
        if len(recent) < 5 or not all(identity < len(rows) and rows[identity].observed for rows, _ in recent):
            return None
        actor = np.array([rows[identity].position for rows, _ in recent])
        mover = np.array([ball for _, ball in recent])
        if not np.isfinite(mover).all():
            return None
        digest = sha256(np.asarray([actor, mover], dtype='<f4').tobytes()).hexdigest()
        return dict(frame=len(self.actions), span=self.scene.extended[identity][1], prefix=digest)

    def trace(self):
        if self.cue is None:
            return None
        actor, mover = self.path()
        begin = self.cue['frame'] - 4
        end = begin + 5 + self.witness_frames
        tracks = np.stack((actor[begin:end], mover[begin:end]), axis=1)
        return dict(tracks=[row.tolist() if np.isfinite(row).all() else None for row in tracks], **self.cue)


def collect(seeds, name, batch_size, pulse_frames, witness_frames):
    options = [(point, pulse) for point in cases_from_seeds(seeds) for pulse in (-1, 1)]
    results = []
    for i, work in enumerate(batches(options, batch_size)):
        agents = [RandomPulse(point.exploration, pulse, pulse_frames, witness_frames) for point, pulse in work]
        rows = _native_rallies([point for point, _ in work], agents, max_ticks=CAP, opponent_speed=OPPONENT_SPEED)
        for (point, pulse), agent, row in zip(work, agents, rows, strict=True):
            results.append(dict(seed=point.serve, pulse=pulse, trace=agent.trace(), ticks=row['ticks'],
                                action_sha256=row['action_sha256'], track_sha256=agent.scene.signature.hexdigest()))
        if i % 4 == 0:
            print(name, len(results), sum(row['trace'] is not None for row in results), flush=True)
    return [results[i:i + 2] for i in range(0, len(results), 2)]


def numeric(trace):
    return np.array([row if row is not None else [[np.nan, np.nan], [np.nan, np.nan]] for row in trace['tracks']])


def windows(first, second, dt, *, with_labels=False):
    """Generate regular optical histories, independent of encounter geometry."""
    if first is None or second is None:
        return
    if first['prefix'] != second['prefix']:
        raise ValueError('randomized pulse branches changed before the intervention')
    a, b = numeric(first), numeric(second)
    stop = min(len(a), len(b))
    if stop < 8 or not np.isfinite(a[:5]).all() or not np.array_equal(a[:5], b[:5]):
        return
    initial = a[:5]
    span = first['span']
    speed = max(.06 * span, np.linalg.norm(slope(initial[:, 1], dt * np.arange(5.))))
    for end in range(8, stop + 1 - (2 if with_labels else 0)):
        a_now, b_now = a[end - 4:end], b[end - 4:end]
        if not np.isfinite(np.stack((a_now, b_now))).all():
            continue
        delta = a_now - b_now
        separation = np.linalg.norm(delta, axis=-1) / span
        time = dt * np.arange(4.)
        rates = [slope(delta[:, track], time) for track in range(2)]
        change = np.linalg.norm(rates, axis=-1) / speed
        gate = np.arcsinh([separation[-1, 1], np.mean(separation[:, 1]), change[1], separation[-1, 0], change[0]])
        outcome = None
        if with_labels:
            after = (a[end:end + 2] - b[end:end + 2])[:, 1]
            if not np.isfinite(after).all():
                continue
            outcome = bool(np.min(np.linalg.norm(after, axis=-1)) / span > .16)
        scores = [effect_features(initial[:, 0], initial[:, 1], side[:, 0], side[:, 1], scale=span, dt=dt)
                  for side in (a_now, b_now)]
        if all(s is not None for s in scores):
            yield end, gate, outcome, scores


def fit_gate(pairs):
    cues = [row for a, b in pairs for row in windows(a['trace'], b['trace'], CAMERA_SECONDS, with_labels=True)]
    values = np.array([row[1] for row in cues])
    y = np.array([row[2] for row in cues])
    center, scale = values.mean(axis=0), np.maximum(.08, values.std(axis=0))
    design = np.column_stack((np.ones(len(values)), np.clip((values - center) / scale, -4., 4.)))
    weights = np.linalg.solve(design.T @ design + 2. * np.eye(design.shape[1]), design.T @ y)
    return dict(mean=center.tolist(), scale=scale.tolist(), weights=weights.tolist(), examples=len(y), positives=int(y.sum()))


def gate_score(feature, gate):
    value = np.clip((feature - gate['mean']) / gate['scale'], -4., 4.)
    return float(np.clip(np.r_[1., value] @ gate['weights'], 0., 1.))


def select(first, second, dt, gate):
    for frame, feature, _, effects in windows(first, second, dt):
        support = gate_score(feature, gate)
        if support > GATE_CUTOFF:
            return frame, support, effects
    return None


def contrasts(rows, gate):
    examples = []
    for a, b in rows:
        witness = select(a['trace'], b['trace'], CAMERA_SECONDS, gate)
        if witness is None or abs(a['ticks'] - b['ticks']) < 12:
            continue
        _, _, (left, right) = witness
        target = np.log1p(a['ticks'] / NATIVE_HZ) - np.log1p(b['ticks'] / NATIVE_HZ)
        examples.append((a['seed'], left - right, target))
    return examples


def gate_held(rows, gate):
    measured = [row for a, b in rows for row in windows(a['trace'], b['trace'], CAMERA_SECONDS, with_labels=True)]
    scores = np.array([gate_score(row[1], gate) for row in measured])
    y = np.asarray([row[2] for row in measured])
    n = int(y.sum())
    auc = (rankdata(scores)[y].sum() - n * (n + 1) / 2) / (n * (~y).sum())
    return dict(auc=float(auc), positives=n, windows=len(y))


def run(train, held, batch_size, *, extended):
    began = perf_counter()
    roster = EXTENDED_SCHEDULE if extended else SCHEDULE
    schedule = {name: np.random.default_rng(roster[name]).choice(2147483646, n, replace=False).tolist()
                for name, n in dict(practice=train, held=held).items()}
    if set(schedule['practice']).intersection(schedule['held']):
        raise ValueError('randomized native gate schedules overlap')
    receipt = OUTPUT / 'recurrence-surprise-extended-native.json.gz' if extended else RECEIPT
    weights = OUTPUT / 'recurrence-surprise-extended-weights.json' if extended else WEIGHTS
    pulse_frames, witness_frames = (12, 105) if extended else (PULSE, SOURCE_OBSERVATIONS)
    report = _read(receipt) if receipt.exists() else dict(schema='tt24-recurrence-surprise-native-1', schedule=schedule)
    if report['schedule'] != schedule:
        raise ValueError('random-time native course uses a different schedule')
    for name, seeds in schedule.items():
        if name not in report:
            report[name] = collect(seeds, name, batch_size, pulse_frames, witness_frames)
            _write(receipt, report)
    gate = fit_gate(report['practice'])
    data, check = (contrasts(report[name], gate) for name in ('practice', 'held'))
    if len(data) < 2:
        raise ValueError('randomized gate produced too few native recurrence comparisons')
    state = fit_credit(data, shuffle_seed=411508008)
    result = qualification(state, check)
    gate_result = gate_held(report['held'], gate)
    result['passed'] = result['passed'] and gate_result['auc'] > .75
    _write(weights, dict(state, gate=gate, gate_held=gate_result, qualification=result))
    report.update(qualification=result, gate_held=gate_result, weights_sha256=_hash(weights),
                  pulse_frames=pulse_frames, witness_frames=witness_frames, seconds=perf_counter() - began)
    _write(receipt, report)
    print('random-pulse gate', gate_result, result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=250)
    parser.add_argument('--held', type=int, default=180)
    parser.add_argument('--batch', type=int, default=42)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.train, args.held, args.batch, extended=args.extended)


if __name__ == '__main__':
    main()

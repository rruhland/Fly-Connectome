"""Test a Pong-learned evidence metric on physically measured later-rally motor choices."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import betaincinv
from scipy.stats import rankdata

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt22_branch_motor import FEATURE_SCALE, MOTORS, OutcomeScorer

ARCHIVES = ROOT / 'artifacts' / 'tt22'
OUTPUT = ROOT / 'artifacts' / 'tt23'
TRACE = ARCHIVES / 'value-source.json.gz'
SCORES = ARCHIVES / 'duration-pong.json.gz'
PRACTICE = ARCHIVES / 'branch-motor-practice.json.gz'
CHECKPOINT = OUTPUT / 'source-metric-checkpoint.json.gz'
CURVES = OUTPUT / 'source-metric-curves.json.gz'
MODES = ('pong', 'cold', 'shuffled')
BUDGETS = (0, 4, 8, 16, 24, 32)
EARLY = (4, 8, 16)
LIVE_BUDGET = 16
LIVE_TRAIN = 1648291713
ARCHIVE_SEED = 1389775219
SHUFFLE = 334685171
MOTOR_GRID = np.asarray(MOTORS, float)
MOTOR_SCALE = np.array((.055, .14, .018, 1.50))


def approach(offset, side_speed, closing, time, projected):
    """Mirror-invariant lateral kinematics in units of approach travel."""
    depth = np.maximum(.001, closing * time)
    return np.stack((np.arcsinh(abs(offset) / depth),
                     np.arcsinh(np.sign(offset) * side_speed / closing),
                     np.arcsinh(abs(projected) / depth),
                     np.log1p(time)), axis=-1)


def optical_approach(actor, mover, tick):
    visible = actor[max(0, tick - 20):tick + 1]
    visible = visible[np.isfinite(visible).all(axis=1)]
    if len(visible) < 5 or not np.isfinite(np.r_[actor[tick], mover[tick - 3], mover[tick]]).all():
        return None
    span = np.ptp(visible, axis=0)
    transverse = int(np.argmax(span))
    if span[transverse] < .009 or span[transverse] < 2. * span[1 - transverse]:
        return None
    p = mover[tick] - actor[tick]
    v = (mover[tick] - mover[tick - 3]) / 3.
    closing = -np.sign(p[1 - transverse]) * v[1 - transverse]
    if closing < .004:
        return None
    time = abs(p[1 - transverse]) / closing
    if not 3. <= time <= 24.:
        return None
    offset, side_speed = p[transverse], v[transverse]
    return approach(offset, side_speed, closing, time, offset + time * side_speed)


def pong_samples(traces, public):
    """Join archived RGB tracks to native public return events by full action digest."""
    samples = []
    matched = 0
    for episode in traces:
        launch = episode['case']['serve']
        row = public.get((launch, episode['action_sha256']))
        if row is None:
            continue
        if row['case'] != episode['case'] or row['ticks'] != episode['ticks']:
            raise ValueError('native optical and referee archives disagree')
        matched += 1
        actor, mover = (np.array([entry[index] if entry[index] is not None else [np.nan, np.nan]
                                 for entry in episode['trace']]) for index in (0, 1))
        for event in row['contacts']:
            tick = (event['tick'] - 1) // 4 - 8
            if tick < 3:
                continue
            feature = optical_approach(actor, mover, tick)
            if feature is not None:
                samples.append(dict(launch=launch, policy=episode['policy'], tick=tick,
                                    features=feature.tolist(), legal=event['returned']))
    return samples, matched


def fit_pong():
    traces, scored = _read(TRACE), _read(SCORES)
    public = {(e['case']['serve'], e['action_sha256']): e
              for group in ('screen', 'validation', 'held') for runs in scored[group].values() for e in runs}
    train, matched_train = pong_samples(traces['train'], public)
    held, matched_held = pong_samples(traces['held'], public)
    if {s['launch'] for s in train} & {s['launch'] for s in held}:
        raise ValueError('native source train and held launches overlap')
    x = np.array([s['features'] for s in train])
    y = np.array([s['legal'] for s in train], dtype=float)
    h = np.array([s['features'] for s in held])
    z = np.array([s['legal'] for s in held], dtype=bool)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .075)
    normalized = np.clip((x - mean) / scale, -4., 4.)
    penalty = normalized.T @ normalized + 5. * np.eye(x.shape[1])
    permutation = np.random.default_rng(SHUFFLE).permutation(len(y))
    fitted, results = {}, {}
    for name, labels in (('pong', y), ('shuffled', y[permutation])):
        coefficients = np.linalg.solve(penalty, normalized.T @ (labels - labels.mean()))
        fitted[name] = coefficients.tolist()
        prediction = np.clip(labels.mean() + np.clip((h - mean) / scale, -4., 4.) @ coefficients, 0., 1.)
        n, positives = len(z), int(z.sum())
        auc = (rankdata(prediction)[z].sum() - positives * (positives + 1) / 2) / (positives * (n - positives))
        results[name] = dict(auc=float(auc), brier=float(np.mean((prediction - z)**2)))
    return dict(sources={p.name: _hash(p) for p in (TRACE, SCORES)}, arms=fitted, mean=mean.tolist(),
                scale=scale.tolist(), train=train, held=held, matched_train=matched_train,
                matched_held=matched_held, shuffle=permutation.tolist(), held_quality=results)


def target_coordinates(features, model, mode):
    features = np.asarray(features)
    if mode == 'cold':
        return np.zeros(features.shape[:-1])
    time = np.maximum(.1, features[..., 9])
    closing = np.maximum(.005, -features[..., 4])
    offset = features[..., 0] - features[..., 1] - features[..., 6]
    native = approach(offset, features[..., 3], closing, time, -features[..., 6])
    coordinate = np.clip((native - model['mean']) / model['scale'], -4., 4.)
    direction = np.asarray(model['arms'][mode])
    return coordinate @ (direction / np.linalg.norm(direction))


def context_kernel(a, b, model, mode):
    distance = np.sum(((a[:, None] - b[None, :]) / FEATURE_SCALE)**2, axis=2)
    left, right = (target_coordinates(features, model, mode) for features in (a, b))
    source_distance = (left[:, None] - right[None, :])**2
    # Retain a shared context component for observations the source considers dissimilar.
    return np.exp(-.15 * distance) * (.40 + .60 * np.exp(-.5 * source_distance))


def action_kernel():
    gap = (MOTOR_GRID[:, None] - MOTOR_GRID[None, :]) / MOTOR_SCALE
    return np.exp(-.5 * np.sum(gap**2, axis=2))


def opening_prior(openings):
    return np.clip(OutcomeScorer(openings, contextual=False).priors[0], .02, .90)


def legal_matrix(rows):
    effect = np.full((len(rows), len(MOTORS)), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        for motor, legal, *_ in row['outcomes']:
            effect[i, motor] = legal
    return effect


def acquire(legal, contexts, motors, prior, order, uniforms, evaluate):
    """Run a causal bandit replay. Read a branch label only after choosing its motor."""
    alpha = np.broadcast_to(2. * prior, legal.shape).copy()
    beta = np.broadcast_to(2. * (1. - prior), legal.shape).copy()
    observed = []
    choices = {}
    for step in range(len(order) + 1):
        if step in BUDGETS and len(evaluate):
            posterior = alpha[evaluate] / (alpha[evaluate] + beta[evaluate])
            choices[step] = np.argmax(np.where(legal[evaluate] >= 0, posterior, -1.), axis=1)
        if step == len(order):
            break
        cue = order[step]
        eligible = np.flatnonzero(legal[cue] >= 0)
        sampled = betaincinv(alpha[cue, eligible], beta[cue, eligible], uniforms[step, eligible])
        motor = eligible[np.argmax(sampled)]
        result = legal[cue, motor]
        weights = contexts[:, cue, None] * motors[None, :, motor]
        if result:
            alpha += weights
        else:
            beta += weights
        observed.append((int(cue), int(motor), bool(result)))
    return observed, choices


def train_live(path):
    model = fit_pong()
    data = _read(PRACTICE)
    openings = [row for row in data['contexts'] if row['leg'] == 0]
    later = [row for row in data['contexts'] if row['leg'] > 0]
    features = np.array([r['features'] for r in later])
    legal = legal_matrix(later)
    prior = opening_prior(openings)
    rng = np.random.default_rng(LIVE_TRAIN)
    order = rng.permutation(len(later))[:LIVE_BUDGET]
    uniforms = rng.uniform(.000001, .999999, size=(len(order), len(MOTORS)))
    motors = action_kernel()
    logs = {}
    for name in MODES:
        context = context_kernel(features, features, model, name)
        observations, _ = acquire(legal, context, motors, prior, order, uniforms, ())
        logs[name] = observations
    result = dict(schema='tt23-source-metric-checkpoint-1', source=model, practice=PRACTICE.name,
                  practice_sha256=_hash(PRACTICE), schedule=LIVE_TRAIN, budget=LIVE_BUDGET,
                  order=order.tolist(), observations=logs, donors=data['donors'])
    _write(path, result)
    print('Pong cues', len(model['train']), len(model['held']), 'native held quality', model['held_quality'], flush=True)
    print('Frozen practice outcomes', {name:sum(row[2] for row in log) for name,log in logs.items()}, flush=True)
    return result


class SourceMetricScorer:
    """Use Pong to pool outcome evidence, with the existing target-only opening."""

    def __init__(self, openings, later, source, mode, records):
        self.opening = OutcomeScorer(openings, contextual=False)
        self.prior = np.clip(self.opening.priors[0], .02, .90)
        self.source, self.mode = source, mode
        self.experiences = np.array([later[cue]['features'] for cue, _, _ in records])
        self.actions = np.array([motor for _, motor, _ in records])
        self.earned = np.array([reward for _, _, reward in records])
        self.motors = action_kernel()

    def ranking(self, features, later, indices, errors):
        if not later:
            return self.opening.ranking(features, False, indices, errors)
        context = context_kernel(features[None, :], self.experiences, self.source, self.mode)[0]
        weights = context * self.motors[np.ix_(indices, self.actions)]
        score = (2. * self.prior[indices] + weights @ self.earned) / (2. + weights.sum(axis=1))
        return score - .28 * errors


def summarize(traces, train_yields, groups, repetitions):
    rng = np.random.default_rng(ARCHIVE_SEED + 1)
    donor_groups = [np.flatnonzero(groups == seed) for seed in np.unique(groups)]
    counts = np.array([len(group) for group in donor_groups])
    draws = rng.integers(len(donor_groups), size=(4096, len(donor_groups)))

    def stats(values):
        """Keep all cues from the same held rally together in the resampling."""
        total = np.array([values[group].sum(axis=0) for group in donor_groups])
        resampled = total[draws].sum(axis=1) / counts[draws].sum(axis=1)[..., None]
        return dict(mean=values.mean(axis=0).tolist(),
                    donor_bootstrap95=np.quantile(resampled, (.025, .975), axis=0).tolist())

    scores = {name: stats(values.mean(axis=0)) for name, values in traces.items()}
    for name, values in train_yields.items():
        scores[name]['practice_returns'] = float(values.mean())
    early = [BUDGETS.index(n) for n in EARLY]
    differences = {}
    for control in ('cold', 'shuffled'):
        difference = (traces['pong'] - traces[control]).mean(axis=0)
        differences[control] = stats(np.column_stack((difference, difference[:, early].mean(axis=1))))
    return dict(budgets=BUDGETS, early=EARLY, repetitions=repetitions, cue_count=len(groups),
                donor_count=len(donor_groups), scores=scores, pong_minus=differences)


def curves(checkpoint, output, repetitions):
    began = perf_counter()
    frozen = _read(checkpoint)
    if frozen['practice_sha256'] != _hash(PRACTICE):
        raise ValueError('physical practice archive changed')
    source = frozen['source']
    practice = _read(PRACTICE)
    openings = [row for row in practice['contexts'] if row['leg'] == 0]
    later = [row for row in practice['contexts'] if row['leg'] > 0]
    feature = np.array([r['features'] for r in later])
    groups = np.array([r['seed'] for r in later])
    legal = legal_matrix(later)
    context = {name: context_kernel(feature, feature, source, name) for name in MODES}
    motors = action_kernel()
    draws = np.random.default_rng(ARCHIVE_SEED).integers(2**31, size=repetitions).tolist()
    held_results = {name: np.zeros((repetitions, len(later), len(BUDGETS)), dtype=np.int8) for name in MODES}
    held_choices = {name: np.zeros_like(held_results[name], dtype=np.uint8) for name in MODES}
    earned = {name: np.zeros((repetitions, len(np.unique(groups))), dtype=np.int8) for name in MODES}
    executed = {name: [] for name in MODES}
    for fold, seed in enumerate(np.unique(groups)):
        training = np.flatnonzero(groups != seed)
        held = np.flatnonzero(groups == seed)
        prior = opening_prior([row for row in openings if row['seed'] != seed])
        for r, draw in enumerate(draws):
            rng = np.random.default_rng([draw, int(seed)])
            order = rng.permutation(training)[:max(BUDGETS)]
            uniforms = rng.uniform(.000001, .999999, size=(len(order), len(MOTORS)))
            for name in MODES:
                observed, selected = acquire(legal, context[name], motors, prior, order, uniforms, held)
                for index, budget in enumerate(BUDGETS):
                    held_choices[name][r, held, index] = selected[budget]
                    held_results[name][r, held, index] = legal[held, selected[budget]]
                earned[name][r, fold] = sum(t[2] for t in observed)
                executed[name].append(dict(fold=int(seed), repetition=r, observed=observed))
    summary = summarize(held_results, earned, groups, repetitions)
    result = dict(schema='tt23-source-metric-curves-1', checkpoint_sha256=_hash(checkpoint),
                  checkpoint=checkpoint.name, practice_sha256=frozen['practice_sha256'], draws=draws,
                  held_cues=[[r['seed'], r['tick'], r['leg']] for r in later], summary=summary,
                  held_choices={name: rows.tolist() for name, rows in held_choices.items()},
                  held_results={name: rows.tolist() for name, rows in held_results.items()},
                  practice=executed, seconds=perf_counter() - began)
    _write(output, result)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=CHECKPOINT, type=Path)
    parser.add_argument('--output', default=CURVES, type=Path)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--curves-only', action='store_true')
    parser.add_argument('--repetitions', type=int, default=32)
    args = parser.parse_args()
    if args.repetitions < 1 or (args.fit_only and args.curves_only):
        parser.error('select at most one phase with a positive repetition count')
    if not args.curves_only:
        train_live(args.checkpoint)
    if not args.fit_only:
        curves(args.checkpoint, args.output, args.repetitions)


if __name__ == '__main__':
    main()

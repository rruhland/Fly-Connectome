"""Learn a distribution of successful native optical outcomes, separately from motor commands."""

import numpy as np
from scipy.special import logsumexp

from .tt3_pong import CAMERA_SECONDS, NATIVE_HZ
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_recurrence_surprise import OUTPUT, select


NATIVE = OUTPUT / 'recurrence-surprise-extended-native.json.gz'
GATE = OUTPUT / 'recurrence-surprise-extended-weights.json'
OLD_ARM = OUTPUT / 'recurrence-surprise-arm.json.gz'
RESULT = OUTPUT / 'hindsight-source.json'
FEATURES = (0, 1, 4, 7)
MODES = ('pong', 'cold', 'shuffled')
SEED = 142874205


def visits(source, gate):
    samples = []
    for a, b in source:
        witness = select(a['trace'], b['trace'], CAMERA_SECONDS, gate)
        if witness is None or abs(a['ticks'] - b['ticks']) < NATIVE_HZ / 10:
            continue
        _, _, values = witness
        x = np.asarray(values)[:, FEATURES]
        choice = int(a['ticks'] < b['ticks'])
        samples.append((a['seed'], x, choice))
    return samples


def fit(records):
    labels = np.array([winner for _, _, winner in records])
    samples = np.stack([features for _, features, _ in records])
    scale = np.maximum(.22, samples.reshape((-1, len(FEATURES))).std(axis=0))
    center = np.median(samples.reshape((-1, len(FEATURES))), axis=0)
    rng = np.random.default_rng(SEED)
    shuffled = labels ^ rng.integers(0, 2, len(labels))
    return dict(center=center.tolist(), scale=scale.tolist(),
                prototypes={'pong': samples[np.arange(len(samples)), labels].tolist(),
                            'shuffled': samples[np.arange(len(samples)), shuffled].tolist()},
                samples=len(records), features=list(FEATURES), radius=.70, shuffle_seed=SEED)


def value(outcome, model, mode):
    value = np.asarray(outcome)
    if mode == 'cold':
        return np.zeros(value.shape[:-1])
    normalized = (value[..., None, :] - np.array(model['prototypes'][mode])) / model['scale']
    score = -.5 * np.sum(normalized**2, axis=-1) / model['radius']**2
    # A density over observed good endpoints, with a common finite penalty
    # for unseen outcomes. Transfer cannot gain merely from extrapolation.
    return np.maximum(-14., logsumexp(score, axis=-1) - np.log(score.shape[-1]))


def compare(x, outcomes, model):
    answer = {}
    for mode in MODES:
        scores = value(x, model, mode)
        advantage = scores[:, 0] - scores[:, 1]
        agreement = .5 + .5 * np.sign(advantage) * np.sign(outcomes)
        answer[mode] = dict(concordance=float(np.mean(agreement)), correct=float(np.sum(agreement)),
                            pairs=len(outcomes))
    return answer


def arm_audit(arm, gate, model):
    cases = []
    for attempt in arm['practice']:
        plus, minus = (attempt[key] for key in ('plus', 'minus'))
        witness = select(plus['trace'], minus['trace'], 1 / 60, gate)
        if witness is not None:
            _, _, features = witness
            cases.append((np.asarray(features)[:, FEATURES], plus['public_score'] - minus['public_score']))
    ordered = [row for row in cases if abs(row[1]) > .10]
    if not ordered:
        return dict(scored=len(cases), reward_non_ties=0)
    features, rewards = (np.asarray(items) for items in zip(*ordered, strict=True))
    return dict(scored=len(cases), reward_non_ties=len(ordered), ordering=compare(features, rewards, model),
                mean_quality={mode: float(np.mean(value(features, model, mode))) for mode in MODES})


def main():
    native, gate = _read(NATIVE), _read(GATE)['gate']
    practice, held = (visits(native[name], gate) for name in ('practice', 'held'))
    model = fit(practice)
    x, outcomes = (np.asarray(v) for v in zip(*[(f, 1 - 2 * o) for _, f, o in held], strict=True))
    result = dict(schema='tt24-hindsight-goal-1', value=model, native_held=compare(x, outcomes, model),
                  earlier_arm_audit=arm_audit(_read(OLD_ARM), gate, model),
                  receipt_sha256={p.name: _hash(p) for p in (NATIVE, GATE, OLD_ARM)})
    _write(RESULT, result)
    print(result['native_held'], result['earlier_arm_audit'], flush=True)


if __name__ == '__main__':
    main()

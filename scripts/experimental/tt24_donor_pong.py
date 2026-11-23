"""Compare native optical motion priors for exploration on articulated donors."""

import argparse
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import DESIGNS, FEATURES, NATIVE, ROOT, SHAPE, Native, World, _checkpoint, _exploration, _utility
from .tt24_pong_world_strong import RAW as SOURCE


RECEIPT = ROOT / 'tt24_donor_pong.json.gz'
PLAN = ROOT / 'tt24_donor_pong-plan.json'
SEEDS = (2064310765, 1215868689, 1965512817)
MODES = ('pong', 'shuffled', 'cold', 'matched_random')
BUDGETS = (0, 1, 2, 4, 8, 12)
DIRECTIONS, PRACTICE, VALID, HELD, CAP = 6, 1, 4, 8, 260
NOISE = .18
COMPONENTS = ((1, 2), (3,), (4, 5), (6,), (7, 8), (12,), (13,), (14,), (15, 16), (17, 18), (19,))
VERTICAL, HORIZONTAL, POSE = ([3, 6, 9, 12, 13, 14, 19], [1, 2, 4, 5, 7, 8, 15, 16, 17, 18], [0, 7, 8, 9, 10, 11])


def features(row, speed):
    size = .076 / row['apparent_span']
    time = 2.1 / speed
    depth, closing, gap, lateral, paddle = np.clip((row['distance'] * size / .13 - 1., row['closing_speed'] * time / 1.5,
        row['tangent_gap'] * size / .065, row['tangent_speed'] * time / .38, row['current'] * size / .04), -3., 3.)
    return np.clip((abs(gap), depth, abs(lateral), closing, abs(paddle), depth * closing, closing**2 - .5,
                    depth**2 - .5, abs(gap * closing), abs(lateral * closing), np.tanh(2. * closing)), -3., 3.)


def prepared(episodes, speed):
    x, travel, groups = [], [], []
    for index, episode in enumerate(episodes):
        for cue in episode['cues']:
            x.append(features(cue, speed))
            travel.append(np.log1p(abs(cue['later'] - cue['current']) / cue['apparent_span']))
            groups.append(index)
    return np.asarray(x), np.asarray(travel), np.asarray(groups)


def groups():
    multiplicity = np.array([len(c) for c in COMPONENTS])
    return [np.flatnonzero(multiplicity == count) for count in (1, 2)]


def multipliers(strength):
    gain = np.ones(FEATURES)
    for bucket in groups():
        ordered = bucket[np.argsort(strength[bucket], kind='stable')]
        for component, value in zip(ordered, np.exp(np.linspace(-.48, .48, len(bucket))), strict=True):
            gain[list(COMPONENTS[component])] = value
    return gain


def source_fit():
    tape = _read(SOURCE)['rows']
    sizes = tape['practice']['episodes']
    speed = np.median([abs(cue['closing_speed']) for row in sizes for cue in row['cues']])
    x, y, episode = prepared(sizes, speed)
    q, observed, _ = prepared(tape['held']['episodes'], speed)
    standard = np.maximum(.05, x.std(axis=0))
    q = (q - x.mean(axis=0)) / standard
    x = (x - x.mean(axis=0)) / standard
    altered = y.copy()
    rng = np.random.default_rng(88006217)
    for index in np.unique(episode):
        subset = episode == index
        altered[subset] = rng.permutation(y[subset])
    regular = x.T @ x + 32. * np.eye(len(COMPONENTS))
    measures, priors = {}, {}
    for name, outcomes in (('pong', y), ('shuffled', altered)):
        coefficient = np.linalg.solve(regular, x.T @ (outcomes - outcomes.mean()))
        fit = q @ coefficient + outcomes.mean()
        measures[name] = dict(rms=float(np.sqrt(np.mean((fit - observed)**2))), coefficient=coefficient.tolist(),
                              correlation=float(np.corrcoef(fit, observed)[0, 1]))
        priors[name] = multipliers(np.abs(coefficient))
    measures['constant_rms'] = float(np.sqrt(np.mean((observed - y.mean())**2)))
    measures['practice_cues'], measures['held_cues'] = len(y), len(observed)
    return priors, measures


def register(priors, audit):
    donors = [dict(asdict(design), elbow_axis_xy=list(design.elbow_axis_xy)) for design in DESIGNS]
    result = dict(schema='tt24-donor-pong-1', native_tape_sha256=_hash(SOURCE), donor_native_sha256=_hash(NATIVE),
                  modes=list(MODES), seeds=list(SEEDS), donors=donors, budgets=list(BUDGETS),
                  directions=DIRECTIONS, training_launches_per_body=PRACTICE, validation_per_body=VALID,
                  independent_held_per_body=HELD, train_cap=CAP, initial_weights='zero', noise=NOISE,
                  features=[list(g) for g in COMPONENTS], native_gain={n: a.tolist() for n, a in priors.items()},
                  source_held=audit, primary='mean held consecutive rebounds after 1, 2, 4 and 8 reward updates')
    if PLAN.exists():
        if _read(PLAN) != result:
            raise ValueError('registered donor protocol changed')
    else:
        _write(PLAN, result)
    return result


def mutations(rng, basis, index):
    output = np.empty((DIRECTIONS, *SHAPE))
    for slot in range(DIRECTIONS):
        axes = basis[:, (index + slot) % 5]
        sensory = np.zeros(FEATURES)
        if slot < 2:
            sensory[rng.choice(VERTICAL, 3, replace=False)] = rng.normal(size=3)
        elif slot == 2:
            sensory[rng.choice(HORIZONTAL, 3, replace=False)] = rng.normal(size=3)
        elif slot == 3:
            sensory[POSE] = rng.normal(size=len(POSE))
        elif slot == 4:
            sensory[rng.integers(FEATURES)] = rng.choice((-1., 1.))
        else:
            sensory[:] = rng.normal(size=FEATURES) * .35
        direction = np.outer(axes, sensory) + rng.normal(size=SHAPE) * (.12 if slot < 4 else .35)
        output[slot] = direction * rng.choice((.24, .48, 1., 1.5)) * NOISE / (1. + .013 * index)
    return output


def utility(result, cap):
    eligible = result.copy()
    eligible[..., 0] = .55 * result[..., 0] + .45 * result[..., 1]
    return _utility(eligible, cap)


def cohort(worlds, basis, priors, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(2**31 - 2, 3 * (VALID + HELD + BUDGETS[-1] * PRACTICE), replace=False) + 1
    roster = draws[:3 * VALID].reshape(3, VALID)
    held = draws[3 * VALID:3 * (VALID + HELD)].reshape(3, HELD)
    training = draws[3 * (VALID + HELD):].reshape(BUDGETS[-1], 3, PRACTICE)
    valid_batches = [world.prepare(seeds, 'held') for world, seeds in zip(worlds, roster, strict=True)]
    held_batches = [world.prepare(seeds, 'held') for world, seeds in zip(worlds, held, strict=True)]

    def play(options, batches, cap):
        return np.array([world.play(options, batch, cap) for world, batch in zip(worlds, batches, strict=True)])

    random_gain = multipliers(rng.normal(size=len(COMPONENTS)))
    gains = np.array([priors['pong'], priors['shuffled'], np.ones(FEATURES), random_gain])
    best = np.zeros((len(MODES), *SHAPE))
    incumbent = utility(play(best, valid_batches, 300), 300)
    row = dict(seed=seed, valid=roster.tolist(), held=held.tolist(), training=training.tolist(),
               gains=gains.tolist(), curves={'0': play(best, held_batches, 300).tolist()}, updates=[])
    for iteration, seeds in enumerate(training):
        original = mutations(rng, basis, iteration)
        scaled = original[None] * gains[:, None, None, :]
        scaled *= (np.linalg.norm(original, axis=(1, 2))[None] / np.linalg.norm(scaled, axis=(2, 3)))[..., None, None]
        assert np.allclose(np.linalg.norm(scaled, axis=(2, 3)), np.linalg.norm(original, axis=(1, 2)))
        options = np.clip(np.concatenate((best[:, None], best[:, None] + scaled, best[:, None] - scaled), axis=1), -2., 2.)
        batches = [world.prepare(launch, 'held') for world, launch in zip(worlds, seeds, strict=True)]
        results = play(options, batches, CAP)
        fit = utility(results, CAP).reshape(len(MODES), options.shape[1])
        leaders = np.argsort(-fit[:, 1:], axis=1, kind='stable')[:, :3] + 1
        candidates = options[np.arange(len(MODES))[:, None], leaders]
        assessed = play(candidates, valid_batches, 300)
        score = utility(assessed, 300).reshape(len(MODES), -1)
        winning = np.argmax(score, axis=1)
        updated = score[np.arange(len(MODES)), winning] > incumbent
        best[updated] = candidates[np.arange(len(MODES)), winning][updated]
        incumbent[updated] = score[np.arange(len(MODES)), winning][updated]
        row['updates'].append(dict(train_frames=results[..., 2].reshape(3, len(MODES), -1).sum(axis=(0, 2)).astype(int).tolist(),
                                   validation_frames=assessed[..., 2].reshape(3, len(MODES), -1).sum(axis=(0, 2)).astype(int).tolist(),
                                   utility=incumbent.tolist()))
        if iteration + 1 in BUDGETS:
            row['curves'][str(iteration + 1)] = play(best, held_batches, 300).tolist()
            counts = np.array(row['curves'][str(iteration + 1)])[..., 1].mean(axis=(0, 2))
            print('stream', seed, 'update', iteration + 1, 'held', np.round(counts, 3), flush=True)
    row['frozen'] = best.tolist()
    return row


def summary(rows):
    score = np.array([[np.asarray(r['curves'][str(b)])[..., 1].mean(axis=(0, 2)) for b in BUDGETS] for r in rows])
    early = score[:, 1:5].mean(axis=1)
    return dict(means={name: score[:, :, i].mean(axis=0).tolist() for i, name in enumerate(MODES)},
                early_paired={name: (early[:, 0] - early[:, i]).tolist() for i, name in enumerate(MODES) if i},
                early_means=early.mean(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    priors, audit = source_fit()
    register(priors, audit)
    print('native optical travel prediction', audit, flush=True)
    if args.plan_only:
        return
    start = perf_counter()
    native = Native()
    worlds = [World(native, design) for design in DESIGNS]
    try:
        basis, probe_rows = _exploration(worlds)
        receipt = _read(RECEIPT) if RECEIPT.exists() else dict(plan_sha256=_hash(PLAN), basis=basis.tolist(), probes=probe_rows, streams=[])
        if receipt['plan_sha256'] != _hash(PLAN) or not np.array_equal(receipt['basis'], basis):
            raise ValueError('cached study has different source or donor probes')
        for seed in SEEDS[len(receipt['streams']):]:
            result = cohort(worlds, basis, priors, seed)
            receipt['streams'].append(result)
            receipt['summary'] = summary(receipt['streams'])
            receipt['seconds'] = perf_counter() - start
            for label, weights in zip(MODES, result['frozen'], strict=True):
                _checkpoint(ROOT / f'tt24_donor_pong-{seed}-{label}.npz', weights)
            _write(RECEIPT, receipt)
            print('available streams', len(receipt['streams']), receipt['summary'], flush=True)
    finally:
        for world in worlds:
            world.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()

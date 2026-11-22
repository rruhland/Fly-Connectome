"""Continue a native elapsed-time policy on chosen physical SO-101 outcomes."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import CHECK, OPTIONS, WIDTH
from .tt23_anonymous_replication import ARM as ARM_NEW, OFFSETS, TRAIN
from .tt23_anonymous_target import target_context
from .tt23_optical_contact import OUT

ARM_ORIGINAL = OUT / 'anonymous-action-arm.json.gz'
OUTPUT = OUT / 'adaptive-jog.json.gz'
SEED = 197799922
DRAWS = 256
REWARD_PSEUDO_COUNT = 3.
EXPLORATION = .30
CENTER = len(OPTIONS) // 2
OPENING_HZ = 30
PONG_HZ = 120
POINTS = np.array((0, 4, 8, 16, 32, 48, 64))
CLOCK = np.array((0, 15, 30, 60, 90, 120))
EARLY = (1, 2, 3)
MODES = ('pong', 'flat', 'shuffled')
REFLECTION = np.array((-1., -1., 1., 1.))


def kernel(a, b):
    squared = np.sum(((a[:, None] - b[None, :]) / WIDTH)**2, axis=2)
    return np.exp(-.5 * np.minimum(100., squared))


def prepare_source(group, rows_key):
    rows = group[rows_key]
    same_launch = []
    for trials in zip(*(rows[str(i)] for i in range(len(OPTIONS))), strict=True):
        pose = trials[0]['context']
        if any(row['case'] != trials[0]['case'] or row['context'] != pose for row in trials):
            raise ValueError('source options did not face the same native start and prefix')
        if pose is not None:
            same_launch.append([row['ticks'] for row in trials])
    actual = np.log1p(np.asarray(same_launch) / PONG_HZ)
    raw = np.repeat(actual.mean(axis=1), 2)[:, None]
    source = group['model']
    permuted = group['shuffled']
    if source['x'] != permuted['x']:
        raise ValueError('shuffling source labels changed the available context')
    realized = np.asarray(source['value']) + raw
    if not np.allclose(realized[::2], actual, rtol=0., atol=1e-12):
        raise ValueError('stored centered Pong labels differ from native elapsed time')
    observed = [trial for options in rows.values() for trial in options]
    return dict(x=np.asarray(source['x']), pong=realized, shuffled=np.asarray(permuted['value']) + raw,
                launch={row['case']['serve'] for row in observed},
                native_points=len(observed), native_frames=sum(row['ticks'] for row in observed),
                visible_launches=len(same_launch))


def load_source():
    original, independent = _read(CHECK), _read(TRAIN)['sets']
    if [group['offset'] for group in independent] != list(OFFSETS):
        raise ValueError('the four independent source fits must be complete')
    old = prepare_source(original, 'training')
    banks = [prepare_source(source, 'rows') for source in independent]
    unique = set().union(old['launch'], *(b['launch'] for b in banks))
    if len(unique) != sum(len(row['launch']) for row in [old, *banks]):
        raise ValueError('native source launches overlap')
    pooled = {key: np.concatenate([b[key] for b in banks]) for key in ('x', 'pong', 'shuffled')}
    metrics = ('native_points', 'native_frames', 'visible_launches')
    pooled.update({key:sum(b[key] for b in banks) for key in metrics})
    pooled['launch'] = set().union(*(b['launch'] for b in banks))
    return dict(pooled=pooled, original=old)


def load_arm(path):
    receipt = _read(path)
    if (len(receipt['rows']) != 64 or receipt['macro_hash'] != _hash(MACRO)
            or receipt['source_hash'] != _hash(CHECK)):
        raise ValueError('arm bank or its common motor changed')
    pose = MotorPrimitive(_read(MACRO)['parameters'], None).ready_world
    rows = receipt['rows']
    context = [target_context(row['sight'], pose) for row in rows]
    located = np.array([x is not None for x in context])
    grid = np.array([x if x is not None else np.zeros(len(WIDTH)) for x in context])
    if any(len(row['options']) != len(OPTIONS) for row in rows):
        raise ValueError('incomplete seven-action physical bank')
    return dict(seeds=[row['seed'] for row in rows], width=[row['domain'] for row in rows],
                grid=grid, located=located, baseline=np.array([r['choices']['learned'] for r in rows]),
                ticks=np.array([[trial['ticks'] for trial in r['options']] for r in rows]),
                legal=np.array([[trial['legal'] > 0 for trial in r['options']] for r in rows]),
                contacts=np.array([[trial['contacts'] > 0 for trial in r['options']] for r in rows]))


def source_prior(source, query):
    weights = kernel(query, source['x'])
    total = weights.sum(axis=1)[:, None] + .7
    candidates = {name: (weights @ source[name] + .7 * source[name].mean()) / total
                  for name in ('pong', 'shuffled')}
    candidates['flat'] = np.repeat(candidates['pong'].mean(axis=1, keepdims=True), len(OPTIONS), axis=1)
    return candidates


def stream(prior, mix, seen, elapsed, gate, random_option, tie):
    """Update a single causal history using only the selected branch duration."""
    n_train = len(elapsed)
    train_index = np.arange(n_train)
    ratios = np.full_like(prior, REWARD_PSEUDO_COUNT)
    scores = REWARD_PSEUDO_COUNT * prior.copy()
    predicted = np.full((n_train + 1, len(prior) - n_train), CENTER, dtype=np.uint8)
    actions, durations = np.empty(n_train, dtype=np.uint8), np.empty(n_train, dtype=int)
    for step in range(n_train + 1):
        ranking = scores[n_train:] / ratios[n_train:] + tie[n_train:]
        predicted[step, seen[n_train:]] = np.argmax(ranking, axis=1)[seen[n_train:]]
        if step == n_train:
            break
        choice = int(np.argmax(scores[step] / ratios[step] + tie[step]))
        if gate[step] < EXPLORATION:
            choice = random_option[step]
        if not seen[step]:
            choice = CENTER
        length = elapsed[step, choice]
        actions[step], durations[step] = choice, length
        if seen[step]:
            reward = np.log1p(length / OPENING_HZ)
            for side, index in enumerate((choice, len(OPTIONS) - 1 - choice)):
                weight = mix[side, :, step] * .5
                scores[:, index] += weight * reward
                ratios[:, index] += weight
    return actions, durations, predicted


def select_histories(past, future, source, draws):
    x = np.concatenate((past['grid'], future['grid']))
    seen = np.r_[past['located'], future['located']]
    weights = np.stack((kernel(x, past['grid']), kernel(x, REFLECTION * past['grid'])))
    priors = source_prior(source, x)
    modes = (*MODES, 'center')
    outputs = {name:dict(practice=[], held_by_points=[], held_by_seconds=[]) for name in modes}
    for draw in draws:
        rng = np.random.default_rng(draw)
        gate = rng.random(64)
        options = rng.integers(len(OPTIONS), size=64)
        ties = 1e-10 * rng.random((len(x), len(OPTIONS)))
        for name in modes:
            if name == 'center':
                choices = np.full(64, CENTER)
                ticks = past['ticks'][:, CENTER]
                frozen = np.full((65, 64), CENTER)
            else:
                choices, ticks, frozen = stream(priors[name], weights, seen, past['ticks'], gate, options, ties)
            timeline = np.r_[0, np.cumsum(ticks)]
            finished = np.searchsorted(timeline, CLOCK * OPENING_HZ, side='right') - 1
            outputs[name]['practice'].append(dict(choices=choices.tolist(), ticks=ticks.tolist()))
            outputs[name]['held_by_points'].append(frozen[POINTS])
            outputs[name]['held_by_seconds'].append(frozen[finished])
    return outputs


def readouts(outputs, past, future):
    n = len(future['seeds'])
    metrics, early = {}, {}
    for name, result in outputs.items():
        points = np.asarray(result['held_by_points'])
        timed = np.asarray(result['held_by_seconds'])
        held_at_points = future['legal'][np.arange(n), points]
        per_point = past['legal'][np.arange(n), np.array([x['choices'] for x in result['practice']])]
        timelines = np.cumsum(np.array([x['ticks'] for x in result['practice']]), axis=1)
        first = [float(times[np.argmax(success)] / OPENING_HZ) if success.any() else None
                 for success, times in zip(per_point, timelines, strict=True)]
        has_return = [t for t in first if t is not None]
        metrics[name] = dict(
            held_by_points=held_at_points.sum(axis=2).mean(axis=0).tolist(),
            held_by_seconds=future['legal'][np.arange(n), timed].sum(axis=2).mean(axis=0).tolist(),
            contact_by_points=future['contacts'][np.arange(n), points].sum(axis=2).mean(axis=0).tolist(),
            practice_mean_legal=float(per_point.sum(axis=1).mean()),
            practice_mean_seconds=float(timelines[:, -1].mean() / OPENING_HZ),
            observed_seconds_by_points=(np.column_stack((np.zeros(len(timelines)), timelines))[:, POINTS]
                                        .mean(axis=0) / OPENING_HZ).tolist(),
            trials_finished_by_seconds=np.sum(timelines[:, :, None] <= CLOCK * OPENING_HZ, axis=1).mean(axis=0).tolist(),
            practice_any_legal=float(len(has_return) / len(first)),
            median_first_legal_seconds=float(np.median(has_return)) if has_return else None,
            early_held_returns=float(held_at_points[:, EARLY].mean() * n),
        )
        result['held_by_points'] = points.tolist()
        result['held_by_seconds'] = timed.tolist()
        result['practice_first_legal_seconds'] = first
        early[name] = held_at_points[:, EARLY].mean(axis=1)
    return metrics, early


def intervals(early, width):
    rng = np.random.default_rng(SEED + 1)
    count, n = early['pong'].shape
    bootstrap = 4096
    groups = [np.flatnonzero(width == name) for name in np.unique(width)]
    held = np.concatenate([rng.choice(group, size=(bootstrap, len(group)), replace=True) for group in groups], axis=1)
    replicate = rng.multinomial(count, np.full(count, 1. / count), size=bootstrap) / count
    result = {}
    for control in ('flat', 'shuffled', 'center'):
        delta = early['pong'] - early[control]
        distrib = np.take_along_axis(replicate @ delta, held, axis=1).sum(axis=1)
        result[control] = dict(early_diff=float(delta.mean() * n),
                               paired95=np.quantile(distrib, (.025, .975)).tolist())
    return result


def elapsed_diagnostic(bank):
    longest = np.argmax(bank['ticks'], axis=1)
    played = bank['legal'][np.arange(len(longest)), longest]
    credit = bank['legal'].any(axis=1)
    return dict(available=int(credit.sum()), longest_action_legal=int(played.sum()))


def run(output, repetitions=DRAWS):
    began = perf_counter()
    sources = load_source()
    paths = dict(original=ARM_ORIGINAL, replication=ARM_NEW)
    banks = {name: load_arm(path) for name, path in paths.items()}
    original, replication = banks.values()
    if any(len(set(b['seeds'])) != 64 for b in banks.values()) or set(original['seeds']) & set(replication['seeds']):
        raise ValueError('both physical launch banks must be complete and disjoint')
    init = source_prior(sources['original'], original['grid'])['pong'].argmax(axis=1)
    init[~original['located']] = CENTER
    if not np.array_equal(init, original['baseline']):
        raise ValueError('absolute times changed the original frozen source choice')
    draws = np.random.default_rng(SEED).integers(2**31, size=repetitions).tolist()
    pairs, gate = {}, True
    for panel, source in sources.items():
        pairs[panel] = {}
        for held, future in banks.items():
            train = 'replication' if held == 'original' else 'original'
            past = banks[train]
            outcomes = select_histories(past, future, source, draws)
            metrics, early = readouts(outcomes, past, future)
            difference = intervals(early, np.asarray(future['width']))
            passed = all(d['early_diff'] >= 4 and d['paired95'][0] > 0 for d in difference.values())
            if panel == 'pooled':
                gate &= passed
            pairs[panel][held] = dict(practice_bank=train, held_seeds=future['seeds'], practice_seeds=past['seeds'],
                                      metrics=metrics, pong_minus=difference, passed=passed, traces=outcomes)
            print(panel, held, {name: value['held_by_points'] for name, value in metrics.items()},
                  'early differences', difference, flush=True)
    result = dict(schema='tt23-adaptive-jog-1', seed=SEED, draws=draws, points=POINTS.tolist(), seconds=CLOCK.tolist(),
                  source={name: {key: bank[key] for key in ('native_points', 'native_frames', 'visible_launches')}
                          | {'native_seeds': len(bank['launch'])} for name, bank in sources.items()},
                  exploration=EXPLORATION, prior_count=REWARD_PSEUDO_COUNT,
                  hashes={p.name: _hash(p) for p in (*paths.values(), CHECK, TRAIN, MACRO)},
                  elapsed_diagnostic={name: elapsed_diagnostic(b) for name,b in banks.items()},
                  comparison=pairs, full_point_gate=gate, wall_seconds=perf_counter() - began)
    _write(output, result)
    print('full-point gate', gate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--repetitions', type=int, default=DRAWS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('a positive number of paired repetitions is required')
    run(args.output, args.repetitions)


if __name__ == '__main__':
    main()

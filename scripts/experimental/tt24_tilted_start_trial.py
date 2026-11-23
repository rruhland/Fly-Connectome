"""Reward-only correction for broad joint starts learned on three hinged donors."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState
from .tt24_donor_assess import CHECKPOINT as ORIGINAL, ShiftedBounce
from .tt24_donor_trial import (DESIGNS, FEATURES, FUSED_OUTPUT, ROOT, DonorBounce, Native, World,
                               _exploration, _seeds, _summary, load_public)
from .tt24_posture import WidePosture
from .tt4_transfer_trial import _read, _write


SOURCE = ROOT / 'tt24_tilted_start_training.json.gz'
CHECKPOINT = ROOT / 'tt24_tilted_start_checkpoint.npz'
OUTPUT = ROOT / 'tt24_tilted_start_held.json.gz'
DIAGNOSTIC = ROOT / 'tt24_tilted_start_clean_diagnostic.json.gz'
CPP = Path(__file__).with_name('tt24_tilted_start_native.cpp')
SHAPE = (5, FEATURES + 2)
ROUNDS = 30
DIRECTIONS = 8
PRACTICE_CAP = 180
VALIDATION_CAP = 240
MODES = ('warm', 'cold')


class WideDonor(DonorBounce):
    sample_drop = staticmethod(WidePosture.sample_drop)


class WideShifted(ShiftedBounce):
    sample_drop = staticmethod(WidePosture.sample_drop)


class StartWorld(World):
    def __init__(self, native, design, wide):
        self.native = native
        self.arena = (WideDonor if wide else DonorBounce)(design)
        self.handle = native.open(self.arena._model._address)


def evaluate(world, policies, batch, cap):
    states, physics = batch
    weights = np.ascontiguousarray(policies, dtype=float).reshape(-1, *SHAPE)
    result = np.empty((len(weights), len(states), 4))
    world.native.evaluate(world.handle, weights.ctypes.data, len(weights), states.ctypes.data,
                          physics.ctypes.data, len(states), cap, result.ctypes.data)
    return result


def scores(worlds, batches, policies, cap):
    return np.array([evaluate(world, policies, batch, cap)
                     for world, batch in zip(worlds, batches, strict=True)])


def utility(results, cap):
    hits, run, ticks = np.moveaxis(results[..., :3], -1, 0)
    values = .08 * (hits >= 1) + 1.30 * (run >= 2) + .5 * (run >= 4)
    values += .13 * np.maximum(0, run - 2) + .04 * ticks / cap
    bodies = values.mean(axis=-1)
    return .80 * bodies.mean(axis=0) + .20 * bodies.min(axis=0)


def proposal_directions(rng, basis, round_index):
    directions = rng.normal(size=(DIRECTIONS, *SHAPE)) * .045
    for index, direction in enumerate(directions):
        axis = (round_index + index) % 2
        if index == 0:
            direction[:, -2:] += basis[:, 3:] * .8
        elif index < 4:
            candidate = basis[:, 3 + axis]
            if index == 1:
                direction[:, -2 + axis] += candidate
            elif index == 2:
                direction[:, 10 + axis] += candidate
                direction[:, -2 + axis] += candidate * rng.uniform(.25, .85)
            else:
                direction[:, -2:] += np.outer(candidate, rng.normal(size=2))
                direction[:, 10:12] += .5 * np.outer(candidate, rng.normal(size=2))
        elif index < 6:
            direction[:, [10, 11, 20, 21]] += rng.normal(size=(5, 4)) * .42
        else:
            body = basis[:, rng.integers(5)]
            chosen = rng.choice(SHAPE[1], size=5, replace=False)
            direction[:, chosen] += np.outer(body, rng.normal(size=len(chosen)))
        directions[index] *= rng.choice((.22, .40, .62)) / (1. + .010 * round_index)
    return directions


def saved_original():
    with np.load(ORIGINAL, allow_pickle=False) as source:
        return np.pad(source['weights'], ((0, 0), (0, 2)))


def checkpoint(policies):
    pending = CHECKPOINT.with_name(CHECKPOINT.stem + '.tmp.npz')
    np.savez_compressed(pending, **policies)
    pending.replace(CHECKPOINT)


def train(rounds):
    native = Native(CPP)
    wide = [StartWorld(native, body, True) for body in DESIGNS]
    ordinary = [StartWorld(native, body, False) for body in DESIGNS]
    source_rng = np.random.default_rng(818600962)
    wide_seeds = _seeds(789182005, len(wide), 8)
    ordinary_seeds = _seeds(789182006, len(wide), 4)
    wide_dev = [world.prepare(seed, 'held') for world, seed in zip(wide, wide_seeds, strict=True)]
    ordinary_dev = [world.prepare(seed, 'held') for world, seed in zip(ordinary, ordinary_seeds, strict=True)]
    prior = saved_original()
    initial = np.stack([prior, np.zeros_like(prior)])
    confirm = lambda policies: (scores(wide, wide_dev, policies, VALIDATION_CAP),
                                scores(ordinary, ordinary_dev, policies, VALIDATION_CAP))
    combine = lambda wide_score, ordinary_score: (.85 * utility(wide_score, VALIDATION_CAP) +
                                                  .15 * utility(ordinary_score, VALIDATION_CAP))
    try:
        basis, probes = _exploration(wide)
        initial_wide, initial_ordinary = confirm(initial)
        champion = dict(zip(MODES, initial.copy(), strict=True))
        quality = dict(zip(MODES, combine(initial_wide, initial_ordinary).tolist(), strict=True))
        receipt = dict(source_true_geometry=True, source_reward='physical consecutive rebounds, with emphasis on the second',
                       training_bodies=list(range(len(wide))), validation_wide_seeds=wide_seeds,
                       validation_ordinary_seeds=ordinary_seeds, updates=[],
                       original_sha256=sha256(ORIGINAL.read_bytes()).hexdigest(),
                       babble=probes, initial_wide=initial_wide.tolist(), initial_ordinary=initial_ordinary.tolist(),
                       practice_cap=PRACTICE_CAP, validation_cap=VALIDATION_CAP)
        begin = perf_counter()
        for generation in range(rounds):
            directions = proposal_directions(source_rng, basis, generation)
            launches = source_rng.integers(1, 2**31 - 1, (len(wide), 2)).tolist()
            practice = [world.prepare(seed, 'held') for world, seed in zip(wide, launches, strict=True)]
            options = []
            for mode in MODES:
                centre = champion[mode]
                options.extend(np.clip(np.concatenate((centre[None], centre + directions, centre - directions)), -2., 2.))
            options = np.asarray(options)
            sampled = scores(wide, practice, options, PRACTICE_CAP)
            values = utility(sampled, PRACTICE_CAP)
            row_size = 1 + 2*DIRECTIONS
            candidates = []
            for index in range(len(MODES)):
                start = index * row_size
                choices = 1 + np.argsort(-values[start + 1:start + row_size], kind='stable')[:2]
                candidates.extend(start + choices)
            verified_wide, verified_ordinary = confirm(options[candidates])
            retained = combine(verified_wide, verified_ordinary)
            changes = {}
            for index, mode in enumerate(MODES):
                winner = index*2 + int(np.argmax(retained[index*2:index*2+2]))
                if float(retained[winner]) > quality[mode]:
                    quality[mode] = float(retained[winner])
                    champion[mode] = options[candidates[winner]].copy()
                    changes[mode] = int(candidates[winner])
            receipt['updates'].append(dict(generation=generation + 1, practice_seeds=launches,
                                           practice=sampled.tolist(), candidates=options[candidates].tolist(),
                                           candidate_indices=[int(i) for i in candidates],
                                           validation_wide=verified_wide.tolist(),
                                           validation_ordinary=verified_ordinary.tolist(),
                                           accepted=changes, utility=quality.copy()))
            if changes or generation % 5 == 4:
                print('wide donor training', generation + 1, round(perf_counter()-begin, 1),
                      {key: round(val, 3) for key,val in quality.items()}, flush=True)
            if generation % 5 == 4:
                _write(SOURCE, receipt)
        checkpoint(champion)
        receipt['checkpoint_sha256'] = sha256(CHECKPOINT.read_bytes()).hexdigest()
        receipt['selected_before_independent_held'] = True
        _write(SOURCE, receipt)
    finally:
        for world in wide + ordinary:
            world.close()
        native.working.cleanup()


def episode(arena, native, weights, seed, regime, sense, *, tape=None):
    render = sense == 'wrist' and tape is None
    obs = arena.reset(seed=seed, split='held', regime=regime, render=render)
    read = load_public(FUSED_OUTPUT) if render else CleanState(arena) if sense == 'clean' else None
    issued, action = np.zeros(5), np.zeros(5)
    weights = np.ascontiguousarray(weights)
    home = None
    commands = []
    for tick in range(arena.max_steps):
        if read is not None:
            state = read.observe(obs)
            if home is None:
                home = state[9:11].copy()
            sample = np.ascontiguousarray(np.r_[state, home])
            native.command(sample.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
        elif tape is not None:
            action = np.asarray(tape[tick])
        commands.append(action.tolist())
        obs, _, done = arena.step(action, render=render)
        if tape is not None and done != (tick + 1 == len(tape)):
            raise ValueError(f'raw tape changed the terminal tick for {seed}')
        if done:
            break
    return dict(metrics=arena.judge_metrics(), actions=commands)


def matched(arena, native, weights, seed, regime, sense):
    result = episode(arena, native, weights, seed, regime, sense)
    replay = episode(arena, native, weights, seed, regime, 'replay', tape=result['actions'])
    if digest_json(result) != digest_json(replay):
        raise ValueError(f'raw physical evidence changed for {seed}')
    return result


def validation(native, policies, report):
    cohorts = []
    prohibited = {int(seed) for generation in report['updates'] for group in generation['practice_seeds'] for seed in group}
    prohibited.update(seed for key in ('validation_ordinary_seeds', 'validation_wide_seeds')
                      for group in report[key] for seed in group)
    matrices = np.stack(list(policies.values()))
    for index, design in enumerate(DESIGNS):
        world = StartWorld(native, design, True)
        try:
            seeds = _seeds(190254422 + index, 1, 16)[0]
            if prohibited.intersection(seeds):
                raise ValueError('held donor seed appeared in training or validation')
            rows = evaluate(world, matrices, world.prepare(seeds, 'held'), 300)
            audit = []
            for tag, matrix, results in zip(policies, matrices, rows, strict=True):
                for i in (0, 1):
                    result = matched(world.arena, native, matrix, seeds[i], 'changed', 'clean')
                    got = result['metrics']
                    if list(results[i]) != [got['rebounds'], got['longest_consecutive_rebounds'],
                                            got['steps'], got['paddle_contacts']]:
                        raise ValueError(f'native physical output mismatch at donor {index} policy {tag}')
                    audit.append(dict(mode=tag, seed=seeds[i], **result))
            cohorts.append(dict(body=index, seeds=seeds, scores=dict(zip(policies, rows.tolist(), strict=True)),
                                exact_native_vs_python=audit))
        finally:
            world.close()
    return cohorts


def conclusions(rows):
    stats = _summary(rows)
    for label, summary in stats.items():
        tallies = [r['conditions'][label]['metrics']['rebounds'] for r in rows]
        summary['first'] = sum(x >= 1 for x in tallies)
        summary['second'] = sum(x >= 2 for x in tallies)
    return stats


def held():
    receipt = _read(SOURCE)
    if receipt['checkpoint_sha256'] != sha256(CHECKPOINT.read_bytes()).hexdigest():
        raise ValueError('checkpoint changed after independent testing was sealed')
    with np.load(CHECKPOINT, allow_pickle=False) as sources:
        policies = {name: sources[name].copy() for name in MODES}
    policies.update(original=saved_original(), stationary=np.zeros(SHAPE))
    native = Native(CPP)
    held_starts = _seeds(1657507381, 3, 16)
    setup = (('wide_stock', lambda: WidePosture(view='wrist', width=192, height=144), 'standard'),
             ('wide_altered', WideShifted, 'changed'),
             ('ordinary_stock', lambda: BounceArena(view='wrist', width=192, height=144), 'standard'))
    output = dict(checkpoint_sha256=receipt['checkpoint_sha256'], no_target_practice=True,
                  held_donors=validation(native, policies, receipt), targets={})
    _write(OUTPUT, output)
    try:
        for spec, seeds in zip(setup, held_starts, strict=True):
            label, factory, regime = spec
            arena = factory()
            cases = []
            try:
                for seed in seeds:
                    rows = {name: matched(arena, native, model, seed, regime,
                                          'none' if name == 'stationary' else 'wrist')
                            for name, model in policies.items()}
                    cases.append(dict(seed=seed, conditions=rows))
                    print(label, seed, [(name, row['metrics']['longest_consecutive_rebounds'])
                                        for name, row in rows.items()], flush=True)
                output['targets'][label] = dict(cases=cases, summary=conclusions(cases),
                                                exact_replays=len(cases)*len(policies))
                _write(OUTPUT, output)
            finally:
                arena.close()
    finally:
        native.working.cleanup()


def clean_diagnostic():
    """Separate unseen-body control from camera error after freezing both actors."""
    receipt = _read(SOURCE)
    if receipt['checkpoint_sha256'] != sha256(CHECKPOINT.read_bytes()).hexdigest():
        raise ValueError('diagnostic checkpoint differs from the independently tested weights')
    with np.load(CHECKPOINT, allow_pickle=False) as sources:
        policies = {name: sources[name].copy() for name in MODES}
    policies['original'] = saved_original()
    seeds = _seeds(1657507381, 3, 16)
    native = Native(CPP)
    report = dict(privileged_observations=True, checkpoint_sha256=receipt['checkpoint_sha256'], records={})
    try:
        for label, arena, regime, starts in (
                ('wide_stock', WidePosture(view='none'), 'standard', seeds[0]),
                ('wide_altered', WideShifted(), 'changed', seeds[1])):
            try:
                cases = [dict(seed=seed, conditions={mode: matched(arena, native, matrix, seed, regime, 'clean')
                                                     for mode, matrix in policies.items()})
                         for seed in starts]
                report['records'][label] = dict(cases=cases, summary=conclusions(cases))
                print('held privileged comparator', label,
                      {mode: (row['mean'], row['second'], row['maximum'])
                       for mode, row in report['records'][label]['summary'].items()}, flush=True)
            finally:
                arena.close()
        _write(DIAGNOSTIC, report)
    finally:
        native.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--held', action='store_true')
    modes.add_argument('--clean-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.held:
        held()
    elif args.clean_diagnostic:
        clean_diagnostic()
    else:
        train(ROUNDS)
        held()


if __name__ == '__main__':
    main()

"""Practice physical bouncing with rebound-score selection and matched exposure."""

import argparse
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_donor_trial import Native, STATE
from .tt24_tilted_start_trial import CPP, episode, evaluate, matched, proposal_directions, saved_original
from .tt26_bounce_effects import OUTPUT
from .tt26_bounce_practice import CAP, INITIAL_WEIGHTS, VARIATIONS, VariedBounce


CHECKPOINT = OUTPUT / 'bounce-policy.npz'
RECEIPT = OUTPUT / 'bounce-practice.json.gz'
PRACTICE_SEED = 261810400
VALIDATION_SEED = 261840400
HELD_SEED = 261900400
WRIST_SEED = 261950400
PRACTICE_CAP = 160
MODES = ('varied', 'ordinary')


class PracticeWorld:
    def __init__(self, native, arena):
        self.native = native
        self.arena = arena
        self.handle = native.open(arena._model._address)

    def prepare(self, seeds, split):
        model = self.arena._model
        states = np.empty((len(seeds), mujoco.mj_stateSize(model, STATE)))
        physics = np.empty((len(seeds), 2))
        for i, seed in enumerate(seeds):
            self.arena.reset(seed=seed, split=split, render=False)
            mujoco.mj_getState(model, self.arena._data, states[i], STATE)
            physics[i] = self.arena._drag_force_scale, self.arena._drop.damping
        return states, physics

    def close(self):
        self.native.close(self.handle)
        self.arena.close()


def scores(worlds, batches, policies, cap):
    return np.stack([evaluate(world, policies, batch, cap) for world, batch in zip(worlds, batches, strict=True)])


def utility(counts, cap):
    hits, streak, ticks, _ = np.moveaxis(counts, -1, 0)
    by_profile = (.65 * streak + .35 * hits + .10 * ticks / cap).mean(axis=-1)
    return .85 * by_profile.mean(axis=0) + .15 * by_profile.min(axis=0)


def seed_rows(base, count, shift=0):
    return [[base + shift + 100 * profile + i for i in range(count)] for profile in range(len(VARIATIONS))]


def train(worlds, initial, rounds, report):
    rng = np.random.default_rng(261801311)
    dev_seeds = seed_rows(VALIDATION_SEED, 7)
    validation = {mode: [world.prepare(seeds, 'practice') for world, seeds in zip(group, dev_seeds, strict=True)]
                  for mode, group in worlds.items()}
    champions = {mode: initial.copy() for mode in MODES}
    prior = {mode: scores(worlds[mode], validation[mode], initial[None], CAP) for mode in MODES}
    best = {mode: float(utility(result, CAP)[0]) for mode, result in prior.items()}
    report.update(validation_seeds=dev_seeds, initial_validation={mode: result.tolist() for mode, result in prior.items()},
                  training_updates=[])
    for generation in range(rounds):
        directions = proposal_directions(rng, np.eye(5), generation)
        seeds = seed_rows(PRACTICE_SEED, 2, shift=2 * generation)
        results = {}
        for mode in MODES:
            world = worlds[mode]
            center = champions[mode]
            proposals = np.clip(np.concatenate((center[None], center + directions, center - directions)), -2., 2.)
            practice = [plant.prepare(row, 'practice') for plant, row in zip(world, seeds, strict=True)]
            played = scores(world, practice, proposals, PRACTICE_CAP)
            candidates = 1 + np.argsort(-utility(played, PRACTICE_CAP)[1:], kind='stable')[:2]
            check = scores(world, validation[mode], proposals[candidates], CAP)
            quality = utility(check, CAP)
            winner = int(np.argmax(quality))
            accepted = float(quality[winner]) > best[mode]
            if accepted:
                champions[mode] = proposals[candidates[winner]].copy()
                best[mode] = float(quality[winner])
            results[mode] = dict(practice=played.tolist(), candidates=candidates.tolist(),
                                 validation=check.tolist(), accepted=accepted, best_score=best[mode])
        report['training_updates'].append(dict(launches=seeds, changes=directions.tolist(), outcomes=results))
        print('Score practice', generation + 1, best, flush=True)
    return champions


def physical_audit(worlds, native, initial):
    comparisons = []
    for mode, group in worlds.items():
        for index, world in enumerate(group):
            seed = VALIDATION_SEED + 100 * index + 93
            result = evaluate(world, initial, world.prepare([seed], 'held'), CAP)[0, 0]
            evidence = episode(world.arena, native, initial, seed, 'standard', 'clean')
            metrics = evidence['metrics']
            expected = [metrics[k] for k in ('rebounds', 'longest_consecutive_rebounds', 'steps', 'paddle_contacts')]
            if not np.array_equal(result, expected):
                raise ValueError(f'{mode} native and Python physical outcomes differ: {result}, {expected}')
            comparisons.append(dict(mode=mode, seed=seed, native=result.tolist(), reference=expected))
    return comparisons


def held_summary(cases):
    summary = {}
    for mode in cases[0]['outcomes']:
        rows = [case['outcomes'][mode]['metrics'] for case in cases]
        streak = np.asarray([row['longest_consecutive_rebounds'] for row in rows])
        before = np.asarray([case['outcomes']['before']['metrics']['longest_consecutive_rebounds'] for case in cases])
        summary[mode] = dict(mean_consecutive=float(streak.mean()), longest=int(streak.max()),
                             at_least_two=int(np.sum(streak >= 2)), at_least_ten=int(np.sum(streak >= 10)),
                             rebounds=sum(row['rebounds'] for row in rows),
                             better_than_before=int(np.sum(streak > before)), worse_than_before=int(np.sum(streak < before)))
    return summary


def assess(native, policies, seed_base, count, sense, *, varied):
    cases = []
    rows = seed_rows(seed_base, count)
    for profile, variation in enumerate(VARIATIONS):
        options = dict(view='wrist' if sense == 'wrist' else 'none', width=192, height=144, max_steps=CAP)
        arena = VariedBounce(variation, **options) if varied else BounceArena(**options)
        try:
            for seed in rows[profile]:
                outcomes = {}
                for mode, weights in policies.items():
                    outcomes[mode] = matched(arena, native, weights, seed, 'standard', sense)
                cases.append(dict(profile=profile, seed=seed, outcomes=outcomes))
        finally:
            arena.close()
    return dict(assessed_drops=len(cases), replays=len(cases) * len(policies),
                summary=held_summary(cases), cases=cases)


def run(rounds, wrist):
    began = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    native = Native(CPP)
    worlds = {'varied': [PracticeWorld(native, VariedBounce(variation, view='none', max_steps=CAP))
                         for variation in VARIATIONS],
              'ordinary': [PracticeWorld(native, BounceArena(view='none', max_steps=CAP)) for _ in VARIATIONS]}
    initial = saved_original()
    report = dict(schema='tt26-bounce-score-practice-1', initial_weights_sha256=_hash(INITIAL_WEIGHTS),
                  varied_conditions=[asdict(row) for row in VARIATIONS], native_source_sha256=_hash(CPP),
                  source_controller_observations='metric ball motion and paddle pose',
                  weight_update='antithetic joint-rate readout proposals selected by physical rebound score',
                  training_exposure='identical attempted drops and horizon per branch on each round',
                  practice_cap=PRACTICE_CAP, validation_cap=CAP, jaw_closed=True)
    try:
        report['native_audit'] = physical_audit(worlds, native, initial)
        policies = dict(before=initial, **train(worlds, initial, rounds, report))
        np.savez_compressed(CHECKPOINT, **policies)
        report.update(frozen_before_held=True, checkpoint_sha256=_hash(CHECKPOINT),
                      practice_and_validation={(mode): dict(
                          **{f'{key}_rollouts': sum(np.prod(np.shape(row['outcomes'][mode][key])[:-1]).item()
                                                   for row in report['training_updates']) for key in ('practice', 'validation')},
                          **{f'{key}_steps': sum(int(np.asarray(row['outcomes'][mode][key])[..., 2].sum())
                                                 for row in report['training_updates']) for key in ('practice', 'validation')})
                                               for mode in MODES})
        _write(RECEIPT, report)
        settings = dict(policies, stationary=np.zeros_like(initial))
        for name, seed in (('varied', HELD_SEED), ('ordinary', HELD_SEED + 1000)):
            report[f'held_{name}'] = assess(native, settings, seed, 8, 'clean', varied=name == 'varied')
            print('Held', name, report[f'held_{name}']['summary'], flush=True)
            _write(RECEIPT, report)
        if wrist:
            report['held_wrist'] = assess(native, policies, WRIST_SEED, 4, 'wrist', varied=True)
            print('Held wrist', report['held_wrist']['summary'], flush=True)
        report['seconds'] = perf_counter() - began
        _write(RECEIPT, report)
    finally:
        for group in worlds.values():
            for world in group:
                world.close()
        native.working.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--wrist', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.rounds <= 30:
        parser.error('rounds must be between 1 and 30')
    run(args.rounds, args.wrist)

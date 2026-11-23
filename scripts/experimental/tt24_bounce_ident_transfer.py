"""Measure bounded wrist-calibration transfer across changed SO-101 hardware."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tt19_so101 import WRIST_CAMERA
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import (CalibrationState, CleanState, DirectPupil, MotionFit, WristDecoder,
                                WristPupil, WristState, _save_pupil, _unit_camera_ray, episode, pack,
                                run_deployed, select_pair, summarize)


DEMOS = tuple(range(248_001, 248_005))
TARGET_HELD = tuple(range(248_201, 248_217))
SOURCE_RETURN = tuple(range(248_301, 248_305))
TARGET_REGIME = 'changed'


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _focal(arena):
    return arena.height / (2. * math.tan(math.radians(arena._model.camera(WRIST_CAMERA).fovy[0]) / 2))


def _supervised_residual(source, records, focal):
    """Update the source observation map around its fitted source coefficients."""
    fitted = deepcopy(source)
    fitted.focal = focal
    joints, body, ball, ellipses, _ = zip(*(row for group in records for row in group), strict=True)
    body_forms = fitted._body_forms(joints)
    # Favor a constant and linear local adjustment for the changed mounting.
    prior = np.r_[.003, [.035] * 5, [.4] * 15, [2.] * 35]
    residual = np.asarray(body) - body_forms @ fitted.body_weights
    fitted.body_weights += np.linalg.solve(body_forms.T @ body_forms + np.diag(prior), body_forms.T @ residual)
    valid = np.array([ellipse is not None for ellipse in ellipses])
    rays = _unit_camera_ray([ellipse for ellipse in ellipses if ellipse is not None], focal)
    camera_forms = fitted._camera_forms(rays)
    residual = np.asarray(ball)[valid] - camera_forms @ fitted.camera_weights
    precision = camera_forms.T @ camera_forms + .8 * np.eye(camera_forms.shape[1])
    fitted.camera_weights += np.linalg.solve(precision, camera_forms.T @ residual)
    fitted.default = np.mean([sequence[0][-1][:3] for sequence in records], axis=0)
    return fitted


def _train_target_actor(pairs):
    folds = pairs
    if len(pairs) == 1:
        # One example is split into contiguous blocks for selecting a feature pair.
        folds = list(zip(*(np.array_split(channel, 4) for channel in pairs[0]), strict=True))
    pair, rank = select_pair(folds)
    return DirectPupil(*pack(pairs), pair=pair), rank


def _capture(arena):
    paired, records, results = [], [], []
    for seed in DEMOS:
        seq = []
        result, samples = episode(arena, seed, teacher_fraction=1., collect=True, regime=TARGET_REGIME,
                                  sensor_factory=lambda env: CalibrationState(env, seq))
        results.append({key: value for key, value in result.items() if key != 'commands'})
        paired.append(samples)
        records.append(seq)
        print('demonstration', seed, result['longest_consecutive_rebounds'], len(seq), flush=True)
    return results, paired, records


def _evaluate(arena, seeds, actor, decoder=None, motion=None, *, regime=TARGET_REGIME):
    sensor = CleanState if decoder is None else lambda _: WristState(decoder, motion)
    rows = [episode(arena, seed, actor=actor, sensor_factory=sensor, regime=regime)[0] for seed in seeds]
    return dict(summary=summarize(rows), episodes=rows)


def _verify_tape(arena, row, regime):
    arena.reset(seed=row['seed'], split='held', regime=regime, render=False)
    for command in row['commands']:
        _, _, done = arena.step(command, render=False)
    assert done
    expected = {key: value for key, value in row.items() if key != 'commands'}
    assert arena.judge_metrics() == expected


def _write(path, data):
    contents = json.dumps(data, separators=(',', ':'))
    with gzip.open(path, 'wt', encoding='utf-8') as output:
        output.write(contents)


def fit_target_with_source_pair(arena, source_pair, pairs, decoder, motion):
    """Keep the pair choice as a prior while fitting every coefficient afresh."""
    source_pair = tuple(map(int, source_pair))
    actor = DirectPupil(*pack(pairs), pair=source_pair)
    result = _evaluate(arena, TARGET_HELD, actor, decoder, motion)
    top = max(result['episodes'], key=lambda row: row['longest_consecutive_rebounds'])
    _verify_tape(arena, top, TARGET_REGIME)
    result['exact_replay_seed'] = top['seed']
    result['selected_pair'] = source_pair
    result['adaptation_demo_count'] = len(pairs)
    result['adaptation_frames'] = sum(len(sequence[0]) for sequence in pairs)
    return result


def verify_exports(folder, source_path, report, arena, budget):
    checkpoints = (('map', f'residual_map_fixed_readout_{budget}'), ('target', f'full_target_only_{budget}'))
    evidence = {}
    for name, cohort in checkpoints:
        model = folder / f'bounce-ident-transfer-{name}.npz'
        row = max(report['cohorts'][cohort]['episodes'], key=lambda value: value['longest_consecutive_rebounds'])
        live = run_deployed(arena, row['seed'], model, regime=TARGET_REGIME)
        assert json.loads(json.dumps(live)) == json.loads(json.dumps(row))
        evidence[name] = dict(seed=live['seed'], consecutive=live['longest_consecutive_rebounds'],
                              decisions=len(live['commands']), exact=True)
    with np.load(source_path, allow_pickle=False) as stock, np.load(folder / 'bounce-ident-transfer-map.npz',
                                                                  allow_pickle=False) as target:
        same = all(np.array_equal(stock[key], target[key]) for key in stock.files if key.startswith('actor_'))
    assert same
    evidence['source_readout_identical_in_map_checkpoint'] = same
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--source', type=Path, default=Path('artifacts/tt24/bounce-ident-pupil.npz'))
    parser.add_argument('--budgets', nargs='+', type=int, default=(1, 4), choices=(1, 2, 4))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_hash = _digest(args.source)
    source = WristPupil.load(args.source)
    source_arena = BounceArena(view='wrist', width=192, height=144)
    target_arena = BounceArena(view='wrist', width=192, height=144, arm='alternate')
    target_clean = BounceArena(view='none', arm='alternate')
    report = dict(source_checkpoint_sha256=source_hash, demo_seeds=DEMOS, held_seeds=TARGET_HELD,
                  source_return_seeds=SOURCE_RETURN, arm='alternate', regime=TARGET_REGIME,
                  calibration='privileged training supervision', runtime='wrist RGB and own joint angles', cohorts={})
    cohorts = report['cohorts']
    try:
        source_before = _evaluate(source_arena, SOURCE_RETURN, source.actor, source.sensor.decoder, source.sensor.motion,
                                  regime='standard')
        demos, pairs, records = _capture(target_arena)
        report['demonstrations'] = demos
        cohorts['stationary'] = _evaluate(target_clean, TARGET_HELD, None)
        cohorts['frozen'] = _evaluate(target_arena, TARGET_HELD, source.actor, source.sensor.decoder, source.sensor.motion)
        cohorts['clean_frozen_readout'] = _evaluate(target_clean, TARGET_HELD, source.actor)
        for name, value in cohorts.items():
            print(name, value['summary'], flush=True)
        focal = _focal(target_arena)
        for budget in args.budgets:
            target_data = records[:budget]
            decoder = WristDecoder(target_data, focal)
            motion = MotionFit(decoder, target_data, quadratic=False)
            refit, ranks = _train_target_actor(pairs[:budget])
            residual = _supervised_residual(source.sensor.decoder, target_data, focal)
            residual_motion = MotionFit(residual, target_data, quadratic=False)
            candidates = (
                ('new_map_fixed_readout', source.actor, decoder, motion),
                ('residual_map_fixed_readout', source.actor, residual, residual_motion),
                ('full_target_only', refit, decoder, motion),
            )
            for label, readout, percept, velocity in candidates:
                name = f'{label}_{budget}'
                result = _evaluate(target_arena, TARGET_HELD, readout, percept, velocity)
                cohorts[name] = result
                top = max(result['episodes'], key=lambda row: row['longest_consecutive_rebounds'])
                _verify_tape(target_arena, top, TARGET_REGIME)
                result['exact_replay_seed'] = top['seed']
                result['adaptation_demo_count'] = budget
                result['adaptation_frames'] = sum(len(seq) for seq in target_data)
                print(name, result['summary'], flush=True)
            cohorts[f'full_target_only_{budget}']['selected_pair'] = refit.pair
            cohorts[f'full_target_only_{budget}']['held_demo_pair_errors'] = ranks
            if budget == max(args.budgets):
                _save_pupil(args.out / 'bounce-ident-transfer-target.npz', refit, decoder, motion)
                _save_pupil(args.out / 'bounce-ident-transfer-map.npz', source.actor, residual, residual_motion)
                label = f'target_only_fitted_coefficients_source_pair_{budget}'
                cohorts[label] = fit_target_with_source_pair(target_arena, source.actor.pair, pairs[:budget], decoder, motion)
                print(label, cohorts[label]['summary'], flush=True)

        restored = WristPupil.load(args.source)
        source_after = _evaluate(source_arena, SOURCE_RETURN, restored.actor, restored.sensor.decoder,
                                 restored.sensor.motion, regime='standard')
        assert source_before == source_after and _digest(args.source) == source_hash
        report['source_return'] = dict(exact=True, before=source_before, after=source_after)
        print('source exact return', source_after['summary'], flush=True)
        report['checkpoint_replay'] = verify_exports(args.out, args.source, report, target_arena, max(args.budgets))
        _write(args.out / 'bounce-ident-transfer.json.gz', report)
    finally:
        source_arena.close()
        target_arena.close()
        target_clean.close()


if __name__ == '__main__':
    main()

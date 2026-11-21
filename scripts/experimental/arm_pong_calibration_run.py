"""Compare optical shot calibration at original arm speed on matched native matches."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from arm_plays_pong import MatchCase, NATIVE_HZ, run_match
from arm_pong_calibration import ShotCalibrationController
from button_pong_skill import DesiredPongPolicy
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor


ARM_SEED = 147_884_320
ARM_TICKS = 80
DEV_ENTROPY = 758_291_688
HELD_OUT_ENTROPY = 638_054_901
VARIANTS = {
    'fixed_083': dict(cap=.083, adaptive=False),
    'optical_091': dict(cap=.091),
    'optical_100': dict(cap=.100),
    'guard_091': dict(cap=.091, guard=True),
}


def schedule(count, development):
    if count < 2 or count % 2:
        raise ValueError('match count must be an even integer of at least two')
    rng = np.random.default_rng(DEV_ENTROPY if development else HELD_OUT_ENTROPY)
    low, high = (1_710_000_000, 1_860_000_000) if development else (1_870_000_000, 2_110_000_000)
    games = rng.choice(high - low, size=count, replace=False) + low
    panel_seeds = rng.integers(0, 2**32, size=count)
    controller_seeds = rng.integers(0, 2**32, size=count)
    wiring = rng.permutation(np.resize((0, 3), count))
    return [MatchCase(*map(int, case)) for case in zip(games, panel_seeds, wiring, controller_seeds, strict=True)]


def _play(job):
    condition, variant, case, source, memory, first_to, max_ticks = job
    policy = DesiredPongPolicy(learn=True, exploration=.10) if condition == 'blank_calibrated' else (
        DesiredPongPolicy.load(source, learn=True, exploration=.10))
    motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=memory)
    controller = None
    if variant is not None:
        controller = ShotCalibrationController(policy, motor, case.controller_seed, **VARIANTS[variant])
    result = run_match(case, policy, motor, first_to=first_to, max_ticks=max_ticks, opponent_speed=.35,
                       panel_kwargs={'gear': SOURCE_GEAR}, controller=controller)
    if controller is not None:
        result['calibration'] = controller.report()
    return condition, result


def _summary(matches):
    counts = ('points_won', 'points_lost', 'scored_after_own_return', 'own_returns', 'own_opportunities')
    return {name: {**{key: sum(row[key] for row in rows) for key in counts},
                   'wins': sum(row['winner'] == 'own' for row in rows),
                   'losses': sum(row['winner'] == 'opponent' for row in rows),
                   'caps': sum(row['winner'] is None for row in rows),
                   'adjusted_frames': sum(row.get('calibration', {}).get('adjusted_frames', 0) for row in rows),
                   'optical_crossings': sum(len(row.get('calibration', {}).get('crossings', ())) for row in rows)}
            for name, rows in matches.items()}


def _paired(matches, active, control):
    def measures(row):
        return (row['points_won'] - row['points_lost'], row['winner'] == 'own', row['scored_after_own_return'])

    differences = np.array(list(map(measures, matches[active]))) - np.array(list(map(measures, matches[control])))
    draws = np.random.default_rng(731_055_167).integers(len(differences), size=(5000, len(differences)))
    intervals = np.quantile(differences[draws].mean(axis=1), (.025, .975), axis=0)
    return {name: {'mean_difference': float(differences[:, index].mean()),
                   'match_bootstrap_ci95': intervals[:, index].tolist(),
                   'positive_matches': int((differences[:, index] > 0).sum()),
                   'negative_matches': int((differences[:, index] < 0).sum())}
            for index, name in enumerate(('point_margin', 'first_to_win', 'goals_after_return'))}


def run(source, *, development, selected, count, workers, first_to=3, seconds=20):
    cases = schedule(count, development)
    memory, practice = pretrain_motor(ARM_SEED, ticks=ARM_TICKS, gear=SOURCE_GEAR)
    variants = dict(full=None)
    if development:
        variants.update({name: name for name in VARIANTS})
    else:
        variants.update(full_calibrated=selected, blank_calibrated=selected)
    matches = {name: [] for name in variants}
    max_ticks = seconds * NATIVE_HZ
    jobs = [(name, variant, case, source, memory, first_to, max_ticks)
            for case in cases for name, variant in variants.items()]
    source_files = (Path(__file__), Path(__file__).with_name('arm_pong_calibration.py'),
                    Path(__file__).with_name('arm_plays_pong.py'), Path(__file__).with_name('pong_button_panel.py'),
                    Path(__file__).with_name('button_pong_skill.py'), source)
    hashes = {file.name: sha256(file.read_bytes()).hexdigest() for file in source_files}
    with ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for ordinal, (name, row) in enumerate(pool.map(_play, jobs), 1):
            matches[name].append(row)
            if ordinal % len(variants) == 0:
                print(f'Calibration matches {ordinal}/{len(jobs)}', flush=True)
    if hashes != {file.name: sha256(file.read_bytes()).hexdigest() for file in source_files}:
        raise RuntimeError('experiment files changed during the evaluation')
    comparisons = {name + '_vs_full': _paired(matches, name, 'full') for name in variants if name != 'full'}
    if not development:
        comparisons['full_calibrated_vs_blank_calibrated'] = _paired(matches, 'full_calibrated', 'blank_calibrated')
    return dict(development=development, first_to=first_to, max_ticks=max_ticks, opponent_speed=.35,
                gearing=SOURCE_GEAR, matches_per_condition=count, variants=variants, variant_settings=VARIANTS,
                arm_practice_seed=ARM_SEED, arm_practice=practice, arm_memory=memory, source_sha256=hashes,
                cases=[asdict(case) for case in cases], summary=_summary(matches), paired=comparisons, matches=matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--selected', choices=VARIANTS)
    parser.add_argument('--matches', type=int)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not args.development and args.selected is None:
        parser.error('the held-out run requires a frozen --selected variant')
    count = args.matches if args.matches is not None else 6 if args.development else 16
    result = run(args.checkpoint, development=args.development, selected=args.selected, count=count,
                 workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(summary=result['summary'], paired=result['paired']), indent=2), flush=True)


if __name__ == '__main__':
    main()

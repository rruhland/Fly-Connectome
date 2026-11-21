"""Compare separately practiced arm and Pong skills on matched native matches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from arm_plays_pong import MatchCase, NATIVE_HZ, run_match
from button_pong_skill import DesiredPongPolicy, NaivePongPolicy
from pong_button_panel import FAST_GEAR, SOURCE_GEAR, VisualButtonMotor, pretrain_motor


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_ENTROPY = 13_073_177_693
PILOT_ENTROPY = 928_480_551
ARM_SOURCE_SEED = 147_884_320
ARM_SOURCE_TICKS = 80
CONDITIONS = ('both_live', 'arm_only_live', 'pong_only_live', 'fresh_live', 'both_frozen',
              'interception_only', 'reactive', 'disconnected', 'stationary')
PAIRS = (('both_live', 'arm_only_live'), ('both_live', 'pong_only_live'),
         ('pong_only_live', 'fresh_live'), ('arm_only_live', 'fresh_live'),
         ('both_frozen', 'arm_only_live'), ('both_live', 'both_frozen'),
         ('interception_only', 'reactive'), ('both_frozen', 'interception_only'),
         ('both_live', 'reactive'), ('both_live', 'disconnected'), ('both_live', 'stationary'))
SOURCE_FILES = (
    'scripts/experimental/run_arm_pong_matches.py', 'scripts/experimental/arm_plays_pong.py',
    'scripts/experimental/pong_button_panel.py', 'scripts/experimental/planar_position_arm.py',
    'scripts/experimental/camera_reaching_arena.py', 'scripts/experimental/camera_motor_learner.py',
    'scripts/experimental/button_pong_skill.py', 'scripts/experimental/pong_score_placement.py',
    'scripts/experimental/return_metrics.py', 'src/fly_connectome/pong.py',
)


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def matched_cases(count, *, pilot):
    rng = np.random.default_rng(PILOT_ENTROPY if pilot else QUALIFICATION_ENTROPY)
    low, high = (800_000_000, 900_000_000) if pilot else (1_200_000_000, 1_700_000_000)
    pong = rng.choice(high - low, size=count, replace=False) + low
    panel = rng.integers(1, 2**32, size=count)
    controller = rng.integers(1, 2**32, size=count)
    # These two wiring seeds produce opposite directions. Randomize which
    # matched scene gets each, with no relationship to its native serve seed.
    wiring = rng.permutation(np.resize((0, 3), count))
    return [MatchCase(*map(int, row)) for row in zip(pong, panel, wiring, controller, strict=True)]


def _opening_needs_motion(row):
    if row['first_incoming'] is None:
        return False
    incoming = row['first_incoming']
    y = incoming['pre_ball'][1] + incoming['pre_velocity'][1] / NATIVE_HZ
    y = max(y, .025 - y)
    y = min(y, 1.975 - y)
    return abs(y - .5) > .1125


def _run_one(job):
    ordinal, condition, case, checkpoint, arm_memory, gear, first_to, max_ticks, speed = job
    trained_pong = condition in ('both_live', 'pong_only_live', 'both_frozen', 'disconnected', 'stationary')
    practiced_arm = condition not in ('pong_only_live', 'fresh_live')
    frozen = condition in ('both_frozen', 'disconnected', 'stationary')
    reactively = condition == 'reactive'
    if reactively:
        policy = NaivePongPolicy()
    elif condition == 'interception_only':
        policy = DesiredPongPolicy.load(checkpoint.with_name('source-interception.json'), learn=False,
                                        exploration=0., aim=False)
    elif trained_pong:
        policy = DesiredPongPolicy.load(checkpoint, learn=not frozen, exploration=.10 if not frozen else 0.)
    else:
        policy = DesiredPongPolicy(learn=True, exploration=.10)
    motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=arm_memory if practiced_arm else None)
    row = run_match(case, policy, motor, first_to=first_to, max_ticks=max_ticks, opponent_speed=speed,
                    panel_kwargs={'gear': gear}, contact_enabled=condition != 'disconnected',
                    stationary=condition == 'stationary')
    row['initial_approach_needs_motion'] = _opening_needs_motion(row)
    row['first_correct_binding_tick'] = next((event['tick'] for event in row['binding_events']
                                              if event['down_button'] == row['initial_down_button']), None)
    row['first_press_tick'] = next((event['tick'] for event in row['contacts']
                                   if event['button'] is not None), None)
    row['last_binding_correct'] = bool(row['binding_events'] and
                                       row['binding_events'][-1]['down_button'] == row['initial_down_button'])
    if condition in ('disconnected', 'stationary') and row['physical_motion_ticks']:
        raise AssertionError('a control generated a game drive')
    if condition == 'disconnected' and row['binding_events']:
        raise AssertionError('a stationary optical paddle was assigned a learned direction')
    return ordinal, condition, row


def _metrics(row):
    first = row['first_incoming']
    first_present = int(first is not None)
    first_return = int(first is not None and first['success'])
    by_later_point = row['points'][1:]
    after_first = row['own_returns'] - first_return
    opportunities_after_first = row['own_opportunities'] - first_present
    needs_motion = int(row['initial_approach_needs_motion'])
    binding_before_first = (row['first_correct_binding_tick'] is not None and first is not None
                            and row['first_correct_binding_tick'] < row['first_incoming_match_tick'])
    return dict(first_return=(first_return, first_present),
                first_return_requiring_motion=(first_return * needs_motion, needs_motion),
                later_return=(after_first, opportunities_after_first),
                all_return=(row['own_returns'], row['own_opportunities']),
                return_per_game_second=(row['own_returns'], row['seconds']),
                later_point_return=(sum(point['own_returns'] for point in by_later_point),
                                    sum(point['own_opportunities'] for point in by_later_point)),
                goals_after_own_return=(row['scored_after_own_return'], 1),
                completed_point_won=(row['points_won'], row['completed_points']),
                lost_points=(row['points_lost'], 1),
                own_match_win=(int(row['winner'] == 'own'), 1),
                match_completed=(int(row['winner'] is not None), 1),
                binding_before_first=(int(binding_before_first), first_present),
                press_latency=(0 if row['first_press_tick'] is None else row['first_press_tick'],
                               int(row['first_press_tick'] is not None)),
                binding_latency=(0 if row['first_correct_binding_tick'] is None
                                 else row['first_correct_binding_tick'], int(row['first_correct_binding_tick'] is not None)))


def _interval(matched, *, draws=5000):
    count = len(next(iter(matched.values())))
    rng = np.random.default_rng(81_263_830)
    samples = rng.integers(count, size=(draws, count))
    centers, bootstrap = {}, {}
    for condition, rows in matched.items():
        metrics = [_metrics(row) for row in rows]
        centers[condition] = {}
        bootstrap[condition] = {}
        for metric in metrics[0]:
            values = np.asarray([row[metric] for row in metrics], dtype=float)
            numerator, denominator = values.sum(axis=0)
            sampled_sum = values[samples].sum(axis=1)
            rates = np.full(draws, np.nan)
            np.divide(sampled_sum[:, 0], sampled_sum[:, 1], out=rates, where=sampled_sum[:, 1] > 0)
            centers[condition][metric] = dict(numerator=float(numerator), denominator=float(denominator),
                                              estimate=float(numerator / denominator) if denominator else None)
            bootstrap[condition][metric] = rates
    differences = {}
    for left, right in PAIRS:
        if left not in centers or right not in centers:
            continue
        compare = {}
        for metric in centers[left]:
            baseline = centers[right][metric]['estimate']
            estimate = centers[left][metric]['estimate']
            values = bootstrap[left][metric] - bootstrap[right][metric]
            observed = values[np.isfinite(values)]
            compare[metric] = dict(delta=None if baseline is None or estimate is None else estimate - baseline,
                                   interval95=np.quantile(observed, [.025, .975]).tolist() if len(observed) else None,
                                   finite_draws=len(observed))
        differences[f'{left}_minus_{right}'] = compare
    return centers, differences


def _audit_matched(groups):
    cases = [[row['case'] for row in rows] for rows in groups.values()]
    if any(group != cases[0] for group in cases[1:]):
        raise AssertionError('compared match schedules differ')
    for rows in zip(*groups.values(), strict=True):
        openings = []
        for row in rows:
            incoming = row['first_incoming']
            opening = None if incoming is None else {key: item for key, item in incoming.items() if key != 'success'}
            openings.append((row['points'][0]['serve'], row['initial_down_button'], opening))
        if any(item != openings[0] for item in openings[1:]):
            raise AssertionError('matched opening trajectories differ before the first own opportunity')
        if 'disconnected' in groups and 'stationary' in groups:
            lookup = dict(zip(groups, rows, strict=True))
            disconnected = lookup['disconnected']
            stationary = lookup['stationary']
            if disconnected['drives_sha256'] != stationary['drives_sha256'] or disconnected['points'] != stationary['points']:
                raise AssertionError('disconnected arms changed the physics of stationary Pong')


def experiment(args):
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = _digest(checkpoint)
    gear = FAST_GEAR if args.gear == 'fast' else SOURCE_GEAR
    cases = matched_cases(args.matches, pilot=args.pilot)
    sources = {name: _digest(ROOT / name) for name in SOURCE_FILES}
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = args.output / 'matches'
    receipts.mkdir()
    memory, arm_practice = pretrain_motor(ARM_SOURCE_SEED, ticks=ARM_SOURCE_TICKS, gear=gear)
    _write(args.output / 'arm-motor.json', memory)
    max_ticks = round(args.seconds * NATIVE_HZ)
    source_schedule = checkpoint.with_name('source-schedule.json')
    practice_files = {name: _digest(checkpoint.with_name(name)) for name in ('source-schedule.json',
                      'source-practice.json', 'source-summary.json', 'source-interception.json')
                      if checkpoint.with_name(name).exists()}
    if source_schedule.exists():
        practice_seeds = {seed for block in json.loads(source_schedule.read_text())['starts'].values() for seed in block}
        if practice_seeds.intersection(case.pong_seed for case in cases):
            raise AssertionError('direct Pong and arm-driven evaluation share seed starts')
    conditions = args.arms
    if 'interception_only' in conditions and 'source-interception.json' not in practice_files:
        raise ValueError('the interception-only arm requires a source-interception checkpoint')
    manifest = dict(pilot=args.pilot, conditions=conditions, schedule=[asdict(case) for case in cases],
                    schedule_entropy=PILOT_ENTROPY if args.pilot else QUALIFICATION_ENTROPY,
                    opponent_speed=args.opponent_speed, max_ticks=max_ticks, first_to=args.first_to,
                    gear=gear, pong_checkpoint_sha256=checkpoint_hash, pong_practice_artifacts=practice_files,
                    arm_pretraining=arm_practice, arm_pretraining_seed=ARM_SOURCE_SEED,
                    arm_checkpoint_sha256=_digest(args.output / 'arm-motor.json'), source_sha256=sources,
                    inference='same two-button optical permutation learner, except stationary control',
                    primary=['first incoming return', 'actual point scored after an own return'],
                    bootstrap=dict(draws=5000, unit='matched independent match stream', seed=81_263_830))
    _write(args.output / 'manifest.json', manifest)
    starts = time.monotonic()
    jobs = [(ordinal, condition, case, checkpoint, memory, gear, args.first_to, max_ticks, args.opponent_speed)
            for ordinal, case in enumerate(cases) for condition in conditions]
    grouped = {condition: [None] * len(cases) for condition in conditions}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            ordinal, condition, row = future.result()
            grouped[condition][ordinal] = row
            _write(receipts / f'{ordinal:03d}-{condition}.json', row)
            if count % 8 == 0 or count == len(jobs):
                _write(args.output / 'progress.json', dict(done=count, total=len(jobs),
                       wall_seconds=time.monotonic() - starts))
                print(f'Native matches {count}/{len(jobs)} in {time.monotonic() - starts:.1f}s', flush=True)
    _audit_matched(grouped)
    stats, comparisons = _interval(grouped)
    if (checkpoint_hash != _digest(checkpoint) or sources != {name: _digest(ROOT / name) for name in SOURCE_FILES}
            or any(digest != _digest(checkpoint.with_name(name)) for name, digest in practice_files.items())):
        raise AssertionError('experiment code or source checkpoint changed during execution')
    totals = {condition: dict(physics_ticks=sum(row['ticks'] for row in rows),
                              camera_samples=sum(row['screen_samples'] for row in rows),
                              probing_ticks=sum(row['probe_ticks'] for row in rows),
                              motor_probe_ticks=sum(row['arm_motor_probe_ticks'] for row in rows),
                              scored_points=sum(row['points_won'] for row in rows),
                              first_point_goal_after_return=sum(row['points'][0]['scored_after_own_return'] for row in rows),
                              later_points_goals_after_return=sum(point['scored_after_own_return'] for row in rows
                                                                  for point in row['points'][1:]),
                              lost_points=sum(row['points_lost'] for row in rows),
                              time_caps=sum(row['winner'] is None for row in rows),
                              stationary_opportunities=sum(row['first_incoming'] is not None
                                                           and not row['initial_approach_needs_motion'] for row in rows),
                              first_correct_binding_median_ticks=_median(rows, 'first_correct_binding_tick'),
                              first_press_median_ticks=_median(rows, 'first_press_tick'),
                              first_physical_return_median_ticks=_median(rows, 'return_ticks', first=True))
              for condition, rows in grouped.items()}
    report = dict(manifest_sha256=_digest(args.output / 'manifest.json'), comparisons=comparisons,
                  conditions=stats, totals=totals, native_match_count=len(jobs), wall_seconds=time.monotonic() - starts)
    _write(args.output / 'results.json', report)
    print(json.dumps(totals, indent=2), flush=True)


def _median(rows, key, *, first=False):
    data = [row[key][0] for row in rows if row[key]] if first else [row[key] for row in rows if row[key] is not None]
    return float(np.median(data)) if data else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--matches', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--first-to', type=int, default=3)
    parser.add_argument('--opponent-speed', type=float, default=.35)
    parser.add_argument('--gear', choices=('fast', 'source'), default='source')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--arms', choices=CONDITIONS, nargs='+', default=CONDITIONS)
    args = parser.parse_args()
    if min(args.matches, args.seconds, args.first_to, args.workers) <= 0 or len(set(args.arms)) != len(args.arms):
        parser.error('counts and duration must be positive, and selected comparison arms unique')
    experiment(args)


if __name__ == '__main__':
    main()

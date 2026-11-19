"""Independently qualify score-trained play against same-start Pong controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import secrets
import signal
import time

import numpy as np

from fly_connectome.control.session import model_bytes, source_signature
from run_pong_inverse_cycle import save, sha
from run_pong_rapid_recovery_qualification import rate, read, worker_init
from run_pong_surprise_recovery import MODULUS, exclusions, runtime_dependencies

TASKS = {'scoring': dict(opponent=.35, starts=32, arms=('trained', 'center', 'fixed_exploration')),
         'retention': dict(opponent=.65, starts=20, arms=('trained', 'center'))}
WORKERS = 4
WALL_SECONDS = 300
DURATION = 12.
GATES = dict(scoring_gain_floor=.15, score_p_max=.05, retention_first_floor=.90,
             retention_later_floor=.95, retention_later_minimum=30,
             retention_max_loss_vs_center=.05)
SCORE_RUNTIME = ('pong_score_placement.py', 'pong_transfer_arena.py', 'run_pong_score_placement.py',
                 'pong_timing_controller.py', 'pong_rapid_motor_recovery.py', 'pong_rapid_motor_memory.py')


def prepare(directory, prior):
    model_bytes()
    used, exclusions_metadata = exclusions(prior)
    rng = np.random.default_rng(secrets.randbits(128))
    schedules = {}
    for task, config in TASKS.items():
        seeds = []
        while len(seeds) < config['starts']:
            seed = int(rng.integers(1, MODULUS))
            if seed not in used and ((seed*48271+1) % MODULUS) % 2 == len(seeds) % 2:
                used.add(seed)
                seeds.append(seed)
        rng.shuffle(seeds)
        schedules[task] = seeds
    directory.mkdir(parents=True, exist_ok=False)
    schedule = directory / 'schedules.json'
    save(schedule, schedules)
    spec = dict(contract='fresh-paired-score-only-learned-placement-v1', sources=source_signature(),
                dependencies=runtime_dependencies(), harness_sha256=sha(Path(__file__)),
                schedule_sha256=sha(schedule), excluded_schedules=exclusions_metadata, tasks=TASKS,
                episode_seconds=DURATION, gates=GATES, workers=WORKERS, wall_seconds=WALL_SECONDS,
                scoring_endpoint='player scores after at least one own hit, divided by all planned starts',
                natural_serve_wins='reported separately and contribute zero own-hit wins',
                frozen='freeze only the scoring parameters; identical carried timing and live motor/timing updates in every arm',
                policy_seed='independently assigned from the trial index, shared across matched arms',
                scout_policy='fresh starts are hidden until the score checkpoint and source are frozen')
    save(directory / 'spec.json', spec)
    return sha(directory / 'spec.json')


def verify(directory, digest):
    path = directory / 'spec.json'
    if sha(path) != digest:
        raise ValueError('spec SHA-256 differs from registration')
    spec = read(path)
    model_bytes()
    if (spec['sources'] != source_signature() or spec['dependencies'] != runtime_dependencies()
            or spec['harness_sha256'] != sha(Path(__file__))
            or spec['schedule_sha256'] != sha(directory / 'schedules.json')):
        raise ValueError('prepared environment or schedules changed')
    return spec


def play(output, task, arm, seed, policy_seed, score_state, motor_state, timing_state, deadline):
    from pong_score_placement import ScorePlacement
    from pong_transfer_arena import play as play_native
    from run_pong_score_placement import build_controller

    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return False
    model = ScorePlacement(**(score_state if arm == 'trained' else {}), updates_enabled=False)
    before = model.to_state()
    exploration = .65 if arm == 'fixed_exploration' else 0.
    controller = build_controller(policy_seed, motor_state, timing_state, model, exploration, timing_updates=True)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        row = play_native(controller, seed, opponent_speed=TASKS[task]['opponent'], duration=DURATION)
    except TimeoutError:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if model.to_state() != before or row['own_hit_win'] != int(row['score'] > 0 and row['hits'] > 0):
        raise ValueError('frozen score state or own-hit scoring audit differs')
    row.update(arm=arm, task=task, natural_serve_win=bool(row['score'] > 0 and not row['hits']),
               aimed_contacts=sum(choice.offset != 0 for choice in controller.placement.choices),
               score_choices=len(controller.placement.choices))
    save(output / task / arm / f'{seed}.json', row)
    return True


def metrics(rows):
    exposed = [row['contacts'] for row in rows if row['contacts']]
    first_hits = sum(events[0][1] for events in exposed)
    later = [event for events in exposed for event in events[1:]]
    return dict(games=len(rows), own_hit_wins=sum(row['own_hit_win'] for row in rows),
                own_hit_win_fraction=rate(sum(row['own_hit_win'] for row in rows), len(rows)),
                natural_serve_wins=sum(row['natural_serve_win'] for row in rows),
                first_returns=first_hits, first_opportunities=len(exposed), first_fraction=rate(first_hits, len(exposed)),
                later_returns=sum(event[1] for event in later), later_opportunities=len(later),
                later_fraction=rate(sum(event[1] for event in later), len(later)),
                losses=sum(row['score'] < 0 for row in rows), timeouts=sum(row['score'] == 0 for row in rows),
                aimed_contacts=sum(row['aimed_contacts'] for row in rows), cameras=sum(row['cameras'] for row in rows))


def paired(reference, candidate):
    shared = reference.keys() & candidate.keys()
    gain = loss = clock_mismatch = 0
    for seed in shared:
        base, trial = reference[seed], candidate[seed]
        difference = trial['own_hit_win'] - base['own_hit_win']
        gain += difference > 0
        loss += difference < 0
        base_contacts, candidate_contacts = base['contacts'], trial['contacts']
        clock_mismatch += (base_contacts[0][0] if base_contacts else None) != (
                          candidate_contacts[0][0] if candidate_contacts else None)
    n = gain+loss
    p = sum(math.comb(n, k) for k in range(gain, n+1)) / 2**n
    return dict(pairs=len(shared), gained_own_hit_wins=gain, lost_own_hit_wins=loss,
                own_hit_win_gain=rate(gain-loss, len(shared)), exact_one_sided_p=p,
                first_clock_mismatches=clock_mismatch)


def assess(stats, evidence, complete):
    center = evidence['scoring']['center']
    random = evidence['scoring']['fixed_exploration']
    ref, trial = (stats['retention'][arm] for arm in ('center', 'trained'))
    score = (complete and center['own_hit_win_gain'] >= GATES['scoring_gain_floor']
             and center['exact_one_sided_p'] <= GATES['score_p_max'])
    first = trial['first_fraction'] >= max(GATES['retention_first_floor'],
                                          ref['first_fraction']-GATES['retention_max_loss_vs_center'])
    later = (trial['later_opportunities'] >= GATES['retention_later_minimum']
             and trial['later_fraction'] >= max(GATES['retention_later_floor'],
                                                 ref['later_fraction']-GATES['retention_max_loss_vs_center']))
    return dict(score_gain_qualification=score, retention_first=first, retention_later=later,
                trained_beats_fixed_exploration=complete and random['own_hit_win_gain'] > 0
                    and random['exact_one_sided_p'] <= GATES['score_p_max'],
                full_qualification=score and first and later)


def run(directory, digest, score_checkpoint, motor_checkpoint, timing_checkpoint):
    import pong_score_placement

    verify(directory, digest)
    runtime_root = Path(pong_score_placement.__file__).parent
    runtime = {name: sha(runtime_root / name) for name in SCORE_RUNTIME}
    input_paths = dict(score=score_checkpoint, motor=motor_checkpoint, timing=timing_checkpoint)
    inputs = {name: sha(path) for name, path in input_paths.items()}
    score, motor, timing = (read(path) for path in input_paths.values())
    if not any(score['visits']):
        raise ValueError('the candidate checkpoint has no score-feedback updates')
    schedules = read(directory / 'schedules.json')
    output = directory / 'run'
    output.mkdir(exist_ok=False)
    for task, config in TASKS.items():
        for arm in config['arms']:
            (output / task / arm).mkdir(parents=True)
    started = time.monotonic()
    deadline = started+WALL_SECONDS
    save(output / 'dispatch.json', dict(spec_sha256=digest, runtime_sha256=runtime,
         checkpoints_sha256=inputs, deadline=deadline))
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=worker_init) as pool:
        jobs = {pool.submit(play, output, task, arm, seed, 1245 + index, score, motor, timing,
                            deadline): (task, arm, seed)
                for task, config in TASKS.items() for index, seed in enumerate(schedules[task])
                for arm in config['arms']}
        for future in as_completed(jobs):
            try:
                if not future.result():
                    errors.append((jobs[future], 'deadline'))
            except Exception as error:
                errors.append((jobs[future], repr(error)))
    verify(directory, digest)
    if ({name: sha(runtime_root / name) for name in SCORE_RUNTIME} != runtime
            or {name: sha(path) for name, path in input_paths.items()} != inputs):
        raise ValueError('frozen evaluation source or checkpoint changed')
    receipts = {task: {arm: {row['seed']: row for row in
                (read(path) for path in (output / task / arm).glob('*.json'))} for arm in config['arms']}
                for task, config in TASKS.items()}
    complete = not errors and all(set(schedules[task]) == set(rows)
                                  for task, arms in receipts.items() for rows in arms.values())
    stats = {task: {arm: metrics(list(rows.values())) for arm, rows in arms.items()} for task, arms in receipts.items()}
    evidence = {task: {arm: paired(rows, arms['trained']) for arm, rows in arms.items() if arm != 'trained'}
                for task, arms in receipts.items()}
    summary = dict(complete=complete, errors=errors, spec_sha256=digest, elapsed_wall_seconds=time.monotonic()-started,
                   arms=stats, paired=evidence, qualification=assess(stats, evidence, complete))
    save(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('prepare')
    planning.add_argument('directory', type=Path)
    planning.add_argument('prior_experiments', type=Path)
    running = commands.add_parser('run')
    running.add_argument('directory', type=Path)
    running.add_argument('--score-checkpoint', type=Path, required=True)
    running.add_argument('--motor-checkpoint', type=Path, required=True)
    running.add_argument('--timing-checkpoint', type=Path, required=True)
    running.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.directory.resolve(), args.prior_experiments.resolve()))
    else:
        report = run(args.directory.resolve(), args.spec_sha256, args.score_checkpoint.resolve(),
                     args.motor_checkpoint.resolve(), args.timing_checkpoint.resolve())
        print(json.dumps({key: report[key] for key in ('complete', 'errors', 'qualification')}, sort_keys=True))

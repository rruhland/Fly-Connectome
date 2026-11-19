"""Practice with game scores, then evaluate frozen placement on fresh starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from pong_rapid_motor_memory import RapidMotorMemory
from pong_score_placement import OFFSETS, ScorePlacement, ScorePlacementController
from pong_timing_controller import TimedMotorController
from pong_transfer_arena import interception_policy, play
from pong_visual_button_effects import VisualButtonEffects
from run_pong_surprise_recovery import MODULUS, exclusions


SCORING_OPPONENT = .35
RETENTION_OPPONENT = .65
EPISODE_SECONDS = 12.
BOOTSTRAP_DRAWS = 10_000
POLICY_RNG_ENTROPY = 3070771140478503531


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def verify_schedule(schedule):
    keys = ('practice', 'scoring', 'retention')
    if set(schedule) != set(keys) or not all(schedule[key] for key in keys):
        raise ValueError('nonempty practice, scoring and retention starts are required')
    seeds = [seed for key in keys for seed in schedule[key]]
    if (not all(type(seed) is int and 0 < seed < MODULUS for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('all scheduled starts must be valid and mutually disjoint')
    return schedule


def make_schedule(used, counts, rng_seed):
    rng = np.random.default_rng(rng_seed)
    used = set(used)
    schedule = {}
    for key, count in counts.items():
        seeds = []
        while len(seeds) < count:
            seed = int(rng.integers(1, MODULUS))
            if seed not in used:
                seeds.append(seed)
                used.add(seed)
        schedule[key] = seeds
    return verify_schedule(schedule)


def policy_rng_schedule(schedule):
    """Generate one exploration stream per chronological start, independent of play."""
    streams = {}
    index = 0
    for phase in ('practice', 'scoring', 'retention'):
        streams[phase] = []
        for _ in schedule[phase]:
            stream = np.random.SeedSequence(POLICY_RNG_ENTROPY, spawn_key=(index,))
            streams[phase].append(int(stream.generate_state(1, dtype=np.uint64)[0]))
            index += 1
    return streams


def build_controller(policy_seed, motor_state, timing_state, score_policy, exploration=0., *,
                     timing_updates=False, forecast_center=False):
    if motor_state['schema'] == 'public-visual-motor-order-memory-v1':
        learner = RapidMotorMemory.from_state(motor_state)
    else:
        learner = VisualButtonEffects.from_state(motor_state)
    timing = TimedMotorController(interception_policy(), learner,
                                 timing_state=timing_state, timing_updates=timing_updates)
    episode = score_policy.episode(policy_seed, exploration=exploration)
    return ScorePlacementController(timing, episode, forecast_center=forecast_center)


def run_episode(seed, opponent, motor_state, timing_state, score_policy, *,
                policy_seed, exploration=0., forecast_center=False, alternative=None):
    controller = build_controller(policy_seed, motor_state, timing_state, score_policy, exploration,
                                  forecast_center=forecast_center)
    configuration = {} if alternative is None else alternative
    row = play(controller, seed, opponent_speed=opponent, duration=EPISODE_SECONDS,
               **configuration)
    row['policy_seed'] = policy_seed
    row['choices'] = [asdict(choice) for choice in controller.placement.choices]
    row['natural_serve_win'] = int(row['score'] == 1 and row['hits'] == 0)
    row['loss'] = int(row['score'] == -1)
    row['timeout'] = int(row['score'] == 0)
    return row


def evaluate(job):
    (arm, seed, policy_seed, opponent, motor_state, timing_state,
     score_state, exploration, forecast_center) = job
    torch.set_num_threads(1)
    policy = ScorePlacement(**score_state, updates_enabled=False)
    row = run_episode(seed, opponent, motor_state, timing_state, policy,
                      policy_seed=policy_seed, exploration=exploration,
                      forecast_center=forecast_center)
    return arm, row


def summarize(rows):
    first = [bool(row['contacts'][0][1]) for row in rows if row['contacts']]
    hits = sum(row['hits'] for row in rows)
    opportunities = sum(row['opportunities'] for row in rows)
    return dict(rallies=len(rows), own_hit_wins=sum(row['own_hit_win'] for row in rows),
                natural_serve_wins=sum(row['natural_serve_win'] for row in rows),
                losses=sum(row['loss'] for row in rows), timeouts=sum(row['timeout'] for row in rows),
                first_returns=sum(first), first_opportunities=len(first),
                returns=hits, opportunities=opportunities,
                cameras=sum(row['cameras'] for row in rows),
                action_count=sum(len(row['choices']) for row in rows),
                aimed_contacts=sum(choice['offset'] != 0. for row in rows for choice in row['choices']),
                natural_serve_assignments=sum(row['natural_serve_win'] and bool(row['choices'])
                                              for row in rows))


def paired_difference(learned, control):
    if [row['seed'] for row in learned] != [row['seed'] for row in control]:
        raise ValueError('paired starts must have identical ordered seeds')
    differences = (np.array([row['own_hit_win'] for row in learned])
                   - np.array([row['own_hit_win'] for row in control]))
    rng = np.random.default_rng(5931021)
    sample = rng.integers(len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    return dict(own_hit_win_difference=float(differences.mean()), sample_count=len(differences),
                confidence_interval=np.quantile(differences[sample].mean(axis=1), [.025, .975]).tolist(),
                improved=int((differences > 0).sum()), regressed=int((differences < 0).sum()))


def practice_sequential(out, schedule, policy_seeds, motor_state, timing_state, initial):
    policy = ScorePlacement(**initial)
    rows = []
    for index, (seed, policy_seed) in enumerate(zip(schedule, policy_seeds, strict=True)):
        # Encounter fresh starts in order and update immediately after each rally.
        exploration = .80 - .35 * index / len(schedule)
        before = int(policy.visits.sum())
        row = run_episode(seed, SCORING_OPPONENT, motor_state, timing_state, policy,
                          policy_seed=policy_seed, exploration=exploration)
        row.update(exploration=exploration, assignment_updated=int(policy.visits.sum()) != before)
        rows.append(row)
        write(out / 'practice.json', rows)
        write(out / 'score-state.json', policy.to_state())
        print(f'practice {index + 1}/{len(schedule)} | score {row["score"]:+}',
              flush=True)
    return policy, rows


def frozen_comparison(schedule, policy_seeds, motor_state, timing_state, initial, endpoint, workers):
    jobs = []
    for task, opponent, arms in (
            ('scoring', SCORING_OPPONENT,
             [('trained', endpoint, 0., False), ('cold', initial, 0., False),
              ('explore', initial, 1., False), ('forecast_center', initial, 0., True)]),
            ('retention', RETENTION_OPPONENT,
             [('trained', endpoint, 0., False), ('cold', initial, 0., False)])):
        for seed, policy_seed in zip(schedule[task], policy_seeds[task], strict=True):
            for arm, state, exploration, forecast_center in arms:
                jobs.append(((task, arm), seed, policy_seed, opponent, motor_state, timing_state,
                             state, exploration, forecast_center))
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (task, arm), row in pool.map(evaluate, jobs):
            rows.setdefault(task, {}).setdefault(arm, []).append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--motor-state', type=Path, required=True)
    parser.add_argument('--timing-state', type=Path, required=True)
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--prior-experiments', type=Path)
    parser.add_argument('--practice', type=int, default=96)
    parser.add_argument('--scoring', type=int, default=48)
    parser.add_argument('--retention', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.schedule is None and args.prior_experiments is None:
        parser.error('provide a paired schedule or the prior experiment directory')
    if min(args.practice, args.scoring, args.retention, args.workers) <= 0:
        parser.error('positive practice, scoring, retention and worker counts required')
    torch.set_num_threads(1)
    motor_state = read(args.motor_state)
    timing_state = read(args.timing_state)
    if args.schedule:
        schedule = verify_schedule(read(args.schedule))
    else:
        used, _ = exclusions(args.prior_experiments)
        schedule = make_schedule(used, dict(practice=args.practice, scoring=args.scoring,
                                            retention=args.retention), 913851731)
    args.out.mkdir(parents=True, exist_ok=False)
    initial = ScorePlacement().to_state()
    policy_seeds = policy_rng_schedule(schedule)
    write(args.out / 'schedule.json', schedule)
    write(args.out / 'policy-rng.json', dict(entropy=POLICY_RNG_ENTROPY, streams=policy_seeds))
    write(args.out / 'input.json', dict(motor_state=motor_state, timing_state=timing_state,
          score_state=initial, candidate_offsets=OFFSETS, episode_seconds=EPISODE_SECONDS))
    policy, practice = practice_sequential(args.out, schedule['practice'], policy_seeds['practice'],
                                          motor_state, timing_state, initial)
    endpoint = policy.frozen_branch().to_state()
    write(args.out / 'score-state-frozen.json', endpoint)
    rows = frozen_comparison(schedule, policy_seeds, motor_state, timing_state, initial,
                             endpoint, args.workers)
    write(args.out / 'frozen.json', rows)
    write(args.out / 'summary.json', dict(practice=summarize(practice),
          scoring={arm: summarize(group) for arm, group in rows['scoring'].items()},
          retention={arm: summarize(group) for arm, group in rows['retention'].items()},
          score_difference={arm: paired_difference(rows['scoring']['trained'], rows['scoring'][arm])
                            for arm in ('cold', 'explore', 'forecast_center')}))


if __name__ == '__main__':
    main()

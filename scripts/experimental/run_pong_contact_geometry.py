"""Evaluate camera contact geometry around a fixed learned score table."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pong_contact_geometry import ContactGeometryController
from pong_score_placement import ScorePlacement
from pong_transfer_arena import play
from run_pong_score_placement import build_controller, read, summarize, write
from run_pong_surprise_recovery import MODULUS, exclusions


FIRST_SWITCH = 2.4


def evaluate(job):
    name, policy_seed, seed, condition, motor_state, timing_state, score_state = job
    torch.set_num_threads(1)
    policy = ScorePlacement(**score_state, updates_enabled=False)
    controller = build_controller(policy_seed, motor_state, timing_state, policy,
                                  timing_updates=condition == 'mixed')
    if name in ('camera_geometry', 'uncertainty_guard'):
        controller = ContactGeometryController(controller.timing, controller.placement,
                                               guard=name == 'uncertainty_guard')
    game = {'opponent_speed': .35 if condition == 'scoring' else .65}
    if condition == 'mixed':
        game.update(delay_frames=12, mapping=(1, 2, 0), switch_seconds=FIRST_SWITCH)
    row = play(controller, seed, duration=12., **game)
    row.update(choices=[asdict(choice) for choice in controller.placement.choices],
               natural_serve_win=int(row['score'] == 1 and not row['hits']),
               loss=int(row['score'] == -1), timeout=int(row['score'] == 0))
    return condition, name, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('timing_r2', type=Path)
    parser.add_argument('score_state', type=Path)
    parser.add_argument('--starts', type=int, default=10)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    motor_state = read(args.timing_r2 / 'checkpoint-acquired.json')
    restored = read(args.timing_r2 / 'checkpoints' / 'restored.json')
    delayed = read(args.timing_r2 / 'checkpoints' / 'held-12.json')
    score_state = read(args.score_state)
    empty = ScorePlacement().to_state()
    used, _ = exclusions(args.out.parent)
    jobs = []
    rng = np.random.default_rng(1842674)
    policy_rng = np.random.default_rng(9413871042)
    for condition in ('scoring', 'retention', 'mixed'):
        for _ in range(args.starts):
            seed = int(rng.integers(1, MODULUS))
            while seed in used:
                seed = int(rng.integers(1, MODULUS))
            used.add(seed)
            policy_seed = int(policy_rng.integers(1 << 62, 1 << 63))
            for name in ('original', 'camera_geometry', 'uncertainty_guard', 'center'):
                if name == 'center' and condition == 'scoring':
                    continue
                jobs.append((name, policy_seed, seed, condition, motor_state,
                             delayed if condition == 'mixed' else restored,
                             empty if name == 'center' else score_state))
    write(args.out / 'schedule.json', [[j[0], j[1], j[2], j[3]] for j in jobs])
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for condition, name, row in pool.map(evaluate, jobs):
            rows.setdefault(condition, {}).setdefault(name, []).append(row)
            write(args.out / 'rows.json', rows)
    summary = {condition: {arm: summarize(group) for arm, group in arms.items()}
               for condition, arms in rows.items()}
    for arm, group in rows['mixed'].items():
        post_switch = [[returned for time, returned in row['contacts'] if time >= FIRST_SWITCH]
                       for row in group]
        firsts = [returns[0] for returns in post_switch if returns]
        later = [hit for returns in post_switch for hit in returns[1:]]
        summary['mixed'][arm].update(first_after_switch=[sum(firsts), len(firsts)],
                                     later_after_switch=[sum(later), len(later)])
    write(args.out / 'summary.json', summary)


if __name__ == '__main__':
    main()

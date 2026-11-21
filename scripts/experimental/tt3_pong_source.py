"""Fit the source role and outcome models and collect the paired Pong reference."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .tt3_learning import RELATIONS, ReturnMemory
from .tt3_pong import (
    CAMERA_SIZE,
    CAMERA_TICKS,
    MINIMUM_LATER_OPPORTUNITIES,
    MINIMUM_RETENTION_CASES,
    PRACTICE_OPPONENT_SPEED,
    RALLY_TICKS,
    RETENTION_METRICS,
    cases_from_seeds,
    evaluate_pong,
    paired_pong,
    practice_pong,
    source_episodes,
    state_sha256,
    summarize_pong,
)
from .tt3_visual import RoleModel


def write_json(path, contents):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(contents, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_schedule(seed, exposures, practice_starts, evaluation_starts, practice_points,
                    evaluation_points, max_ticks, opponent_speed):
    rng = np.random.default_rng(seed)

    def distinct_starts(low, count):
        return (rng.choice(200_000_000, size=count, replace=False) + low).tolist()

    role_seeds = distinct_starts(100_000_000, exposures)
    practice = cases_from_seeds(distinct_starts(350_000_000, practice_starts))
    evaluation = cases_from_seeds(distinct_starts(600_000_000, evaluation_starts))
    return dict(schema='native-pong-source-v1', seed=seed, role_episodes=role_seeds,
                practice=[asdict(case) for case in practice], evaluation=[asdict(case) for case in evaluation],
                practice_points=practice_points, evaluation_points=evaluation_points, max_ticks=max_ticks,
                opponent_speed=opponent_speed, camera_size=CAMERA_SIZE, camera_ticks=CAMERA_TICKS,
                minimum_matches=MINIMUM_RETENTION_CASES, minimum_later=MINIMUM_LATER_OPPORTUNITIES,
                retention_criteria=RETENTION_METRICS)


def main(argv=None):
    from .tt3_pong import PongCase

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=59369005)
    parser.add_argument('--raw-episodes', type=int, default=48)
    parser.add_argument('--practice-starts', type=int, default=72)
    parser.add_argument('--evaluation-starts', type=int, default=48)
    parser.add_argument('--practice-points', type=int, default=4)
    parser.add_argument('--evaluation-points', type=int, default=6)
    parser.add_argument('--ticks', type=int, default=RALLY_TICKS)
    parser.add_argument('--opponent-speed', type=float, default=PRACTICE_OPPONENT_SPEED)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.raw_episodes, args.practice_starts, args.evaluation_starts, args.practice_points,
           args.evaluation_points, args.ticks) < 1:
        parser.error('exposure, practice, evaluation and tick counts must be positive')
    torch.set_num_threads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    schedule = source_schedule(args.seed, args.raw_episodes, args.practice_starts, args.evaluation_starts,
                               args.practice_points, args.evaluation_points, args.ticks, args.opponent_speed)
    write_json(output / 'pong-schedule.json', schedule)
    physics = dict(max_ticks=args.ticks, opponent_speed=args.opponent_speed)
    role_model = RoleModel.fit(source_episodes(schedule['role_episodes'], **physics))
    write_json(output / 'pong-roles.json', asdict(role_model))
    memory = ReturnMemory()
    practice = practice_pong([PongCase(**case) for case in schedule['practice']], role_model, memory,
                             points_per_match=args.practice_points, **physics)
    checkpoint = dict(role_model=asdict(role_model), return_memory=memory.state(),
                      schedule_sha256=state_sha256(schedule), schema=schedule['schema'])
    write_json(output / 'pong-source.json', checkpoint)
    practice_summary = {name: summarize_pong([row for row in practice if row['relation'] == index])
                        for index, name in enumerate(RELATIONS) if any(row['relation'] == index for row in practice)}
    write_json(output / 'pong-practice.json', dict(summary=practice_summary, points=practice))
    print(json.dumps(dict(return_memory=memory.state(), practice_summary=practice_summary)), flush=True)
    if args.source_only:
        return

    cases = [PongCase(**case) for case in schedule['evaluation']]
    eval_options = dict(points_per_match=args.evaluation_points, **physics)
    trained = evaluate_pong(cases, role_model, memory, **eval_options)
    write_json(output / 'pong-before.json', dict(source_state_sha256=state_sha256(checkpoint),
                                                schedule_sha256=state_sha256(schedule),
                                                summary=summarize_pong(trained), points=trained))
    reset = evaluate_pong(cases, role_model, ReturnMemory(), **eval_options)
    write_json(output / 'pong-reset.json', dict(summary=summarize_pong(reset), points=reset,
                                               reset_vs_before=paired_pong(trained, reset)))
    print(json.dumps(dict(before=summarize_pong(trained), reset=summarize_pong(reset))), flush=True)


if __name__ == '__main__':
    main()

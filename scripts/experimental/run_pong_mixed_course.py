"""Carry one Pong learner through two opponents, a hidden remap and a delay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from pong_score_placement import ScorePlacement
from pong_transfer_arena import play
from run_pong_inverse_cycle import save, sha
from run_pong_score_placement import build_controller


PHASES = (
    ('ordinary_scoring', .35, (0, 1, 2), 0),
    ('remap_and_delay', .35, (1, 2, 0), 12),
    ('faster_opponent', .65, (1, 2, 0), 12),
    ('ordinary_again', .35, (0, 1, 2), 0),
)


def summarize(rows):
    contacts = [row['contacts'] for row in rows if row['contacts']]
    return dict(games=len(rows), own_hit_wins=sum(row['own_hit_win'] for row in rows),
                first=sum(int(c[0][1]) for c in contacts), openings=len(contacts),
                later=sum(int(hit) for c in contacts for _, hit in c[1:]),
                later_opportunities=sum(len(c) - 1 for c in contacts),
                losses=sum(row['score'] == -1 for row in rows),
                natural_wins=sum(row['score'] == 1 and not row['hits'] for row in rows),
                score_updates=sum(row['updates'] for row in rows))


def arm_run(root, arm, schedules, motor, timing, learned_score, guarded, carry_risk):
    torch.set_num_threads(1)
    score = ScorePlacement(**learned_score) if arm == 'scoring' else ScorePlacement(updates_enabled=False)
    builder = build_controller
    shot_memory = None
    if carry_risk:
        from pong_carried_shot_value import ShotMemory, build_carried_shot_controller
        builder = build_carried_shot_controller
        shot_memory = ShotMemory()
    elif guarded:
        from pong_risk_aware_placement import build_risk_aware_controller
        builder = build_risk_aware_controller
    rows = []
    totals = {}
    for phase, opponent, mapping, lag in PHASES:
        phase_rows = []
        for case in schedules[phase]:
            # Only the arena receives the current opponent, map and delay.
            kwargs = {'memory': shot_memory} if carry_risk else {}
            controller = builder(case['policy'], motor, timing, score, timing_updates=True, **kwargs)
            initial = controller.timing.state()
            visits = int(score.visits.sum())
            row = play(controller, case['serve'], opponent_speed=opponent, duration=20.,
                       delay_frames=lag, mapping=mapping)
            timing = controller.timing.state()
            row.update(phase=phase, initial_timing=initial, final_timing=timing,
                       updates=int(score.visits.sum()) - visits,
                       guarded_approaches=getattr(controller, 'cancelled', {}))
            if carry_risk:
                row['initial_shot_memory'] = shot_memory.to_state()
                shot_memory = controller.shot_memory
                row['final_shot_memory'] = shot_memory.to_state()
            rows.append(row)
            phase_rows.append(row)
            save(root / f'{arm}.json', rows)
        totals[phase] = summarize(phase_rows)
    save(root / f'{arm}-final.json', dict(timing=timing, score=score.to_state(),
                                          shot_memory=shot_memory.to_state() if carry_risk else None))
    return totals


def run(root, motor_file, timing_file, score_file, games, schedule_seed, guarded=False, carry_risk=False):
    if games <= 0:
        raise ValueError('each phase needs at least one game')
    root.mkdir(parents=True, exist_ok=False)
    environment_rng = np.random.default_rng(schedule_seed)
    policy_rng = np.random.default_rng(67227996)
    schedules = {}
    used = set()
    for phase, *_ in PHASES:
        cases = []
        while len(cases) < games:
            seed = int(environment_rng.integers(1, 2147483647))
            if seed in used or ((seed * 48271 + 1) % 2147483647) % 2 != len(cases) % 2:
                continue
            used.add(seed)
            cases.append(dict(serve=seed, policy=int(policy_rng.integers(1, 2147483647))))
        schedules[phase] = cases
    save(root / 'spec.json', dict(phases=PHASES, schedules=schedules, games_per_phase=games, schedule_seed=schedule_seed, guarded=guarded, carry_risk=carry_risk,
         motor_sha=sha(motor_file), timing_sha=sha(timing_file), score_sha=sha(score_file),
         script_sha=sha(Path(__file__)), measure='All returns, opening returns and wins after '
         'an own hit. Chronological center and scoring streams carry delay and button belief.'))
    data = [json.loads(f.read_text()) for f in (motor_file, timing_file, score_file)]
    with ProcessPoolExecutor(2) as pool:
        jobs = {arm: pool.submit(arm_run, root, arm, schedules, *data, guarded, carry_risk) for arm in ('center', 'scoring')}
        save(root / 'summary.json', {arm: job.result() for arm, job in jobs.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('output', 'motor', 'timing', 'score'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--schedule-seed', type=int, default=54728034)
    parser.add_argument('--guarded', action='store_true')
    parser.add_argument('--carry-risk', action='store_true')
    args = parser.parse_args()
    run(args.output, args.motor, args.timing, args.score, args.games, args.schedule_seed, args.guarded, args.carry_risk)

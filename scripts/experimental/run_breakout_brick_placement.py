"""Compare online brick aiming, its frozen prior, centered returns and Pong aiming."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from breakout_brick_placement import BrickPlacement, BrickShotPolicy, canonical_camera
from pong_carried_shot_value import ShotMemory, build_carried_shot_controller
from pong_score_placement import ScorePlacement


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'artifacts/m1b/2026-11-19-pong-after-mixed-course.json'
SCHEDULES = {'practice': (194740131, 0), 'heldout': (981405461, 0),
             'heldout_delay': (478271173, 12)}
PRACTICE_ARMS = ('brick_live', 'brick_prior', 'center', 'pong')
TEST_ARMS = ('brick_trained', 'brick_prior', 'center', 'pong')


def save(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def score_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def sparse_layout(case, bricks):
    """Make matched sparse endgames with disjoint practice and held-out cells."""
    from breakout_arena import BreakoutLayout, COLUMNS, ROWS

    rng = np.random.default_rng(case.layout_seed)
    parity = int(case.layout_split == 'heldout')
    while True:
        cells = rng.choice(ROWS * COLUMNS, size=bricks, replace=False).tolist()
        if sum(cells) % 2 == parity:
            return BreakoutLayout(tuple(divmod(cell, COLUMNS) for cell in cells))


def summarize(lives):
    seconds = sum(row['elapsed_seconds'] for row in lives)
    score = sum(row['score'] for row in lives)
    own_score = sum(row['post_own_bricks'] for row in lives)
    returns = sum(row['own_returns'] for row in lives)
    opportunities = sum(row['own_opportunities'] for row in lives)
    return dict(lives=len(lives), bricks=score, bricks_after_own_return=own_score,
                serve_bricks=sum(row['serve_bricks'] for row in lives),
                seconds=seconds, bricks_per_life=score / len(lives),
                bricks_after_own_return_per_life=own_score / len(lives),
                bricks_per_minute=60. * score / seconds,
                bricks_after_own_return_per_minute=60. * own_score / seconds,
                returns=returns, opportunities=opportunities,
                return_fraction=returns / opportunities if opportunities else None,
                lives_lost=sum(row['life_lost'] for row in lives),
                boards_cleared=sum(row['cleared'] for row in lives),
                truncated=sum(row['truncated'] for row in lives),
                score_updates=sum(row['score_updates'] for row in lives),
                noncenter_choices=sum(shot['offset'] != 0. for row in lives for shot in row['shots']))


def play_life(case, layout, checkpoint, timing, detector, arm, policy, seed, *, cap, lag):
    from pong_role_transfer import RoleController

    camera = case.camera(layout=layout, max_episode_ticks=round(cap * 120), delay_frames=lag)
    frozen_score = ScorePlacement(**checkpoint['score'], updates_enabled=False)
    controller = build_carried_shot_controller(
        seed, checkpoint['motor'], timing, frozen_score,
        memory=ShotMemory(**checkpoint['shot_memory']), timing_updates=True)
    brick = None
    if arm == 'center':
        controller.timing.contact_offset = 0.
    elif policy is not None:
        brick = BrickPlacement(controller.timing, policy, seed)
    bridge = RoleController(controller, detector)
    frame = camera.frame()
    bridge.reset(frame)
    elapsed = 0.
    score = 0.
    requests = bytearray()
    previous_updates = policy.updates if policy is not None else 0
    while True:
        if brick is not None:
            brick.camera(*canonical_camera(frame, detector))
        action, _ = bridge.choose()
        result = camera.step(int(action))
        requests.append(action)
        elapsed += result.elapsed_seconds
        score += result.reward
        if brick is not None:
            brick.observe_score(result.reward, terminated=result.terminated, truncated=result.truncated)
        bridge.observe(result)
        frame = result.frame
        if result.terminated or result.truncated:
            row = camera.summary()
            row.update(score=int(score), elapsed_seconds=elapsed, cameras=len(requests),
                       truncated=result.truncated, shots=[] if brick is None else brick.shots,
                       score_updates=0 if policy is None else policy.updates - previous_updates,
                       action_sha256=hashlib.sha256(requests).hexdigest())
            return row, controller.timing.state()


def run_arm(root, phase, cases, layouts, agent_seeds, cap, lag, checkpoint, trained_state, arm):
    from pong_role_transfer import RoleAdapter

    torch.set_num_threads(1)
    policy = None
    if arm == 'brick_live':
        policy = BrickShotPolicy()
    elif arm == 'brick_prior':
        policy = BrickShotPolicy(updates_enabled=False)
    elif arm == 'brick_trained':
        policy = BrickShotPolicy.from_state(trained_state, updates_enabled=False)
    timing = checkpoint['timing']
    detector = RoleAdapter()
    rows = []
    initial_shot_state = None if policy is None else score_digest(policy.to_state())
    for case, layout, seed in zip(cases, layouts, agent_seeds, strict=True):
        start_timing = timing if phase == 'practice' else checkpoint['timing']
        start_role = detector.state()
        row, timing = play_life(case, layout, checkpoint, start_timing, detector, arm, policy, seed,
                                cap=cap, lag=lag)
        row.update(case=asdict(case), policy_seed=seed,
                   initial_timing=start_timing, final_timing=timing,
                   initial_role=start_role, final_role=detector.state())
        rows.append(row)
        save(root / f'{phase}-{arm}.json', rows)
    state = None if policy is None else policy.to_state()
    report = dict(total=summarize(rows), chronology=[summarize(rows[i:i + 12])
                   for i in range(0, len(rows), 12)], final_shot_state=state,
                  initial_shot_sha256=initial_shot_state,
                  final_shot_sha256=None if state is None else score_digest(state))
    save(root / f'{phase}-{arm}-summary.json', report)
    return report


def matched_differences(root, phase):
    trained = json.loads((root / f'{phase}-brick_trained.json').read_text())
    rng = np.random.default_rng(67604816)
    indices = rng.integers(len(trained), size=(4000, len(trained)))

    def vector(rows, metric):
        return np.array([row[metric] for row in rows], dtype=float)

    differences = {}
    for arm in ('brick_prior', 'center', 'pong'):
        base = json.loads((root / f'{phase}-{arm}.json').read_text())
        if [row['case'] for row in trained] != [row['case'] for row in base]:
            raise ValueError('held-out comparisons require matched environment cases')
        a, b = (vector(rows, 'post_own_bricks') for rows in (trained, base))
        at, bt = (vector(rows, 'elapsed_seconds') for rows in (trained, base))
        life = a - b
        boot_life = life[indices].mean(axis=1)
        boot_minute = 60. * (a[indices].sum(axis=1) / at[indices].sum(axis=1)
                             - b[indices].sum(axis=1) / bt[indices].sum(axis=1))
        differences[arm] = dict(
            bricks_after_own_return_per_life=float(life.mean()),
            per_life_bootstrap_95=np.quantile(boot_life, [.025, .975]).tolist(),
            bricks_after_own_return_per_minute=float(60. * (a.sum() / at.sum() - b.sum() / bt.sum())),
            per_minute_bootstrap_95=np.quantile(boot_minute, [.025, .975]).tolist())
    return differences


def run(root, checkpoint_path, *, practice_lives, heldout_lives, duration,
        workers, mode, trained_path=None, bricks=None):
    import breakout_arena
    import breakout_brick_placement
    import pong_role_transfer

    if (practice_lives < 1 or heldout_lives < 1 or not np.isfinite(duration) or duration < 1.
            or workers < 1 or (bricks is not None and not 2 <= bricks <= 8)
            or mode == 'heldout' and trained_path is None):
        raise ValueError('a positive experiment budget and a trained state for held-out mode are required')
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    phases = {'practice': ('practice', practice_lives, PRACTICE_ARMS),
              'heldout': ('heldout', heldout_lives, TEST_ARMS),
              'heldout_delay': ('heldout', heldout_lives, TEST_ARMS)}
    chosen = tuple(phase for phase in phases if mode in ('all', phases[phase][0]))
    root.mkdir(parents=True, exist_ok=False)
    environments = {
        phase: breakout_arena.breakout_schedule(SCHEDULES[phase][0], phases[phase][1],
                                                split=phases[phase][0])
        for phase in chosen
    }
    layouts = {phase: [None if bricks is None else sparse_layout(case, bricks) for case in cases]
               for phase, cases in environments.items()}
    seeds = {
        phase: np.random.default_rng([715952263, SCHEDULES[phase][0]])
                 .integers(1, 2**31, size=len(cases)).tolist()
        for phase, cases in environments.items()
    }
    source_paths = [Path(__file__), Path(breakout_arena.__file__),
                    Path(breakout_brick_placement.__file__), Path(pong_role_transfer.__file__)]
    save(root / 'spec.json', dict(
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(), duration=duration,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        schedules={phase: dict(environment=[asdict(case) for case in cases],
                               environment_seed=SCHEDULES[phase][0], delay_frames=SCHEDULES[phase][1],
                               custom_layouts=[None if layout is None else layout.cells
                                               for layout in layouts[phase]],
                               agent_seeds=seeds[phase], arms=phases[phase][2])
                   for phase, cases in environments.items()},
        heldout_policy_frozen=True, heldout_timing_starts_shared=True,
        primary='bricks after an own return per matched life and minute of camera time'))
    state = None if trained_path is None else json.loads(trained_path.read_text())
    reports = {}
    with ProcessPoolExecutor(max_workers=workers) as workers_pool:
        for phase, cases in environments.items():
            jobs = {arm: workers_pool.submit(run_arm, root, phase, cases, layouts[phase],
                                             seeds[phase], duration, SCHEDULES[phase][1],
                                             checkpoint, state, arm)
                    for arm in phases[phase][2]}
            reports[phase] = {arm: task.result() for arm, task in jobs.items()}
            if phase == 'practice':
                state = reports[phase]['brick_live']['final_shot_state']
                save(root / 'brick-checkpoint.json', state)
            else:
                reports[phase]['matched_differences'] = matched_differences(root, phase)
            save(root / 'summary.json', reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pong-checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--brick-checkpoint', type=Path)
    parser.add_argument('--practice-lives', type=int, default=48)
    parser.add_argument('--heldout-lives', type=int, default=24)
    parser.add_argument('--duration', type=float, default=18.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bricks-per-layout', type=int, choices=range(2, 9))
    parser.add_argument('--mode', choices=('all', 'practice', 'heldout'), default='all')
    args = parser.parse_args()
    run(args.output, args.pong_checkpoint, practice_lives=args.practice_lives,
        heldout_lives=args.heldout_lives, duration=args.duration, workers=args.workers,
        mode=args.mode, trained_path=args.brick_checkpoint, bricks=args.bricks_per_layout)

"""Run a small matched full-board and sparse-board visual rebound pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import process_time
from types import SimpleNamespace

import numpy as np
import torch

from breakout_arena import BreakoutLayout, breakout_schedule
from breakout_visual_rebound import VisualReboundPlacement, VisualReboundPolicy, centered_forecast
from pong_carried_shot_value import ShotMemory, build_carried_shot_controller
from pong_score_placement import ScorePlacement
from run_breakout_brick_placement import CHECKPOINT, save, summarize


ENVIRONMENT_SEEDS = {'practice': (1178373443, 2035816627), 'test': (916374201, 1513611593)}
AGENT_SEED = 240791443
TEST_ARMS = ('visual_trained', 'visual_empty', 'center')


def schedules(practice_lives, test_lives):
    """Construct distinct boards before any controller sees a camera."""
    return board_cases((('practice', practice_lives), ('test', test_lives)), ENVIRONMENT_SEEDS,
                       agent_seed=AGENT_SEED, seen=set())


def board_cases(phase_counts, environment_seeds, *, agent_seed, seen):
    agent_rng = np.random.default_rng(agent_seed)
    result = {}
    for phase, count in phase_counts:
        split = 'practice' if phase == 'practice' else 'heldout'
        groups = []
        for board, seed in zip(('full', 'sparse'), environment_seeds[phase], strict=True):
            cases = []
            for case in breakout_schedule(seed, count // 2, split=split):
                layout_seed = case.layout_seed
                while True:
                    cells = BreakoutLayout.generate(layout_seed, split=split).cells
                    if board == 'sparse':
                        selection = np.random.default_rng(layout_seed).choice(len(cells), 2, replace=False)
                        cells = tuple(sorted(cells[index] for index in selection))
                    if cells not in seen:
                        seen.add(cells)
                        break
                    layout_seed += 1
                cases.append(dict(episode=case, layout=BreakoutLayout(cells), board=board,
                                  agent_seed=int(agent_rng.integers(1, 2**31))))
            groups.append(cases)
        result[phase] = [case for match in zip(*groups, strict=True) for case in match]
    return result


def play(case, checkpoint, policy, duration, lag_frames=0):
    from pong_role_transfer import RoleAdapter, RoleController

    episode = case['episode']
    camera = episode.camera(max_episode_ticks=round(duration * 120), layout=case['layout'], delay_frames=lag_frames)
    seed = case['agent_seed']
    pong_score = ScorePlacement(**checkpoint['score'], updates_enabled=False)
    controller = build_carried_shot_controller(
        seed, checkpoint['motor'], checkpoint['timing'], pong_score,
        memory=ShotMemory(**checkpoint['shot_memory']), timing_updates=True)
    if policy is None:
        controller.timing.contact_offset = centered_forecast
        placement = None
    else:
        placement = VisualReboundPlacement(controller.timing, policy, seed)
    detector = RoleAdapter()
    bridge = RoleController(controller, detector)
    frame = camera.frame()
    bridge.reset(frame)
    controller_started = None
    requests = bytearray()
    start_updates = 0 if policy is None else policy.updates
    while not camera.needs_reset:
        if placement is not None:
            placement.camera(frame, detector)
        request, _ = bridge.choose()
        if bridge.engaged and controller_started is None:
            controller_started = detector.seconds
        transition = camera.step(int(request))
        requests.append(request)
        public = SimpleNamespace(frame=transition.frame, reward=transition.reward,
                                 elapsed_seconds=transition.elapsed_seconds,
                                 terminated=transition.terminated, truncated=transition.truncated)
        bridge.observe(public)
        frame = public.frame
        if placement is not None:
            placement.observe(detector.seconds, detector.canonical_entities, detector.pair_ids,
                              public.reward, terminated=public.terminated, truncated=public.truncated)
    result = camera.summary()
    rebounds = [] if placement is None else placement.observed_rebounds
    # Reconcile the visual estimator only after the life ends. These labels
    # never enter the controller, angle posterior, or score calibration.
    true_contacts = [contact for contact in result['contacts'] if contact['success']]
    matches = []
    for rebound in rebounds:
        matched = min(true_contacts, key=lambda contact: abs(contact['time_seconds'] - rebound['seconds']),
                      default=None)
        if matched is not None and abs(matched['time_seconds'] - rebound['seconds']) < .14:
            matches.append(dict(contact=matched['camera'], time_error=rebound['seconds'] - matched['time_seconds'],
                                offset_error=rebound['contact_offset'] - matched['offset']))
    return dict(result, elapsed_seconds=result['elapsed'], case=asdict(episode), board=case['board'],
                layout=asdict(case['layout']), agent_seed=seed, visual_rebounds=rebounds, matches=matches,
                controller_started_seconds=controller_started,
                score_updates=0 if policy is None else policy.updates - start_updates,
                shots=[] if placement is None else placement.shots,
                action_sha256=hashlib.sha256(requests).hexdigest())


def run_arm(output, checkpoint, phase, arm, cases, duration, trained=None, lag_frames=0):
    began = process_time()
    torch.set_num_threads(1)
    policies = {'visual_live': lambda: VisualReboundPolicy(),
                'visual_trained': lambda: VisualReboundPolicy.from_state(trained, updates_enabled=False),
                'visual_empty': lambda: VisualReboundPolicy(updates_enabled=False), 'center': lambda: None}
    policy = policies[arm]()
    initial = None if policy is None else policy.to_state()
    rows = []
    for case in cases:
        rows.append(play(case, checkpoint, policy, duration, lag_frames))
        save(output / f'{phase}-{arm}.json', rows)
    report = {}
    for board in ('all', 'full', 'sparse'):
        subset = [row for row in rows if board in ('all', row['board'])]
        matches = [match for row in subset for match in row['matches']]
        report[board] = dict(summarize(subset), visual_rebounds=sum(len(row['visual_rebounds']) for row in subset),
                             matched_rebounds=len(matches), offset_rmse=None if not matches else float(
                                 np.sqrt(np.mean([match['offset_error']**2 for match in matches]))))
    final = None if policy is None else policy.to_state()
    if arm != 'visual_live' and final != initial:
        raise ValueError('test shot-state updates are disabled')
    return dict(report, initial_state=initial, final_state=final, cpu_seconds=process_time() - began)


def paired_lifts(output):
    rows = {arm: json.loads((output / f'test-{arm}.json').read_text()) for arm in TEST_ARMS}
    result = {}
    for control in TEST_ARMS[1:]:
        matched = tuple(zip(rows['visual_trained'], rows[control], strict=True))
        if any(a['case'] != b['case'] or a['layout'] != b['layout'] or a['agent_seed'] != b['agent_seed']
               for a, b in matched):
            raise ValueError('pilot arms require identical matched cases and agent seeds')
        result[control] = {}
        for board in ('all', 'full', 'sparse'):
            pairs = [(a, b) for a, b in matched if board in ('all', a['board'])]
            a, b = (summarize(group) for group in zip(*pairs, strict=True))
            result[control][board] = {
                output_key: a[input_key] - b[input_key] if a[input_key] is not None
                and b[input_key] is not None else None
                for output_key, input_key in (
                    ('bricks_per_life', 'bricks_after_own_return_per_life'),
                    ('bricks_per_minute', 'bricks_after_own_return_per_minute'),
                    ('return_fraction', 'return_fraction'))}
    return result


def run(output, checkpoint_path, practice_lives, test_lives, duration, workers=2):
    if any(count < 4 or count % 4 for count in (practice_lives, test_lives)) or duration <= 0. or workers < 1:
        raise ValueError('each phase requires a positive multiple of four lives, duration and worker count')
    checkpoint_data = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_data)
    cases = schedules(practice_lives, test_lives)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'spec.json', dict(
        checkpoint_sha256=hashlib.sha256(checkpoint_data).hexdigest(), duration=duration,
        phase_cases={phase: [dict(case, episode=asdict(case['episode']), layout=asdict(case['layout']))
                             for case in rows] for phase, rows in cases.items()},
        test_arms=TEST_ARMS, train_on='visible inbound/outbound paddle-rail turns and public brick score',
        primary='visual_trained vs visual_empty on post-return bricks per life and minute',
        centered_forecast=True, public_only_transition=True))
    practice = run_arm(output, checkpoint, 'practice', 'visual_live', cases['practice'], duration)
    save(output / 'practice-summary.json', practice)
    with ProcessPoolExecutor(max_workers=min(workers, len(TEST_ARMS)), mp_context=get_context('spawn')) as pool:
        jobs = {arm: pool.submit(run_arm, output, checkpoint, 'test', arm, cases['test'], duration,
                                 practice['final_state']) for arm in TEST_ARMS}
        test = {arm: task.result() for arm, task in jobs.items()}
    save(output / 'summary.json', dict(practice=practice, test=test, matched_lifts=paired_lifts(output)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--practice-lives', type=int, default=12)
    parser.add_argument('--test-lives', type=int, default=12)
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.checkpoint, args.practice_lives, args.test_lives, args.duration, args.workers)

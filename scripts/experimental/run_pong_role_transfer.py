"""Measure camera role discovery and actual returns on matched rotated Pong starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import torch

from fly_connectome.control.environment import PongCamera
from pong_carried_shot_value import ShotMemory, build_carried_shot_controller
from pong_role_transfer import NATIVE_PONG_BINDING, PIXEL, RoleAdapter, RoleBinding, RoleController
from pong_score_placement import ScorePlacement
from pong_transfer_arena import play
from return_metrics import ReturnCounter
from run_pong_score_placement import retention_difference, write


EXPECTED_BINDINGS = (
    NATIVE_PONG_BINDING,
    RoleBinding.from_station(1, .453125),
    RoleBinding.from_station(0, .453125),
    RoleBinding.from_station(1, -.453125),
)
POSITION_TOLERANCE = 2 * PIXEL


def distractor_mask(seed):
    """Fixed spectator silhouettes, kept off the paddle rails."""
    rng = np.random.default_rng(seed)
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for shape, (y0, x0) in zip(((2, 2), (2, 3), (2, 7), (6, 1)),
                               ((14, 14), (14, 39), (39, 14), (39, 39)), strict=True):
        y, x = rng.integers(8, size=2) + (y0, x0)
        height, width = shape
        mask[y:y + height, x:x + width] = True
    return mask


def make_controller(state, policy_seed):
    return build_carried_shot_controller(
        policy_seed, state['motor'], state['timing'],
        ScorePlacement(**state['score'], updates_enabled=False),
        memory=ShotMemory(**state['shot_memory']), timing_updates=True,
    )


def audit_pair(game, detector):
    """Check published positions using physics, exclusively in the evaluator."""
    if detector.pair_ids is None:
        return None
    lookup = {row['id']: row for row in detector.canonical_entities}
    paddle, ball = (lookup[identity] for identity in detector.pair_ids)
    own_position = (float(game.body.position[0]) - .5, game.config.paddle_x - .5)
    ball_position = (float(game.ball[0, 1]) - .5, float(game.ball[0, 0]) - .5)
    errors = [math.dist(row['normalized_position'], expected) if row['observed'] else None
              for row, expected in ((paddle, own_position), (ball, ball_position))]
    return dict(ids=list(detector.pair_ids), errors=errors,
                correct=all(error is not None and error <= POSITION_TOLERANCE for error in errors))


def trial(state, seed, policy_seed, distractor_seed, rotation, arm, seconds, opponent_speed):
    if arm == 'native':
        return play(make_controller(state, policy_seed), seed,
                    opponent_speed=opponent_speed, duration=seconds)

    camera = PongCamera(seed, max_episode_ticks=round(120 * seconds), opponent_speed=opponent_speed)
    game, counter = camera.game, ReturnCounter()
    original_step = game.step
    game.step = lambda drive, **kwargs: counter.step(game, original_step, drive, **kwargs)
    mask = distractor_mask(distractor_seed)

    def capture(frame):
        return frame.rot90(rotation) | mask

    controller = None
    if arm in ('online', 'frozen'):
        detector = RoleAdapter(arm, NATIVE_PONG_BINDING if arm == 'frozen' else None)
        controller = RoleController(make_controller(state, policy_seed), detector)
        controller.reset(capture(game.render(64, 64)[0]))

    contacts, requests = [], bytearray()
    first_ready, first_pair = None, None
    pair_samples = dict(visible=0, correct=0, unavailable=0)
    while not camera.needs_reset:
        if controller and first_ready is None and detector.ready:
            first_ready = camera.total_ticks / 120
            first_pair = audit_pair(game, detector)
        if controller and len(requests) % 10 == 0:
            pair = audit_pair(game, detector)
            if pair is not None and all(error is not None for error in pair['errors']):
                pair_samples['visible'] += 1
                pair_samples['correct'] += int(pair['correct'])
            else:
                pair_samples['unavailable'] += 1
        request = controller.choose()[0] if controller else 1
        requests.append(request)
        before = counter.opportunities
        transition = camera.step(request)
        if controller:
            controller.observe(replace(transition, frame=capture(transition.frame)))
        if counter.opportunities > before:
            contacts.append((camera.total_ticks / 120, bool(transition.hits)))

    first = counter.first_incoming
    first_trajectory = None if first is None else {k: v for k, v in first.items() if k != 'success'}
    result = dict(seed=seed, rotation=rotation, hits=counter.returns,
                  opportunities=counter.opportunities, contacts=contacts,
                  own_hit_win=int(transition.reward > 0 and counter.returns > 0),
                  score=int(transition.reward), elapsed=camera.total_ticks / 120,
                  cameras=len(requests), first_trajectory=first_trajectory,
                  action_sha256=hashlib.sha256(requests).hexdigest())
    if controller:
        result.update(binding=detector.state(), binding_seconds=detector.binding_seconds,
                      correct_binding=detector.binding == EXPECTED_BINDINGS[rotation],
                      first_policy_seconds=first_ready, probe_cameras=detector.probe_count,
                      first_pair=first_pair, pair_samples=pair_samples)
    return result


def summarize(rows):
    contact_rows = [row for row in rows if row['contacts']]
    bound = [row['binding_seconds'] for row in rows if row.get('binding_seconds') is not None]
    policy = [row['first_policy_seconds'] for row in rows if row.get('first_policy_seconds') is not None]
    return dict(starts=len(rows), first_returns=sum(row['contacts'][0][1] for row in contact_rows),
                first_opportunities=len(contact_rows), returns=sum(row['hits'] for row in rows),
                opportunities=sum(row['opportunities'] for row in rows),
                later_returns=sum(hit for row in rows for _, hit in row['contacts'][1:]),
                later_opportunities=sum(len(row['contacts']) - 1 for row in contact_rows),
                own_hit_wins=sum(row['own_hit_win'] for row in rows),
                losses=sum(row['score'] < 0 for row in rows),
                discovered=sum(row.get('binding') is not None for row in rows),
                correct_bindings=sum(row.get('correct_binding', False) for row in rows),
                correct_first_pairs=sum(bool(row.get('first_pair') and row['first_pair']['correct'])
                                        for row in rows),
                pair_samples={name: sum(row.get('pair_samples', {}).get(name, 0) for row in rows)
                              for name in ('visible', 'correct', 'unavailable')},
                median_binding_seconds=median(bound) if bound else None,
                median_first_policy_seconds=median(policy) if policy else None)


def evaluate(job):
    arm, rotation, state, case, seconds, opponent_speed = job
    torch.set_num_threads(1)
    row = trial(state, case['serve'], case['policy'], case['distractor'], rotation, arm,
                seconds, opponent_speed)
    return arm, rotation, row


def run(out, checkpoint, schedule, *, seconds, opponent_speed, workers):
    schedule_data = schedule.read_bytes()
    cases = json.loads(schedule_data)
    rotations = range(4)
    if isinstance(cases, dict):
        rotations = cases.get('rotations', rotations)
        cases = [dict(serve=case['environment_seed'], policy=case['policy_seed'],
                      distractor=case['distractor_seed']) for case in cases['starts']]
    if not cases or len({case['serve'] for case in cases}) != len(cases):
        raise ValueError('nonempty, unique held-out starts required')
    if not rotations or any(rotation not in range(4) for rotation in rotations):
        raise ValueError('at least one legal camera rotation required')
    checkpoint_data = checkpoint.read_bytes()
    state = json.loads(checkpoint_data)
    jobs = [(arm, rotation, state, case, seconds, opponent_speed) for case in cases
            for rotation in rotations for arm in ('online', 'frozen', 'passive', 'native')
            if arm != 'native' or rotation == 0]
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for arm, rotation, row in pool.map(evaluate, jobs):
            rows.setdefault(rotation, {}).setdefault(arm, []).append(row)

    summary, comparisons = {}, {}
    for rotation, arms in rows.items():
        signatures = [[row.get('first_trajectory') for row in arms[arm]]
                      for arm in ('online', 'frozen', 'passive')]
        if not signatures[0] == signatures[1] == signatures[2]:
            raise AssertionError('matched starts had different first incoming trajectories')
        for group in arms.values():
            if any(row['hits'] != sum(hit for _, hit in row['contacts']) for row in group):
                raise AssertionError('physics returns disagree with contact annotations')
        summary[rotation] = {arm: summarize(group) for arm, group in arms.items()}
        comparisons[rotation] = {arm: retention_difference(arms['online'], group)
                                 for arm, group in arms.items() if arm != 'online'}
    for arms in rows.values():
        for group in arms.values():
            for index, row in enumerate(group):
                del row['seed']
                row['matched_start'] = index
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'summary.json', dict(conditions=summary, compared_with_online=comparisons,
                    terminal_seconds=seconds, opponent_speed=opponent_speed,
                    first_trajectories_match=True, position_tolerance_pixels=POSITION_TOLERANCE / PIXEL,
                    schedule_sha256=hashlib.sha256(schedule_data).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(checkpoint_data).hexdigest()))
    write(out / 'rows.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'artifacts/m1b/2026-11-19-pong-after-mixed-course.json'))
    parser.add_argument('--schedule', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--opponent-speed', type=float, default=.65)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.out, args.checkpoint, args.schedule, seconds=args.seconds,
        opponent_speed=args.opponent_speed, workers=args.workers)


if __name__ == '__main__':
    main()

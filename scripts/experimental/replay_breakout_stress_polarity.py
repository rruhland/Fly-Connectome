"""Development replay of a single stress course with its pursuit polarity held fixed."""

import argparse
from collections import deque
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from multiprocessing import Pool
import subprocess

import torch

from arcade_transfer_session import CHECKPOINT, TransferSession
from arcade_transfer_schedule import canonical_json
from breakout_arena import BreakoutCamera
from fly_connectome.pong import Body

SOURCE_COMMIT = '54903c9f18c161024fb1dfa568a479eb3185a69f'
REFERENCE_SHA256 = '790dc00be41c497f4a9667e771fcf0d7d5aaeeaf077f6c245263b98fb590bb8d'
SCHEDULE_SHA256 = 'a6db895ceb2bcea20b1cac1b9f74ed0f5344caec105d823e72388c4136ca41bf'
CHECKPOINT_SHA256 = 'd09f1050ad2e5f04fbe12572d506b55cc30bb765db040720607633076c8ecb0e'


def read_pinned(path, expected):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f'input differs from the frozen experiment: {path}')
    return json.loads(contents)


def launch(case, schedule):
    return BreakoutCamera(case['environment_seed'], layout_seed=case['layout_seed'],
                          serve_side=case['serve_side'], layout_split=schedule['layout_split'],
                          distractors=schedule['distractors'], max_episode_ticks=schedule['max_life_ticks'],
                          **schedule['conditions'][case['condition']])


def reachable(arena, event):
    """Compute reachable paddle interval at evaluator-only plane-crossing tick."""
    target = event['ball_x']
    limits = []
    for action in (2, 0):
        body = Body(1, arena.config)
        for name in ('position', 'activation', 'velocity'):
            getattr(body, name).copy_(getattr(arena.body, name))
        queue = deque(arena._queue)
        decision = arena.decisions
        ticks = arena.ticks
        while ticks < event['tick']:
            budget = (decision + 1) * 12 // 5 - decision * 12 // 5
            queue.append(action)
            drive = arena._motor[queue.popleft()]
            for _ in range(min(event['tick'] - ticks, budget)):
                body.step(drive)
                ticks += 1
            decision += 1
        limits.append(float(body.position[0]))
    distance = max(limits[0] - target, target - limits[1], 0.)
    return dict(left=limits[0], right=limits[1], shortfall=max(0., distance - .1125),
                possible=distance <= .1125)


def run(locked, output, checkpoint, saved, schedule):
    torch.set_num_threads(1)
    session = TransferSession('full', saved['policy_seed'], checkpoint)
    rows = []
    for case in saved['lives']:
        initial_state = session.state()
        if not locked:
            assert hashlib.sha256(canonical_json(initial_state)).hexdigest() == case['initial_state_sha256']
        arena = launch(case, schedule)
        actor = session.begin_life()
        actor.reset(arena.frame())
        event = case['contacts'][0]
        initial_reach = reachable(arena, event)
        expected_order = [arena.mapping.index(p) for p in (0, 1, 2)]
        ready = first_correct_order = first_correct_lag = None
        first_right_request = None
        role_reach = None
        first = None
        requests = bytearray()
        passage = []
        while not arena.needs_reset:
            action, info = actor.choose()
            model = actor.controller.timing
            frame = arena.decisions + 1
            correct_binding = (actor.detector.ready and actor.detector.binding.axis == 1
                               and abs(actor.detector.binding.station - .45) < 1.5 / 64)
            paired = actor.detector.pair_ids
            tracks = {r['id']: r for r in actor.detector.canonical_entities}
            if paired is not None and correct_binding:
                paddle, ball = (tracks[p] for p in paired)
                coordinates = [paddle['normalized_position'], ball['normalized_position']]
            else:
                coordinates = None
            settled = model.effective_requests
            confidence = float(model.belief.probabilities[model.belief.confirmed])
            if not arena.contacts:
                if correct_binding and ready is None:
                    ready = frame
                    role_reach = reachable(arena, event)
                if actor.bridge.engaged and first_correct_order is None and settled == expected_order:
                    first_correct_order = frame
                if actor.bridge.engaged and first_correct_lag is None and abs(model.delay_commands - arena.delay_frames) <= 1:
                    first_correct_lag = frame
                if actor.bridge.engaged and first_right_request is None and arena.mapping[action] == (0 if case['serve_side'] > 0 else 2):
                    first_right_request = frame
                reading = dict(frame=frame, action=action, intended=info.get('intended'), mapped=arena.mapping[action],
                               estimated_order=settled, expected_order=expected_order, confidence=confidence,
                               estimated_lag=model.delay_commands, true_delay=arena.delay_frames,
                               ready=correct_binding, camera_coordinates=coordinates,
                               paddle_x=float(arena.body.position[0]), ball_x=arena.ball[0], role=actor.detector.state(),
                               elapsed_seconds=arena.ticks / 120,
                               arrival=asdict(model.arrival_context) if model.arrival_context else None,
                               score_choices=[asdict(choice) for choice in actor.controller.placement.choices])
                if frame % 5 == 0:
                    passage.append(reading)
            step = arena.step(action)
            requests.append(action)
            actor.observe(replace(step, hits=0))
            if first is None and arena.contacts:
                first = dict(event=arena.summary()['first_incoming'], last_decision=reading)
        summary = arena.summary()
        session.end_life()
        before_lock = session.polarity
        if locked:
            session.polarity = checkpoint['interception']['choices'][4]
        action_hash = hashlib.sha256(requests).hexdigest()
        if not locked:
            assert hashlib.sha256(canonical_json(session.state())).hexdigest() == case['final_state_sha256']
            assert action_hash == case['action_sha256']
        assert (first['event']['ball_x'], first['event']['tick']) == (event['ball_x'], event['tick'])
        rows.append(dict(id=case['id'], condition=case['condition'], side=case['serve_side'],
                         polarity=initial_state['polarity'], proposed_next_polarity=before_lock,
                         ready=ready, opening=first, initial_reach=initial_reach, role_reach=role_reach,
                         first_correct_order=first_correct_order, first_within_one_lag=first_correct_lag,
                         first_right_request=first_right_request, later_contacts=summary['contacts'][1:],
                         total_returns=summary['own_returns'], score=summary['score'],
                         initial_timing=initial_state['timing'], end_timing=session.timing,
                         action_sha256=action_hash, passage=passage))
    (output / ('polarity-held.json' if locked else 'exact-replay.json')).write_text(
        json.dumps(rows, indent=2))
    return len(rows), sum(row['opening']['event']['success'] for row in rows), sum(row['total_returns'] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='original full-stress-01.json from the blind qualification')
    parser.add_argument('--schedule', type=Path, required=True)
    args = parser.parse_args()

    repo = CHECKPOINT.parents[2]
    source = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=repo, text=True).strip()
    changed = subprocess.check_output(('git', 'status', '--porcelain', '--', 'src',
                                       'scripts/experimental'), cwd=repo, text=True).strip()
    if source != SOURCE_COMMIT or changed:
        raise ValueError(f'use the untouched frozen candidate at {SOURCE_COMMIT} on PYTHONPATH')

    checkpoint = read_pinned(CHECKPOINT, CHECKPOINT_SHA256)
    saved = read_pinned(args.reference, REFERENCE_SHA256)
    schedule = read_pinned(args.schedule, SCHEDULE_SHA256)
    args.output.mkdir(parents=True, exist_ok=False)
    with Pool(2) as pool:
        summaries = pool.starmap(run, [(locked, args.output, checkpoint, saved, schedule)
                                        for locked in (False, True)])
    print(dict(original=summaries[0], held=summaries[1]))


if __name__ == '__main__':
    main()

"""Verify and render a played point from a physical arm transfer receipt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_replay import caption, recording
from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Arena, TT6Wiring


def replay(path, output, *, course_index, condition, point_index):
    receipt = _read(path)
    course = receipt['courses'][course_index]
    point = course['runs'][condition]['points'][point_index]
    actions = np.asarray(point['actions'], dtype='<f8')
    if sha256(actions.tobytes()).hexdigest() != point['action_sha256']:
        raise ValueError('recorded public commands failed their checksum')
    schedule = course['schedule']
    arena = TT6Arena(schedule['arena_seed'], view_seed=schedule['view_seed'], body=course['body'],
                     **receipt['geometry']['arena'])
    try:
        wiring = TT6Wiring(arena, schedule['arena_seed'])
        title = f'{arena.action_size}-joint {course["body"]} arm  ·  First-person camera'
        status = 'Opponent serves'
        frames = wiring.reset(seed=point['seed'])
        with recording(output, caption(frames[0], title, status, 0), arena.control_dt) as append:
            for action in actions:
                frames, reward, ended, timed_out, info = wiring.step(action)
                for event in info['events']:
                    if event['kind'] == 'legal_return':
                        player = 'Arm' if event['side'] == 1 else 'Opponent'
                        status = f'{player} lands a legal return'
                if ended:
                    status = 'Arm wins the point' if reward > 0 else 'Opponent wins the point'
                elif timed_out:
                    status = 'Time limit reached'
                append(caption(frames[0], title, status, arena.legal_returns))
            if arena.point_result() != point['evaluator'] or json.dumps(arena.events) != json.dumps(point['events']):
                raise ValueError('native replay diverged from the recorded physical events')
        return arena.point_result()
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--course', type=int, default=1)
    parser.add_argument('--condition', default='shared')
    parser.add_argument('--point', type=int, default=1)
    args = parser.parse_args()
    print(replay(args.receipt, args.output, course_index=args.course, condition=args.condition, point_index=args.point))

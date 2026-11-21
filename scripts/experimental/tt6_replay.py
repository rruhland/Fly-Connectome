"""Verify played arm-change points from command tapes and render their original first-person view."""
import argparse
import json
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

from .tt4_replay import caption, recording
from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt6_timing import DelayedWiring


def replay(path, output=None, visit_index=0, point_index=0, *, prepare=None):
    doc = _read(path)
    if prepare is None:
        from .tt7_surface_trial import SURFACE_PROTOCOL, prepare_material

        prepare = prepare_material if doc.get('protocol') == SURFACE_PROTOCOL else prepare_point
    checked = 0
    total_returns = 0
    body_names = {'stock':'Jointed arm', 'gantry':'Four-axis gantry', 'offset':'Five-axis arm', 'rotary':'Six-joint arm'}
    for i, visit in enumerate(doc['visits']):
        if output is not None and i != visit_index:
            continue
        station = visit['installation']
        arena = TT6Arena(station['arena_seed'], view_seed=station['view_seed'],
                         body=visit['body'], **doc['geometry'])
        wiring = TT6Wiring(arena, station['arena_seed'])
        try:
            for j, row in enumerate(visit['points']):
                if output is not None and j != point_index:
                    continue
                disturbance = prepare(arena, row['phase'])
                interface = DelayedWiring(wiring, disturbance, seed=doc['seed'])
                movie = output is not None
                imgs = interface.reset(seed=row['seed'], render=movie)
                title = body_names[visit['body']] + '  ·  First-person camera'
                status = 'Opponent serves'
                actions = np.asarray(row['actions'], dtype='<f8')
                if sha256(actions.tobytes()).hexdigest() != row['action_sha256']:
                    raise ValueError('policy command hash differs from the recorded value')
                sink = recording(output, caption(imgs[0],title,status,0), arena.control_dt) if movie else nullcontext()
                with sink as append:
                    for action in actions:
                        imgs, _, ended, timed_out, info = interface.step(action, render=movie)
                        if movie:
                            for event in info['events']:
                                if event['kind']=='legal_return':
                                    who='Arm' if event.get('side') == 1 else 'Opponent'
                                    status=who+' lands a legal return'
                            if ended:
                                status='Arm wins' if arena.reward>0 else 'Opponent wins'
                            elif timed_out:
                                status='Time limit reached'
                            append(caption(imgs[0],title,status,arena.legal_returns))
                        else:
                            mujoco.mj_kinematics(arena.model, arena.data)
                    if arena.point_result() != row['evaluator']:
                        raise ValueError(f'physical result diverged at visit {i}, point {j}')
                    if json.loads(json.dumps(arena.events)) != row['events']:
                        raise ValueError(f'event history diverged at visit {i}, point {j}')
                    np.testing.assert_array_equal(interface.applied, row['applied'])
                checked += 1
                total_returns += row['player_returns']
        finally:
            arena.close()
    if not checked:
        raise ValueError('no played point matched the requested selection')
    print(f'Verified {checked} points and {total_returns} legal returns.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt',type=Path)
    p.add_argument('--movie',type=Path)
    p.add_argument('--visit',type=int,default=0)
    p.add_argument('--point',type=int,default=0)
    a=p.parse_args()
    replay(a.receipt,a.movie,a.visit,a.point)

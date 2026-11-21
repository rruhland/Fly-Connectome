"""Render a complete first-person game from the saved physical action tapes."""

import argparse
from contextlib import ExitStack
from pathlib import Path

from .tt4_camera import PerspectiveCamera
from .tt4_replay import caption, recording
from .tt4_transfer_trial import _hash, _read
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt14_gain_hardware import GainWiring
from .tt14_gain_trial import wiring_for


def movie(receipt, path, number, *, installation_seed=None):
    game = receipt['games'][number - 1]
    score = ServiceScore(first_server=-1)
    arena = TT8ServiceArena(**receipt['arena'])
    try:
        if 'camera' in receipt:
            arena._rig = tuple(PerspectiveCamera(**eye) for eye in receipt['camera'])
        cache_hall_stereo(arena, reflections=True)
        wires = {changed: wiring_for(arena, receipt, changed) for changed in (False, True)}
        if installation_seed is not None:
            wires[True] = GainWiring(arena, installation_seed, receipt['matrix'])
        with ExitStack() as session:
            append = None
            for row in game['points']:
                server = score.server
                if server != row['server']:
                    raise ValueError('service order differs from the saved tape')
                interface = PointInterface(wires[row['changed']], prepare_point(arena, row['phase']),
                                           server=server, seed=game['seed'])
                frames = interface.reset(seed=row['seed'])
                hardware = receipt.get('profile', receipt.get('case', 'Changed')).capitalize() if row['changed'] else 'Original controls'
                model = row.get('model_kind', receipt.get('label', 'Active')).capitalize()
                title = f'{hardware} / {model} map   |   Arm {score.score[0]}   Opponent {score.score[1]}'
                status = 'Arm serves' if server == 1 else 'Opponent serves'
                image = caption(frames[0], title, status, 0)
                if append is None:
                    append = session.enter_context(recording(path, image, arena.control_dt))
                else:
                    append(image)
                for action in row['actions']:
                    frames, reward, ended, capped, info = interface.step(action)
                    for event in info['events']:
                        if event['kind'] == 'legal_serve':
                            status = 'Legal serve'
                        elif event['kind'] == 'legal_return':
                            status = 'Arm returns' if event['side'] == 1 else 'Opponent returns'
                    if ended or capped:
                        score.record(arena.point_result())
                        title = f'{hardware} / {model} map   |   Arm {score.score[0]}   Opponent {score.score[1]}'
                        status = 'Arm wins the point' if reward > 0 else (
                            'Opponent wins the point' if reward < 0 else 'Let' if info['serve_let'] else 'Time limit')
                    append(caption(frames[0], title, status, arena.legal_returns))
                if arena.point_result() != row['evaluator'] or interface.applied != row['applied']:
                    raise ValueError(f"physical replay diverged on point {row['seed']}")
            if not score.finished or list(score.score) != game['score']:
                raise ValueError('game score differs from the saved game')
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--game', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    receipt = _read(args.receipt)
    installation = None
    if receipt.get('schema') == 'tt17-motor-v1':
        setup = args.receipt.parent / f"motor-{receipt['case']}-setup.json.gz"
        if _hash(setup) != receipt['setup_sha256']:
            raise ValueError('game setup differs from its saved hardware')
        installation = _read(setup)['wiring']
    movie(receipt, args.output, args.game, installation_seed=installation)

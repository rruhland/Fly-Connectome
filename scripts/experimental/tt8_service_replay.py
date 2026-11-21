"""Render the complete recorded game and check every physical point against its receipt."""
import argparse
from contextlib import ExitStack
from pathlib import Path

from .tt4_replay import caption, recording
from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import PointInterface


def movie(receipt, path):
    game = _read(receipt)
    if game['practice']:
        raise ValueError('a full game receipt is required')
    arena = TT8ServiceArena(**game['arena'])
    wiring = TT6Wiring(arena, game['installation_seed'])
    scoreboard = ServiceScore(game['race_to'], game['first_server'])
    try:
        with ExitStack() as context:
            append = None
            for row in game['points']:
                server = scoreboard.server
                if server != row['server']:
                    raise ValueError('server does not match the game score')
                interface = PointInterface(wiring, prepare_point(arena, row['phase']), server=server, seed=game['seed'])
                frames = interface.reset(seed=row['seed'])
                title = f"First-person table tennis   |   Arm {scoreboard.score[0]}   Opponent {scoreboard.score[1]}"
                status = 'Arm serves' if server == 1 else 'Opponent serves'
                image = caption(frames[0], title, status, 0)
                if append is None:
                    append = context.enter_context(recording(path, image, arena.control_dt))
                else:
                    append(image)
                for command in row['actions']:
                    frames, _, ended, capped, info = interface.step(command)
                    for event in info['events']:
                        if event['kind'] == 'serve_strike':
                            status = 'Arm strikes the descending toss'
                        elif event['kind'] == 'legal_serve':
                            status = 'Serve lands on both halves'
                        elif event['kind'] == 'legal_return':
                            status = 'Arm lands a return' if event['side'] == 1 else 'Opponent lands a return'
                    if ended or capped:
                        scoreboard.record(arena.point_result())
                        title = f"First-person table tennis   |   Arm {scoreboard.score[0]}   Opponent {scoreboard.score[1]}"
                        status = ('Arm wins the point' if info['points_for'] else 'Opponent wins the point'
                                  if info['points_against'] else 'Let' if info['serve_let'] else 'Time limit')
                    image = caption(frames[0], title, status, arena.legal_returns)
                    append(image)
                if arena.point_result() != row['evaluator'] or interface.applied != row['applied']:
                    raise ValueError(f"could not reproduce point {row['seed']}")
            if list(scoreboard.score) != game['score'] or scoreboard.finished != game['finished']:
                raise ValueError('score differs from receipt')
    finally:
        arena.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    movie(args.receipt, args.output)

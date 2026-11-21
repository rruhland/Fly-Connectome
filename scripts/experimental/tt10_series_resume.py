"""Verify that a saved series boundary reproduces the next live RGB decisions."""
import argparse
import json
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt10_series_trial import _restore
from .tt10_window_trial import _point_row


def verify(checkpoint, receipt, points):
    state = _read(checkpoint)['series']
    series = _read(receipt)
    if state['arena'] != series['arena']:
        raise ValueError('checkpoint and comparison refer to different installations')
    game = series['games'][state['completed']]
    arena = TT8ServiceArena(**state['arena'])
    wiring = TT6Wiring(arena, state['arena']['seed'])
    score = ServiceScore(first_server=game['first_server'])
    hashes = []
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player = _restore(checkpoint, arena)
        for point in game['points'][:points]:
            row = _point_row(player, arena, wiring, game['seed'], point['seed'], point['phase'], score.server)
            score.record(row['evaluator'])
            if json.loads(json.dumps(row)) != point:
                raise ValueError(f"live continuation diverged on point {row['seed']}")
            hashes.append(dict(point=row['seed'], action_sha256=row['action_sha256']))
    finally:
        arena.close()
    return dict(schema='tt10-series-resume-check-v1', checkpoint_sha256=_hash(checkpoint),
                series_sha256=_hash(receipt), previous_games=state['completed'],
                next_game=game['seed'], verified=hashes)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('receipt', type=Path)
    p.add_argument('--points', type=int, default=3)
    args = p.parse_args()
    if args.points < 1:
        p.error('verify at least one point')
    check = verify(args.checkpoint, args.receipt, args.points)
    _write(args.output, check)
    print(f"{len(check['verified'])} live point records reproduced")

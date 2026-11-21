"""Verify and render an actual sustained rotary rally from a contact receipt."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _read, _write
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_continuous_trial import point_condition
from .tt6_racket import set_racket_response
from .tt6_replay import replay
from .tt6_timing import DelayedWiring


def movie(receipt, output, condition, point_seed):
    source = _read(receipt)
    section = source['runs'][condition]
    rows = section['points'] if isinstance(section, dict) else section
    row = next(row.copy() for row in rows if row['seed'] == point_seed)
    if row['player_returns'] < 2 or not row['rival_returns']:
        raise ValueError('select a point with two legal player returns and an opponent reply')
    station = source['commissioning']['installation']
    geometry, seed = source['geometry'], source['seed']
    if 'applied' not in row:
        arena = TT6Arena(station['arena_seed'], body='rotary', view_seed=station['view_seed'], **geometry)
        try:
            rubber, disturbance = point_condition(row['phase'])
            set_racket_response(arena, rubber)
            interface = DelayedWiring(TT6Wiring(arena, station['arena_seed']), disturbance, seed=seed)
            interface.reset(seed=point_seed, render=False)
            for action in row['actions']:
                interface.step(action, render=False)
            row['applied'] = interface.applied
        finally:
            arena.close()
    normalized = dict(geometry=geometry, seed=seed, visits=[dict(installation=station, body='rotary', points=[row])])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        rec = Path(directory) / 'point.json.gz'
        _write(rec, normalized)
        replay(rec, output)
    return row['evaluator']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--point-seed', type=int, required=True)
    args = parser.parse_args()
    print(movie(args.receipt, args.output, args.condition, args.point_seed))

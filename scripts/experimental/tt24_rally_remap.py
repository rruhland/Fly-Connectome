"""Move a body decoder learned during paddle bouncing into live table tennis."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt23_sequence import verify
from .tt24_effects import transmission
from .tt24_bounce import BounceArena
from .tt24_rally_trial import COACHED, DATA, Station, archive_schedule, make_policy, returns
from .tt24_remap_game import measure


def driver(matrix, decoder, request):
    axes = matrix @ np.clip(decoder @ request[:5], -1., 1.)
    return np.r_[np.clip(axes, -1., 1.), request[5]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=8201333)
    parser.add_argument('--limit', type=int, default=9)
    args = parser.parse_args()
    eye = np.eye(5)
    rng = np.random.default_rng(args.seed)
    mix = transmission(rng)
    bounce = BounceArena(view='none')
    try:
        base, base_cost = measure(bounce, eye, args.seed)
        altered, altered_cost = measure(bounce, mix, args.seed + 1)
    finally:
        bounce.close()
    decoded = np.linalg.lstsq(altered, base, rcond=.01)[0]
    station = Station(_read(COACHED)['metadata']['opening']['far_profile'])
    with np.load(DATA) as data:
        policy, exposure = make_policy(data, 'demo_full')
    cases = []
    try:
        for start in archive_schedule()[:args.limit]:
            results = {}
            for mode, matrix, decoder in [('original', eye, eye), ('frozen', mix, eye),
                                          ('learned', mix, decoded), ('oracle', mix, np.linalg.inv(mix))]:
                results[mode] = station.play(start, 'demo_full', policy,
                                              execute=lambda u: driver(matrix, decoder, u))
            assert len({r['prefix_rgb'] for r in results.values()}) == 1
            cases.append(dict(seed=start['seed'], results=results))
            print(start['seed'], {k: len(returns(r, 1)) for k, r in results.items()}, flush=True)
    finally:
        station.arena.close()
    _write(args.output, dict(source='encoder effects observed on physical paddle-bounce drops',
                             base_cost=base_cost, altered_cost=altered_cost,
                             base_effect=base.tolist(), target_effect=altered.tolist(), decoder=decoded.tolist(),
                             wiring=mix.tolist(), exposure=exposure, cases=cases, verified=verify(cases)))


if __name__ == '__main__':
    main()

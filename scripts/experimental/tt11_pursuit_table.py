"""Test the moving-target game's fixed settling curve in ordinary table tennis."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt6_body_control import _response_bank
from .tt10_series_trial import run


def test_curve(site, folder, seed, games):
    state = _read(folder/f'timing-{site}-inherited-start.json.gz')
    model = state['models'][state['series']['model']]
    mean_curve = np.mean(_response_bank()[0], axis=0)
    delay = max(0., np.sum(1. - mean_curve) - .5)
    count = len(model['lags'])
    model.update(lags=[delay / (1. + delay)] * count, kernels=[mean_curve.tolist() for _ in range(count)])
    start = folder/f'pursuit-table-{site}-start.json.gz'
    _write(start, state)
    run(SimpleNamespace(resume=start, seed=seed, games=games, limit=65, mode='fresh',
                        checkpoint=folder/f'pursuit-table-{site}-player.json.gz',
                        output=folder/f'pursuit-table-{site}-games.json.gz'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site')
    p.add_argument('folder', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--games', type=int, default=2)
    args = p.parse_args()
    test_curve(args.site, args.folder, args.seed, args.games)

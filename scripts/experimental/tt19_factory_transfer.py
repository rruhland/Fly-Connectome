"""Ablate the learned gantry world in complete, zero-target-practice SO-101 games."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt19_factory_serve_trial import held
from .tt19_flight import FlightLaw


def run(folder, catalog_path, seeds):
    proposals_path = folder / 'factory-serve-proposals.json.gz'
    proposals = _read(proposals_path)
    # Select the first factory proposal without consulting any target practice.
    start = dict(factory_pick=0, seeds=[])
    with np.load(catalog_path) as archive:
        catalog = dict(archive)
    report = dict(catalog_sha256=_hash(catalog_path), proposals_sha256=_hash(proposals_path),
                  seeds=seeds, target_practice_points=0, prior='nominal', games=[])
    response = FlightMemory('impulse', (), (), frozen=True).response_matrix
    for seed in seeds:
        game = held(folder, proposals, start, catalog, FlightLaw(), response, seed, 'factory')
        report['games'].append(game)
        _write(folder / 'factory-serve-nominal.json.gz', report)
        print(seed, game['score'], 'returns', game['returns'], 'serves', game['serves'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[930477119, 1529880313])
    args = parser.parse_args()
    run(args.folder, args.catalog, args.seeds)

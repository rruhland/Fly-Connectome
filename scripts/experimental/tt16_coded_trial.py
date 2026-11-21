"""Capture a 12-tick RGB calibration on the TT16 transfer fixtures."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_render import cache_hall_stereo
from .tt16_coded_probe import decode, probe
from .tt16_transfer_protocol import FIXTURES, load_optical_player, make_arena, setup_path


def fit_path(folder, fixture):
    return folder / f'coded-{fixture}-fit.json.gz'


def fit(folder, fixture):
    path = setup_path(folder, fixture)
    spec = _read(path)
    arena = make_arena(spec)
    try:
        cache_hall_stereo(arena, reflections=True)
        _, _, player, memory = load_optical_player(spec, arena)
        seed = int(np.random.default_rng([spec['search_seed'], 27318]).integers(0, 2**31 - 1))
        result = probe(memory.source, TT6Wiring(arena, spec['wiring']), seed, player.tracker_factory)
        result.update(fixture=fixture, seed=seed, setup_sha256=_hash(path), donor_sha256=spec['donor_sha256'])
        _write(fit_path(folder, fixture), result)
        saved = _read(fit_path(folder, fixture))
        reproduced = decode(memory.source, saved['actions'], saved['ticks'], saved['seen'])
        if reproduced != (saved['key'], saved['fit']):
            raise ValueError('camera-only decode did not reproduce from stored observations')
        print(fixture, result['fit'], result['key'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('fixture', choices=FIXTURES)
    parser.add_argument("--game", action="store_true")
    args = parser.parse_args()
    if args.game:
        from .tt16_transfer_bridge import costs
        from .tt16_transfer_game import play_case
        receipt = fit_path(args.folder, args.fixture)
        evidence = _read(receipt)
        play_case(args.folder, args.fixture, 'coded', key=evidence['key'], prep_frames=len(evidence['actions']),
                  prep_effort=costs(evidence['actions']), prep_receipt=receipt)
    else:
        fit(args.folder, args.fixture)

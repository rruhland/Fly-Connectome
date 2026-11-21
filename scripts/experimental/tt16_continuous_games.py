"""Score the first shifting-transport installation of each new calibration cohort."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases
from .tt10_series_trial import PHASES
from .tt16_transfer_bridge import costs
from .tt16_transfer_game import case_prefix, game_clock, play_case, verify
from .tt16_transfer_protocol import FIXTURES, setup_path


def run(folder, fixture):
    spec = _read(setup_path(folder, fixture))
    tape_path = folder / f'coded-{fixture}-continuous.json.gz'
    receipt = _read(tape_path)
    idx, row = next((i, r) for i, r in enumerate(receipt['rows']) if r['mode'] == 'shifting')
    name = fixture+'-shift'
    game_seed = int(np.random.default_rng([row['scene'], 287634]).integers(0,2**31-1))
    spec.update(fixture=name, wiring=row['wiring_seed'], game_seed=game_seed,
                starts=_seeds(game_seed,618,0,65)['tests'], phases=match_phases(PHASES,65,[game_seed,618]),
                camera_setup_sha256=receipt['setup_sha256'])
    _write(setup_path(folder,name), spec)
    routes = {}
    for mode in ('first','adaptive'):
        key = row['decision'][mode]
        count = 24 if mode=='adaptive' and row['decision']['use_second'] else 12
        if mode=='adaptive' and key == row['decision']['first']:
            path = case_prefix(folder,name,'first').with_suffix('.json.gz')
            game = _read(path)
            verify(game, spec)
        else:
            game = play_case(folder,name,mode,key=key,prep_frames=count,
                             prep_effort=costs(receipt['commands'][:count]),prep_receipt=tape_path)
            path = case_prefix(folder,name,mode).with_suffix('.json.gz')
        routes[mode] = dict(game=path.name, game_sha256=_hash(path), score=game['score'],
                            returns=game['metrics']['total_legal_returns'],
                            two_return_points=game['metrics']['two_return_points'],
                            clock=game_clock(game['points'],spec['optics_frames']+count,spec['control_dt']))
    _write(folder / f'coded-{fixture}-continuous-games.json.gz',dict(
        fixture=fixture, row=idx, calibration_sha256=_hash(tape_path), results=routes))
    print(name, {mode: (a['score'],a['returns']) for mode,a in routes.items()},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('fixture',choices=FIXTURES)
    args=parser.parse_args()
    run(args.folder,args.fixture)

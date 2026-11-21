"""Qualify noisy active calibration on the same complete table games."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt16_transfer_bridge import costs
from .tt16_transfer_game import case_prefix, play_case, verify
from .tt16_transfer_protocol import FIXTURES, setup_path


def run(folder, fixture):
    spec = _read(setup_path(folder, fixture))
    fit_path = folder / f'coded-{fixture}-transport.json.gz'
    fits = _read(fit_path)
    if fits['setup_sha256'] != _hash(setup_path(folder, fixture)):
        raise ValueError('wrong camera or installation')
    row, = [r for r in fits['rows'] if r['index'] == 0 and r['condition'] == 'changing']
    calibration_cost = costs(fits['commands'])
    results = {}
    choices = ('coded', 'probe_only', 'search_pitch', 'lag_zero')
    for rule in ('zero', 'timing'):
        mode = f'lag_{rule}'
        key = row['decisions'][rule]
        path = None
        for candidate in choices:
            existing = case_prefix(folder, fixture, candidate).with_suffix('.json.gz')
            if not existing.is_file():
                continue
            game = _read(existing)
            if (game['start'] == spec['donor_sha256'] and game['provenance']['setup']==fits['setup_sha256']
                    and key == game['initial_key'] == game['final_key']):
                # A frozen key is the only calibration result passed to play.
                # Agent RNG and rally updates restart from exactly the same donor.
                verify(game, spec)
                path = existing
                break
        if path is None:
            game = play_case(folder, fixture, mode, key=key, prep_frames=calibration_cost['frames'],
                             prep_effort=calibration_cost, prep_receipt=fit_path)
            path = case_prefix(folder, fixture, mode).with_suffix('.json.gz')
        results[rule] = dict(key=key, physical_game=path.name, game_sha256=_hash(path),
                             checkpoint_sha256=game['checkpoint'], score=game['score'],
                             returns=game['metrics']['total_legal_returns'], two_return_points=game['metrics']['two_return_points'])
    _write(folder / f'coded-{fixture}-transport-games.json.gz',
           dict(fit_sha256=_hash(fit_path), calibration=calibration_cost, row=[0, 'changing'], results=results))
    print(fixture, results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('fixture', choices=FIXTURES)
    args = parser.parse_args()
    run(args.folder, args.fixture)

"""Audit fixed-calibration controller comparisons and their saved continuations."""
import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt17_motor_protocol import setup_path
from .tt17_motor_trial import prefix


def scored(folder, case, mode):
    stem = prefix(folder, case, mode)
    game_path = stem.with_suffix('.json.gz')
    checkpoint_path = stem.with_name(stem.name + '-player.json.gz')
    game = _read(game_path)
    if (_hash(checkpoint_path) != game['checkpoint'] or
            _hash(setup_path(folder, case)) != game['setup_sha256']):
        raise ValueError('receipt has a mismatched checkpoint or setup')
    row = game['games'][0]
    if (len(row['points']) != game['replayed_points'] or
            game['initial_map'] != game['final_map'] or any(p['choice']['adopted'] for p in row['points'])):
        raise ValueError('comparison requires replayed games with the same frozen map')
    fields = ['points', 'points_won', 'own_serves', 'wins_on_own_serve', 'legal_own_serves',
              'total_legal_returns', 'two_return_points']
    return game, dict(game_sha256=_hash(game_path), checkpoint_sha256=game['checkpoint'],
                      score=row['score'], clock=game['clock'], **{k:row['metrics'][k] for k in fields})


def compare(folder, cases, modes):
    results = {}
    for case in cases:
        setup = None
        results[case] = {}
        for mode in modes:
            game, data = scored(folder, case, mode)
            signature = (game['setup_sha256'], game['initial_map'], game['preparation'], game['start'])
            if setup is not None and signature != setup:
                raise ValueError('comparison changed more than the controller')
            setup = signature
            results[case][mode] = data
    return results


def audit(folder, resume_seed):
    altered = [f'{body}-{profile}' for body in ('gantry','five') for profile in ('coupling','gain')]
    fresh = compare(folder, altered, ('bounded','travel','allocate'))
    historical = compare(folder, ('gantry-coupling-tt14', 'five-gain-tt14'), ('bounded','travel','allocate'))
    case = 'gantry-coupling'
    replay_folder = folder / f'resume-travel-{resume_seed}'
    continuation = compare(replay_folder, [case], ['bounded','travel'])
    fixture = _read(setup_path(replay_folder, case))
    if fixture['donor_sha256'] != fresh[case]['travel']['checkpoint_sha256']:
        raise ValueError('continuation differs from the first-game endpoint')
    for row in continuation[case].values():
        if row['clock']['setup'] or row['clock']['additional_dry']:
            raise ValueError('continuation spent an unreported calibration')
    _write(folder/'allocation-summary.json.gz', dict(changed=fresh, tt14=historical, continuation=continuation))
    print({case: {method: row['score'] for method,row in modes.items()}
           for case,modes in fresh.items()}, continuation[case]['travel']['score'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--resume-seed', type=int, default=257341089)
    args = parser.parse_args()
    audit(args.folder, args.resume_seed)

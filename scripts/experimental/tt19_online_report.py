"""Summarize charged interaction and score on the SO-101 online tapes."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write


def summarize(receipt):
    tape = _read(receipt)
    setup = tape['setup']['frames']
    total = setup
    first = dict(legal_serve=None, legal_return=None, second_legal_return=None)
    games = []
    played = 0
    for game in tape['games']:
        path = receipt.parent / game['checkpoint']
        if _hash(path) != game['checkpoint_sha256']:
            raise ValueError(f'checkpoint digest differs for {path.name}')
        if not game['finished']:
            raise ValueError(f'{receipt.name} contains a partial game')
        metrics = game['metrics']
        for key, tick in metrics['first_frames'].items():
            if tick is not None and first[key] is None:
                first[key] = total + tick
        games.append(dict(seed=game['seed'], score=game['score'], metrics=metrics,
                          checkpoint=path.name, checkpoint_sha256=_hash(path),
                          model_samples=_read(path)['tt19_online']['motor']['samples']))
        total += metrics['frames']
        played += len(game['points'])
    if played != tape['replayed_points']:
        raise ValueError(f'physics replay count differs for {receipt.name}')
    dt = tape['fixture']['public_frame_dt']
    prior = _read(receipt.parent / tape['games'][-1]['checkpoint'])['tt19_online']
    return dict(receipt=receipt.name, sha256=_hash(receipt), mode=tape['mode'], solver=tape.get('solver', 'clipped'),
                strict_endline=tape['fixture']['arena'].get('strict_endline', False), seed=tape['seed'],
                view=tape['view'], game_offset=tape.get('game_offset', 0), setup_frames=setup,
                charged_frames=total, game_frames=total - setup,
                first_frames=first, first_seconds={key: None if tick is None else tick * dt for key, tick in first.items()},
                replayed_points=played, observed_impulses=prior['shot_samples'], latest_motor_lag=prior['motor']['lag'],
                pitch_registration=prior.get('pitch_registration'), pitch_updates=prior.get('pitch_updates'),
                setup_commands=tape['setup']['actions'], games=games)


def registration_openings(receipts):
    paths = {}
    for receipt in receipts:
        tape = _read(receipt)
        spec = tape['fixture']['arena']
        if (tape['mode'] == 'warm' and spec.get('strict_endline') and tape['setup']['frames']
                and tape.get('solver') in ('clipped', 'register')):
            key = tape['seed'], tape['solver']
            paths[key] = tape
    comparisons = []
    for (seed, solver), registered in paths.items():
        if solver != 'register' or (seed, 'clipped') not in paths:
            continue
        plain = paths[seed, 'clipped']
        left, right = (tape['games'][0]['points'][0] for tape in (plain, registered))
        if (plain['setup'] != registered['setup'] or left['seed'] != right['seed']
                or left['action_sha256'] != right['action_sha256'] or left['vision_digest'] != right['vision_digest']):
            raise ValueError(f'pitch feedback changed the matched opening for {seed}')
        comparisons.append(dict(seed=seed, setup_frames=plain['setup']['frames'],
                                opening_point=left['seed'], action_sha256=left['action_sha256'],
                                vision_digest=left['vision_digest']))
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('receipts', nargs='+', type=Path)
    args = parser.parse_args()
    runs = [summarize(receipt) for receipt in args.receipts]
    _write(args.path, dict(schema='tt19-online-report-v1', runs=runs,
                           registered_openings=registration_openings(args.receipts),
                           physical_point_replays=sum(run['replayed_points'] for run in runs)))

"""Measure real-time learning milestones from replayed, alternating-service games."""
import argparse
from pathlib import Path


from .tt4_transfer_trial import _hash, _read, _write


def milestones(points):
    frames = 0
    marks = dict(first_legal_serve=None, first_legal_return=None, first_two_returns=None)
    own = legal = won = lost = returns = multiple = 0
    for row in points:
        own += int(row['server'] == 1)
        legal += int(row['server'] == 1 and row['evaluator']['legal_serve'])
        won += row['points_for']
        lost += row['points_against']
        returns += row['player_returns']
        multiple += int(row['player_returns'] >= 2)
        own_returns = 0
        for event in row['events']:
            elapsed = frames + event['step']
            if (row['server'] == 1 and event['kind'] == 'legal_serve'
                    and marks['first_legal_serve'] is None):
                marks['first_legal_serve'] = elapsed
            if event['kind'] == 'legal_return' and event['side'] == 1:
                own_returns += 1
                if marks['first_legal_return'] is None:
                    marks['first_legal_return'] = elapsed
                if own_returns == 2 and marks['first_two_returns'] is None:
                    marks['first_two_returns'] = elapsed
        frames += row['steps']
    return dict(**marks,frames=frames,own=own,legal=legal,won=won,lost=lost,
                returns=returns,multiple=multiple,points=len(points))


def measure(folder):
    results = {}
    for prefix, modes in (
            ('418336241', ('carried','cold','fixed_body','stale_view','choice','world')),
            ('stock-96955769', ('carried','cold','choice')),
            ('349051097', ('carried','cold','world'))):
        receipts = {}
        for mode in modes:
            stem = folder/f'combined-{prefix}-{mode}'
            path = stem.with_suffix('.json.gz')
            record = _read(path)
            checkpoint = stem.with_name(stem.name+'-player.json.gz')
            if _hash(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('saved player checksum changed')
            receipts[mode] = record['points']
            results[stem.name] = dict(sha256=_hash(path),score=record['score'],
                                      full=milestones(record['points']),changed=milestones(record['points'][2:]))
        if 'stale_view' in modes:
            for left,right in zip(receipts['carried'][:2], receipts['stale_view'][:2],strict=True):
                if (left['rgb'],left['action_sha256']) != (right['rgb'],right['action_sha256']):
                    raise ValueError('vision controls diverged before the perturbation')
    _write(folder/'combined-summary.json.gz',results)
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    for name,row in measure(p.parse_args().folder).items():
        print(name, row['score'], row['changed'])

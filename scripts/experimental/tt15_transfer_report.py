"""Check source-task provenance and score all search-to-table-tennis continuations."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt10_series_trial import _game_summary
from .tt13_live_audit import assignment


def audit(folder):
    results = []
    for site in ('five','gantry'):
        discovery = _read(folder/f'discovery-{site}.json.gz')
        for index, run in enumerate(discovery['rows'][:2]):
            prefix = f'back-{site}-{index}'
            bridge_path = folder/f'{prefix}-maps.json.gz'
            fit_path = folder/f'{prefix}-motion-fit.json.gz'
            bridge, fit = _read(bridge_path), _read(fit_path)
            if fit['search'] != _hash(bridge_path) or bridge['seed'] != run['seed']:
                raise ValueError('geometry came from a different search')
            ticks = sum(b['length'] for b in bridge['blocks'])
            for mode in ('carried','cold'):
                actions = np.array(run['cases'][mode]['actions'][:ticks],dtype='<f8')
                if sha256(actions.tobytes()).hexdigest() != bridge['maps'][mode]['action_sha256']:
                    raise ValueError('prefix differs from continuous discovery run')
            for method in ('fixed','cold','carried','motion','search','original'):
                online = method in ('search','original')
                stem = f'reentry-{site}-{index}' if online else prefix
                gameplay = folder/f'{stem}-{method}.json.gz'
                row = _read(gameplay)
                player = folder/f'{stem}-{method}-player.json.gz'
                learned = fit_path if online or method=='motion' else bridge_path
                if row['checkpoint'] != _hash(player) or row.get('maps',row.get('fit')) != _hash(learned):
                    raise ValueError('saved player or source fit differs from receipt')
                if row['metrics'] != _game_summary(row['points']) or not row['finished']:
                    raise ValueError('inconsistent match metrics')
                control = 'fixed' if method=='original' else 'motion' if method=='search' else None
                if control:
                    compared = _read(folder/f'{prefix}-{control}.json.gz')
                    if compared['points'][0]['action_sha256'] != row['points'][0]['action_sha256']:
                        raise ValueError('online and frozen versions differ before their first update')
                saved = _read(player)['port_memory']
                width = len(saved['keys'][0][0])
                truth = np.asarray(assignment(width,row['arena']['seed'],row['installation_seed']))
                opening = row['points'][0].get('model_used',saved['active'])
                exact = (np.asarray(saved['keys'][opening])==truth).all(axis=0)
                final = (np.asarray(saved['keys'][saved['active']])==truth).all(axis=0)
                results.append(dict(site=site,index=index,method=method,score=row['score'],starts=len(row['points']),
                                    returns=row['metrics']['total_legal_returns'],sustained=row['metrics']['sustained_points'],
                                    early_returns=sum(r['player_returns'] for r in row['points'][:4]),
                                    opening_correct_ports=int(exact.sum()), final_correct_ports=int(final.sum()),
                                    source_frames=ticks, checkpoint=row['checkpoint']))
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    a=p.parse_args()
    rows=audit(a.folder)
    _write(a.folder/'back-summary.json.gz',dict(rows=rows))
    for row in rows:
        print(row['site'],row['index'],row['method'],row['score'],row['returns'],row['sustained'],
              row['opening_correct_ports'],row['final_correct_ports'])

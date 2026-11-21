"""Compare bounce forecasts on identical physical trajectories, outside the learner."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt18_surface import SurfaceMemory


def bounces(events):
    pending = None
    for event in events:
        if event['kind'] == 'bounce':
            pending = event if event['side'] == 1 and event['incoming'][1] < 0 else None
        elif event['kind'] == 'separation' and event['collider'].startswith('table') and pending:
            speed = -pending['incoming'][2]
            if speed > .45 and event['velocity'][2] > 0 and event['step'] - pending['step'] <= 3:
                yield pending['step'], pending['location'][:2], event['velocity'][2] / speed
            pending = None


def compare(folder):
    rule = _read(folder/'surface-source-rule.json.gz')['choice']
    stats = {}
    for site in ('gantry', 'five'):
        fixture = {}
        for mode in ('inherited', 'global', 'local', 'transfer'):
            path = folder/f'surface-{site}-{mode}.json.gz'
            data = _read(path)
            if data['checkpoint'] != _hash(folder/f'surface-{site}-{mode}-player.json.gz'):
                raise ValueError('changed checkpoint')
            fixture[mode] = [dict(score=g['score'], returns=g['metrics']['total_legal_returns'],
                                  multiple=g['metrics']['two_return_points'], frames=g['clock']['game_frames'],
                                  sightings=sum(len(p['surface_events']) for p in g['points'])) for g in data['games']]
        track = _read(folder/f'surface-{site}-inherited.json.gz')
        estimates = dict(global_mean=SurfaceMemory(), local=SurfaceMemory(.20),
                         source=SurfaceMemory(rule['width'],rule['lifetime']))
        evaluations = []
        for game in track['games']:
            errors = {name: [] for name in ('constant', *estimates)}
            observed = 0
            for point in game['points']:
                optical = iter(point['surface_events'])
                seen = next(optical, None)
                # A fixed shared prior makes the local/global diagnostic independent
                # of unlogged per-frame changes in the ordinary player's bounce prior.
                def assimilate(event):
                    for memory in estimates.values():
                        memory.samples.append([*event['xy'], event['ratio']])
                for tick, xy, truth in bounces(point['events']):
                    while seen and seen['observed_at'] < tick:
                        assimilate(seen)
                        observed += 1
                        seen = next(optical, None)
                    errors['constant'].append(abs(.86-truth))
                    for name, memory in estimates.items():
                        errors[name].append(abs(memory.predict(xy,.86)-truth))
                while seen:
                    assimilate(seen)
                    observed += 1
                    seen = next(optical, None)
            if not errors['constant']:
                raise ValueError('no physical bounces on held tape')
            evaluations.append(dict(physical=len(errors['constant']), optical=observed,
                                    mean_absolute={k:float(np.mean(v)) for k,v in errors.items()}))
        stats[site] = dict(games=fixture, matched_flight=evaluations)
    _write(folder/'surface-summary.json.gz', stats)
    return stats


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    print(compare(p.parse_args().folder))

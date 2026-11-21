"""Time the motor-map update by rerunning the saved optical traces on one CPU."""
import argparse
import platform
from pathlib import Path
from time import process_time_ns

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_memory import load
from .tt13_live_ports import PortMemory


def measure(path):
    report = _read(path)
    checkpoint = path.with_name(path.name.replace('.json.gz', '-player.json.gz'))
    saved = _read(checkpoint)['port_memory']
    library = load(checkpoint)[0]
    models = [library.models[i] for i in saved['indices']]
    keys = [(tuple(order), tuple(signs)) for order, signs in saved['keys']]
    seen = 1
    samples = []
    for game in report['games']:
        for row in game['points']:
            result = row['choice']['active']
            if 'optical_trace' in row:
                memory = PortMemory(models[0], 'remap')
                memory.models = models[:seen]
                memory.keys = keys[:seen]
                memory.active = row['model_used']
                t0 = process_time_ns()
                observed = memory.observe(row['actions'],row['optical_trace'])
                cost = (process_time_ns() - t0) / 1e6
                if observed['active'] != result:
                    raise ValueError('saved optical update did not reproduce its map choice')
                samples.append(dict(seed=row['seed'], frames=len(row['optical_trace']),
                                    search='candidate_errors' in observed, cpu_ms=cost))
            seen = max(seen, result+1)
    return dict(source=_hash(path), samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    sites = {}
    for site in ('gantry', 'five'):
        sites[site] = measure(args.folder/f'live-{site}-remap.json.gz')
        samples = sites[site]['samples']
        for searched in (False, True):
            times = [s['cpu_ms'] for s in samples if s['search'] == searched]
            if times:
                print(site, searched, len(times), np.median(times), max(times))
    _write(args.folder/'adaptation-cost.json.gz',dict(platform=platform.machine(), sites=sites))

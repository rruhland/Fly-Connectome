"""Test the motor map selected at the end of the first rewired table-tennis game."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_memory import load
from .tt12_motor_tracking import evaluate_maps


def run(folder, site, seed, count):
    maps, parents = {}, {}
    settings = None
    for mode in ('fixed','refit','remap'):
        base = folder/f'live-{site}-{mode}'
        receipt = base.with_suffix('.json.gz')
        checkpoint = base.with_name(base.name+'-player.json.gz')
        table = _read(receipt)
        saved = _read(checkpoint)
        # The bank is append-only. Freeze the map that was active at game one's
        # final reward, before any evidence from the return to the original ports.
        active = table['games'][0]['points'][-1]['choice']['active']
        index = saved['port_memory']['indices'][active]
        model = load(checkpoint)[0].models[index]
        maps[mode] = model.jacobian(np.zeros(model.action_size))[:3]
        parents[mode] = dict(game=_hash(receipt), checkpoint=_hash(checkpoint), index=index, active=active)
        if settings is not None and (table['arena'],table['new_ports']) != settings:
            raise ValueError('tracking requires the same body and wiring for all branches')
        settings = (table['arena'],table['new_ports'])
    arena, ports = settings
    rows = evaluate_maps(maps,arena,ports,seed,count)
    _write(folder/f'tracking-{site}.json.gz',dict(seed=seed,parents=parents,rows=rows,
                                                  arena=arena,ports=ports,implementation=_hash(Path(__file__))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site')
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--count',default=8,type=int)
    args=p.parse_args()
    run(args.folder,args.site,args.seed,args.count)

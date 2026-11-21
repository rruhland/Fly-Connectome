"""Compare table-tennis and cold starts with an entirely unfamiliar second motor mapping."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt14_signal_search import SignalArena, port_maps
from .tt15_context_trial import play, schedule
from .tt15_discovery import DiscoveryPlayer


def save_maps(path, player):
    _write(path,dict(schema='tt15-optical-motor-bank-v1',active=player.active,
                     bank=player.evidence.models.tolist(),local=player.motor.jac.tolist()))
    saved = _read(path)
    np.testing.assert_array_equal(saved['bank'],player.evidence.models)
    np.testing.assert_array_equal(saved['local'],player.motor.jac)


def run(site, output, seed, count):
    priors, parents, settings, _ = port_maps(Path('artifacts/tt13'),site)
    original = priors['fixed']
    results = []
    arena = SignalArena(**dict(settings,render_mode='projected'))
    try:
        for i, s in enumerate(np.random.default_rng(seed).integers(0,2**31-1,count).tolist()):
            ports = int(np.random.default_rng([s,991]).integers(0,2**31-1))
            wires = [TT6Wiring(arena,n) for n in (settings['seed'],ports)]
            blocks, metric = schedule(arena,s)
            cases = {}
            for mode, prior, discover in (
                ('carried',original,True),('cold',np.zeros_like(original),True),
                ('scrambled',original[:,::-1],True),('continuous',original,False)):
                player = DiscoveryPlayer(prior,discover)
                args = arena,wires,s,blocks,metric,bool(i%2)
                result = play(*args,player=player)
                if result != play(*args,tape=result['actions']):
                    raise ValueError('new-mapping physical replay diverged')
                size = len(player.evidence.models)
                cases[mode] = dict(result,events=player.events,bank_size=size)
                if i == 0 and mode in ('carried','cold'):
                    save_maps(output.with_name(f'discovery-{site}-{mode}-maps.json.gz'),player)
            prefix = 4 * original.shape[1]
            if any(case['actions'][:prefix] != cases['cold']['actions'][:prefix] for case in cases.values()):
                raise ValueError('opening probe budget diverged')
            results.append(dict(seed=s,blocks=blocks,ports=ports,metric=metric.tolist(),compressed=bool(i%2),cases=cases))
    finally:
        arena.close()
    _write(output,dict(rows=results,seed=seed,site=site,parents=parents,
                       implementation=_hash(Path(__file__)), learner=_hash(Path(__file__).with_name('tt15_discovery.py'))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',choices=('five','gantry'))
    p.add_argument('output',type=Path)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--count',type=int,default=6)
    a=p.parse_args()
    run(a.site,a.output,a.seed,a.count)

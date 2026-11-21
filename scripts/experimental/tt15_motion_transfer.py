"""Fit a signed motor correspondence to recent self-generated search movements."""
import argparse
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt13_live_ports import effective
from .tt14_signal_search import SignalArena, marker, port_maps
from .tt15_back_transfer import game
from .tt15_context_trial import play
from .tt15_discovery import DiscoveryPlayer


def match_motion(reference, commands, camera, curve):
    n = reference.shape[1]
    ports = np.array(list(permutations(range(n))))
    signs = np.array(list(product([-1,1],repeat=n)))
    bank = (reference.T[ports][:,None,:,:]*signs[None,:,:,None]).reshape(-1,n,3)
    positions = np.asarray(camera)
    q = effective(np.vstack((np.zeros(n),commands)),np.tile(curve,(n,1)))
    end = len(positions)
    times = np.arange(max(20,end-96), end, 3)
    best = np.full(len(bank),np.inf)
    for lag in (0,2,4,6):
        error = np.zeros(len(bank))
        for span in (5,12):
            delta = q[times-lag]-q[times-lag-span]
            movement = positions[times]-positions[times-span]
            residual = delta @ bank - movement
            error += np.linalg.norm(residual,axis=2).clip(max=.15).mean(axis=1)
        best = np.minimum(best,error*.5)
    order, sign = divmod(int(np.argmin(best)),len(signs))
    return (ports[order].tolist(),signs[sign].tolist()), float(best.min())


class Recorder:
    def __init__(self, player):
        self.player = player
        self.positions = []

    def act(self, rgb, reward):
        self.positions.append(marker(rgb))
        return self.player.act(rgb,reward)


def run(folder, site, index):
    source = folder/f'back-{site}-{index}-maps.json.gz'
    record = _read(source)
    priors, _, settings, _ = port_maps(Path('artifacts/tt13'),site)
    s = record['seed']
    arena = SignalArena(**dict(settings,render_mode='projected'))
    try:
        records = _read(folder/f'discovery-{site}.json.gz')
        cohort = records['rows'][index]
        actor = Recorder(DiscoveryPlayer(priors['fixed']))
        wires = [TT6Wiring(arena,n) for n in (settings['seed'],record['ports'])]
        r = play(arena,wires,s,record['blocks'],np.asarray(cohort['metric']),cohort['compressed'],player=actor)
        commands = np.asarray(r['actions'],dtype='<f8')
        if sha256(commands.tobytes()).hexdigest() != record['maps']['carried']['action_sha256']:
            raise ValueError('different exploration traces')
        key, error = match_motion(priors['fixed'],commands,actor.positions,actor.player.motor.curve)
        _write(folder/f'back-{site}-{index}-motion-fit.json.gz',
               dict(seed=s,ports=record['ports'],search=_hash(source),implementation=_hash(Path(__file__)),
                    maps=dict(motion=dict(key=key,error=error,action_sha256=record['maps']['carried']['action_sha256']))))
    finally:
        arena.close()
    game(folder,site,index,'motion')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site',choices=('gantry','five'))
    p.add_argument('index',type=int)
    a=p.parse_args()
    run(a.folder,a.site,a.index)

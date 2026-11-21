"""Blind, uninterrupted hidden-source searches through interleaved body and task changes."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _write
from .tt6_body_arena import TT6Wiring
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import TOLERANCE
from .tt10_target_trial import targets
from .tt14_signal_search import SignalArena, port_maps
from .tt15_context import ContextPlayer


def schedule(arena, seed):
    rng = np.random.default_rng([seed, 153])
    pool = targets(arena, seed, 16)
    current, motor = pool[0], 0
    blocks = []
    for kind in ['initial', *rng.permutation(['stable','motor','reward','both']).tolist()]:
        if kind in ('motor','both'):
            motor = 1 - motor
        if kind in ('reward','both'):
            available = [p for p in pool if .14 < np.linalg.norm(p-current) < .4]
            current = available[int(rng.integers(len(available)))]
        blocks.append(dict(kind=kind, motor=motor, goal=current.tolist(), length=int(rng.integers(120,153))))
    axes = Rotation.random(random_state=rng).as_matrix()
    return blocks, (axes*rng.uniform(.2,.58,3)) @ axes.T


def play(arena, wires, seed, blocks, metric, compressed, *, player=None, tape=None):
    disturbance = Disturbance(delays=((0,0),(76,2),(233,1),(417,3)),noise=((0,0.),(76,.025)))
    interface = DelayedWiring(wires[0], disturbance, seed=seed)
    interface.reset(seed=seed, render=False)
    actions, outcomes, readings = [], [], []
    rng = np.random.default_rng([seed,31])
    for block in blocks:
        interface.wiring = wires[block['motor']]
        goal = np.array(block['goal'])
        hits = []
        for _ in range(block['length']):
            delta = (arena.paddle-goal)/.20
            square = delta @ metric @ delta
            strength = round(float(2. - (np.log1p(square) if compressed else square) + rng.normal(0,.006)),2)
            action = tape[len(actions)] if tape is not None else player.act(arena.frame(),strength)
            interface.step(action,render=False)
            hits.append(int(np.linalg.norm(arena.paddle-goal) <= TOLERANCE))
            actions.append(np.asarray(action).tolist())
            readings.append(strength)
        held = np.flatnonzero(np.convolve(hits,np.ones(4),mode='valid')==4)
        outcomes.append(dict(frames=sum(hits), last=sum(hits[-32:]), acquired=int(held[0]+4) if held.size else None))
    return dict(actions=actions, readings=readings, blocks=outcomes)


def run(site, output, seed, count):
    priors, parents, settings, ports = port_maps(Path('artifacts/tt13'),site)
    arena = SignalArena(**dict(settings,render_mode='projected'))
    rows = []
    try:
        wires = [TT6Wiring(arena,s) for s in (settings['seed'],ports)]
        for i, s in enumerate(np.random.default_rng(seed).integers(0,2**31-1,count).tolist()):
            blocks, metric = schedule(arena,s)
            cases = {}
            for mode in ('selective','reset_both','reward_only','selective_probe','reset_both_probe'):
                player = ContextPlayer([priors['fixed'],priors['remap']],mode)
                args = arena,wires,s,blocks,metric,bool(i%2)
                result = play(*args,player=player)
                if result != play(*args,tape=result['actions']):
                    raise ValueError('physical trajectory or scalar signal failed replay')
                cases[mode] = dict(result,events=player.events)
            rows.append(dict(seed=s,blocks=blocks,metric=metric.tolist(),compressed=bool(i%2),cases=cases))
    finally:
        arena.close()
    _write(output,dict(seed=seed,site=site,parents=parents,rows=rows,
                       code={p:_hash(Path(__file__).with_name(p+'.py')) for p in
                             ('tt15_context','tt15_context_trial','tt14_signal_change','tt14_signal_search')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site',choices=('gantry','five'))
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--count',default=8,type=int)
    a=parser.parse_args()
    run(a.site,a.output,a.seed,a.count)

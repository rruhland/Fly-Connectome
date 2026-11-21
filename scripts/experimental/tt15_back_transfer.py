"""Carry geometry learned while locating an invisible source back into scored table tennis."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import match_phases
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_trial import check_tapes
from .tt10_series_trial import PHASES, _game_summary
from .tt10_window_trial import _point_row
from .tt12_motor_recall import rewire
from .tt13_live_trial import load_player, save_player
from .tt14_signal_search import SignalArena, port_maps
from .tt15_context_trial import play, schedule
from .tt15_discovery import DiscoveryPlayer


def signed_fit(reference, current):
    # Compare observable tool directions. No pitch, referee result, or physical
    # port assignment is given to the solver. Unseen axes have ambiguous signs.
    pos = np.linalg.norm(reference.T[None,:,:] - current.T[:,None,:], axis=-1)
    neg = np.linalg.norm(reference.T[None,:,:] + current.T[:,None,:], axis=-1)
    rows,order = linear_sum_assignment(np.minimum(pos,neg))
    signs = np.where(pos[rows,order] <= neg[rows,order], 1, -1)
    return order.tolist(),signs.tolist()


def harvest(folder, site, index):
    root = Path('artifacts/tt13')
    priors, parents, settings, _ = port_maps(root,site)
    suite = 52843419 if site=='gantry' else 72148637
    seed = int(np.random.default_rng(suite).integers(0,2**31-1,index+1)[-1])
    ports = int(np.random.default_rng([seed,991]).integers(0,2**31-1))
    arena = SignalArena(**dict(settings,render_mode='projected'))
    try:
        blocks, metric = schedule(arena,seed)
        stop = next(i+1 for i,b in enumerate(blocks) if b['motor']==1)
        span = blocks[:stop]
        wires = [TT6Wiring(arena,n) for n in (settings['seed'],ports)]
        maps = {}
        for name, prior in [('carried',priors['fixed']),('cold',np.zeros_like(priors['fixed']))]:
            actor = DiscoveryPlayer(prior)
            result = play(arena,wires,seed,span,metric,bool(index%2),player=actor)
            actions = np.asarray(result['actions'],dtype='<f8')
            maps[name] = dict(local=actor.motor.jac.tolist(),key=signed_fit(priors['fixed'],actor.motor.jac),
                              action_sha256=sha256(actions.tobytes()).hexdigest())
    finally:
        arena.close()
    _write(folder/f'back-{site}-{index}-maps.json.gz',
           dict(seed=seed,start=parents,ports=ports,blocks=span,maps=maps,implementation=_hash(Path(__file__))))


def game(folder, site, index, mode):
    suffix = 'motion-fit' if mode == 'motion' else 'maps'
    receipt = folder/f'back-{site}-{index}-{suffix}.json.gz'
    bridge = _read(receipt)
    donor = Path('artifacts/tt13')/f'live-{site}-fixed-player.json.gz'
    settings = _read(donor)['series']['arena']
    seed = int(np.random.default_rng([bridge['seed'],61287]).integers(0,2**31-1))
    arena = TT8ServiceArena(**settings)
    prefix = folder/f'back-{site}-{index}-{mode}'
    try:
        cache_hall_stereo(arena,reflections=True)
        state, library, player, memory = load_player(donor,arena,'fixed')
        if mode != 'fixed':
            order, signs = bridge['maps'][mode]['key']
            memory.models.append(rewire(memory.source,order,signs))
            memory.keys.append((order,signs))
            memory.active = len(memory.models)-1
            player.adapter = memory.models[-1]
        wiring = TT6Wiring(arena,bridge['ports'])
        starts = _seeds(seed,618,0,65)['tests']
        phases = match_phases(PHASES,65,[seed,618])
        score = ServiceScore(first_server=-1)
        rows = []
        for point,phase in zip(starts,phases,strict=True):
            row = _point_row(player,arena,wiring,seed,point,phase,score.server)
            rows.append(row)
            score.record(row['evaluator'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('game incomplete')
        path = prefix.with_name(prefix.name+'-player.json.gz')
        save_player(path,state,library,player,memory,1)
        data = dict(seed=seed,arena=settings,installation_seed=bridge['ports'], start=_hash(donor),
                    maps=_hash(receipt),mode=mode,checkpoint=_hash(path),score=list(score.score),finished=True,
                    race_to=11,first_server=-1,practice=False,points=rows,metrics=_game_summary(rows))
        _write(prefix.with_suffix('.json.gz'),data)
        check_tapes(_read(prefix.with_suffix('.json.gz')))
        print(site,index,mode,score.score,data['metrics']['total_legal_returns'],flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('site',choices=('five','gantry'))
    p.add_argument('index',type=int)
    p.add_argument('mode',choices=('harvest','fixed','carried','cold','motion'))
    a=p.parse_args()
    if a.mode=='harvest':
        harvest(a.folder,a.site,a.index)
    else:
        game(a.folder,a.site,a.index,a.mode)

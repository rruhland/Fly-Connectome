"""Isolate the effect of a native lateral preference after the same physical near serve."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT,_read,_write
from .tt19_flight import FlightLaw
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_match import CHECKPOINT,HANDOVER
from .tt22_far_cache import FarCache
from .tt22_opponent import RallyArena
from .tt22_ordinary_game import DIR,SEED,StandardMatch,FrozenMemory
from .tt22_rally_trial import CONFIG
from .tt22_spatial import portable_contact
from .tt22_spatial_trial import SOURCE


class IntentPlayer(InversePlayer):
    def __init__(self,law):
        super().__init__(law)
        self.gains=np.zeros(3)

    def search(self,estimate,observation,target):
        self.actor=self.factory.pose(target)[0]
        return super().search(estimate,observation,target)

    def lateral_contact(self,estimate,center):
        request=portable_contact(self.actor[0],estimate[0][0],center[0],self.gains)
        cell=.08*float(np.clip(np.rint(request/.08),-3,3))
        return float(np.clip(center[0],cell-.04,cell+.04))


def verify_rows(rows):
    arena=RallyArena(**CONFIG)
    try:
        for row in rows:
            arena.reset(seed=row['seed'],server=1,render=False)
            for i,(near,far) in enumerate(zip(row['actions'],row['rival_actions'],strict=True)):
                arena._aim=arena._command_to=lambda *args: np.asarray(far)
                _,reward,done,cap,_=arena.step(near,render=False)
                if (done or cap)!=(i+1==len(row['actions'])):
                    raise ValueError('changed termination')
            if arena.events!=row['events'] and __import__('json').dumps(arena.events)!=__import__('json').dumps(row['events']):
                raise ValueError('changed event tape')
            if arena.point_result()!=row['outcome'] or reward!=row['reward']:
                raise ValueError('changed point')
    finally:
        arena.close()


def run(path,count):
    seed=628352316
    mem=_read(SOURCE)
    top=int(np.argmax(np.array(mem['totals'])/mem['counts']))
    prior=np.array(mem['coefficients'][top])
    gains=dict(pong=prior,neutral=np.zeros(3),opposed=-prior)
    feed=_read(DIR/'tt22_opponent-learned.json.gz')['fit']['profile']
    base=FrozenMemory(_read(CHECKPOINT)['evidence'])
    station=StandardMatch(feed)
    cache=FarCache(station.rival)
    station.player.close()
    station.player=IntentPlayer(FlightLaw())
    result=dict(seed=seed,coefficients={k:v.tolist() for k,v in gains.items()},rows={k:[] for k in gains},feed=feed)
    try:
        for i,s in enumerate(np.random.default_rng(seed).integers(2**31,size=count)):
            keys=list(gains)
            opening=None
            for k in keys[i%3:]+keys[:i%3]:
                station.player.gains=gains[k]
                r=station.play(int(s),1,base)
                start=(r['actions'][:HANDOVER],r['rival_actions'][:HANDOVER])
                if opening is not None and opening!=start:
                    raise ValueError('treatment changed the service')
                opening=start
                result['rows'][k].append(r)
                print(i,k,r['longest'],r['outcome']['legal_returns'],flush=True)
                _write(path,result)
    finally:
        station.close()
    flat=[r for rows in result['rows'].values() for r in rows]
    verify_rows(flat)
    result['verified']=len(flat)
    result['opponent_cache']=dict(hits=cache.hits,misses=cache.misses,audited=cache.checked)
    _write(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DIR/'after-service.json.gz')
    p.add_argument('--count',type=int,default=8)
    a=p.parse_args()
    run(a.output,a.count)

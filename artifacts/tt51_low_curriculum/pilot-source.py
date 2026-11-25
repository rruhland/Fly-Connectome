"""Reward-driven practice of a separate elbow-forward RGB/joint receiving skill.

The practice coach may choose easier physical feeds and the starting posture.
The player sees RGB-derived flight, own encoders/references, and actual scores.
No archived failed-court identities or true incoming physics enter the player.
"""
import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
from time import process_time, monotonic
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as combined
from . import tt46_continuous_season as season
from . import tt37_opponent as opponent
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_optical_clinic import OpticalClinic
from .tt41_shallow_model import Chart
from .tt43_stance_chart import path_options
from .tt43_stance_native import dynamic_options

ROOT = Path('artifacts/tt51_low_curriculum')
PLAN = ROOT/'plan.json'
ATLAS = Path('artifacts/tt41/shallow-atlas.json.gz')
CHART = Path('artifacts/tt43_stance/chart.json.gz')
# Complete alternative shallow contact families, not adjustments to old deep tapes.
FAMILIES = ((-1.12,-1),(-1.28,-1),(-1.44,-1),(-1.12,1),(-1.28,1),(-1.44,1))
STANDARD_DRAW = opponent.draw


class RewardMemory:
    def __init__(self, home, learn=True, packet=None):
        self.home=np.asarray(home,dtype=float)
        self.learn=learn
        self.rows=[] if packet is None else copy.deepcopy(packet['rows'])
        self.rng=np.random.default_rng(51025019)
        self.pending=None

    def choose(self, context):
        successes=np.ones(len(FAMILIES)); failures=np.ones(len(FAMILIES))
        for row in self.rows:
            (successes if row['landed'] else failures)[row['family']]+=1
        if self.learn:
            # Reward-only posterior exploration; all actual misses remain evidence.
            score=self.rng.beta(successes,failures)
        else:
            score=successes/(successes+failures)
        family=int(np.argmax(score))
        self.pending=dict(family=family,context=np.asarray(context).tolist(),plan=None)
        return family

    def record(self, landed):
        if self.pending is None:
            return
        if self.learn:
            self.rows.append(dict(self.pending,landed=bool(landed)))
        self.pending=None

    def keep(self, tape, leg):
        had_pending=self.pending is not None
        self.record(True)
        return int(self.learn and had_pending)

    def digest(self):
        import json
        digest=hashlib.sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()
        return dict(rows=len(self.rows),state_sha256=digest)

    def parameters(self):
        return [dict(r['plan'],family=r['family']) for r in self.rows
                if r['landed'] and r['plan'] is not None]


class WorkspaceSkill:
    """A full replacement motor skill with its own reward-earned local practice."""
    def __init__(self, hub, memory):
        self.hub=np.asarray(hub,dtype=float)
        self.priors=memory
        self.engine=OpticalClinic(body=True)
        self.cad=mpc.arena(detail='cad')
        atlas=_read(ATLAS)
        self.chart=Chart(self.cad,self.hub,atlas['seeds'][0],atlas=atlas)
        self.serial=0

    def close(self):
        self.cad.close();self.engine.twin.close()

    def begin(self):
        self.serial+=1;self.diag=Counter();self.notes=[]
        self.engine.previous=None;self.engine.bounced=False
        self.family=None;self.was_out=False;self.next=0
        self.tape=None;self.tape_tick=0
        self.priors.pending=None

    def step(self,tick,eye,x,q,reference,base):
        self.engine.observe(tick,eye,q,reference.copy())
        if x is not None and x[4]>.9:
            self.was_out=True;self.family=None;self.tape=None
            return self.hub
        incoming=(x is not None and x[4]<-.9 and -1.59<x[1]<.76 and .76<x[2]<1.74)
        if incoming and self.family is None:
            self.family=self.priors.choose(np.r_[x,q[:5],reference[:5]-q[:5]])
            self.next=tick
        if incoming and tick>=self.next and self.engine.sync():
            self.next=tick+6
            twin=self.engine.twin
            if twin.launches or twin.card is not None:
                raise RuntimeError('Predictor acquired a live launch or court identity')
            depth,face=FAMILIES[self.family]
            path=twin.partner.flight.path(twin.ball,twin.velocity,twin.spin,1,
                bounced=self.engine.bounced,horizon=.72)
            poses=path_options(self.chart,path,quick=True,depths=(depth,))
            poses=[p for p in poses if p['face']==face and .055<p['eta']<.61
                   and np.max(abs(np.asarray(p['q'])-q[:5]))/4.16<p['eta']-.014
                   and self.chart.guard.route(q[:5],np.asarray(p['q']),steps=7)]
            pick,trials=dynamic_options(twin,self.engine.predict,self.chart.ik,poses,None,32,2,
                [510251,tick,self.serial,self.family],memories=self.priors.parameters())
            self.diag['searches']+=1;self.diag['native_candidates']+=sum(t['tried'] for t in trials)
            note=dict(tick=tick,family=self.family,depth=depth,face=face,
                optical=x.tolist(),body=q.tolist(),poses=len(poses),trials=trials,issued=pick is not None)
            self.notes.append(note)
            if pick is not None:
                self.tape=np.asarray(pick['goals']);self.tape_tick=tick
                if self.priors.pending is not None:
                    self.priors.pending['plan']={k:pick['seed'][k] for k in ('p','v','face')}
                    self.priors.pending['plan'].update(parameters=pick['parameters'],drive_mode=pick['drive_mode'])
                self.diag['issued_plans']+=1
            elif poses and x[1]>-.63:
                p=max(poses,key=lambda p:p['eta']-np.max(abs(np.asarray(p['q'])-q[:5]))/4.16)
                self.diag['approach_ticks']+=1
                # Geometric positioning is a prior, never a rewarded strike label.
                return np.r_[p['q'],self.hub[-1]]
        if self.tape is not None and 0<=tick-self.tape_tick<len(self.tape):
            self.diag['screened_ticks']+=1
            return self.tape[tick-self.tape_tick]
        return self.hub

    def missed(self,tape):
        self.priors.record(False)


class PracticeCoach:
    def __init__(self, memory):self.memory=memory;self.calls=[]
    def draw(self,card,leg,protocol,rng):
        wins=sum(r['landed'] for r in self.memory.rows)
        level=min(2,wins//4)
        r=np.random.default_rng([510252,card['seed'],card['reply_seed'],leg])
        z=((1.078,1.108),(1.044,1.078),(1.010,1.046))[level]
        # Every fourth request remains broad ordinary practice for retention.
        if (len(self.calls)+1)%4==0:
            target,origin=STANDARD_DRAW(card,leg,protocol,rng)
        else:
            spread=(.13,.20,.28)[level]
            target=(float(r.uniform(-spread,spread)),float(r.uniform(-1.34,-1.22)),
                    float(r.uniform(*z)),float(r.uniform(3.20,3.80)))
            origin='tt51_reward_curriculum'
        self.calls.append(dict(level=level,rewarded_attempts=wins,target=list(target)))
        return target,origin


def prepare():
    assert not PLAN.exists()
    old=_read(season.PLAN);rng=np.random.default_rng(510253)
    pool={c['id']:c for c in old['practice']}
    held={c['id']:c for c in old['assessment']}
    practice=[]
    for i in range(24):
        c=list(pool.values())[i%len(pool)]
        practice.append(dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))))
    assessment=[]
    for c in list(held.values())[:8]:
        assessment.append(dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))))
    assert not {c['id'] for c in practice}&{c['id'] for c in assessment}
    ready=np.asarray(_read(mpc.BANK.with_suffix('.json'))['ready'])
    hub=np.r_[_read(CHART)['routes']['either_far'][1]['q'],ready[-1]]
    ROOT.mkdir(parents=True,exist_ok=True)
    _write(PLAN,dict(practice=practice,assessment=assessment,ready=ready.tolist(),hub=hub.tolist(),
        sampling_seed=510253,source_sha256=_hash(Path(__file__)),chart_sha256=_hash(CHART),atlas_sha256=_hash(ATLAS),
        original_schedule_sha256=_hash(season.PLAN),families=[list(f) for f in FAMILIES],
        practice_scaffold='Physical starting arm posture is the declared new workspace; coach changes only far requested goals, then uses actual partner physics/referee/RGB.',
        official='Normal physical initial arm state and unchanged broad TT37 independent draw. No low-feed practice goal, ID, seed or evaluator height enters the actor.',
        learning='Beta score posterior over six complete stroke families, plus motor parameters retained only after actual legal landings. All chosen-family misses retained. No backprop.',
        stage_rule='Four actual rewarded attempts per difficulty step; every fourth far request is broad standard.'))
    print('prepared',_hash(PLAN),hub.tolist(),flush=True)


def run(args):
    packet=_read(PLAN)
    if packet['source_sha256']!=_hash(Path(__file__)):
        raise ValueError('Source changed after predeclaring the course')
    practicing=args.phase=='practice';cards=packet['practice' if practicing else 'assessment'][:args.n]
    ready=np.asarray(packet['hub'] if practicing else packet['ready'])
    memory=RewardMemory(packet['hub'],learn=practicing or args.phase=='online',
        packet=_read(args.memory)['memory'] if args.memory else None)
    coach=PracticeCoach(memory)
    output=dict(schema='tt51_reward_workspace_curriculum_v1',phase=args.phase,cards=cards,ready=ready.tolist(),
        normal_official_start=not practicing,plan_sha256=_hash(PLAN),source_sha256=_hash(Path(__file__)),
        starting_memory_sha256=_hash(args.memory) if args.memory else None,rows=[],memory=dict(rows=memory.rows),complete=False)
    feeds=mpc.measured_launches(args.feeds)
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    began=monotonic();cpu=process_time()
    with mpc.centered(ready):
        a=mpc.arena();res=mpc.resources(a);model=WorkspaceSkill(packet['hub'],memory)
        try:
            with patch.object(opponent,'draw',coach.draw if practicing else STANDARD_DRAW):
                for c in cards:
                    before=memory.digest()
                    row=mpc.play(a,*res,ready,c,feeds[c['id']],raw,model,course.OPTIC,
                        'live' if memory.learn else 'frozen',args.cap)
                    if not practicing:
                        for leg,item in enumerate(row['placements']['requested']):
                            if item['target']!=list(STANDARD_DRAW(c,leg,'random',None)[0]):
                                raise RuntimeError('Official continuous draw changed')
                    if not memory.learn and memory.digest()!=before:
                        raise RuntimeError('Frozen reward memory changed')
                    output['rows'].append(row);output['memory']=dict(rows=memory.rows)
                    output.update(wall_seconds=monotonic()-began,cpu_seconds=process_time()-cpu,
                        actual_control_ticks=sum(len(r['actions']) for r in output['rows']),practice_requests=coach.calls)
                    _write(args.out,output)
                    print(len(output['rows']),c['id'],row['score']['learner'],len(memory.rows),
                        sum(r['landed'] for r in memory.rows),row['planner'],flush=True)
        finally:model.close();res[0].close();a.close()
    output['complete']=True;_write(args.out,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','practice','frozen','online'))
    p.add_argument('--out',type=Path,default=ROOT/'practice.json.gz');p.add_argument('--memory',type=Path)
    p.add_argument('--n',type=int,default=6);p.add_argument('--cap',type=int,default=4)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    args=p.parse_args();prepare() if args.phase=='prepare' else run(args)

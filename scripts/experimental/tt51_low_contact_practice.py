"""Acquire a local receiving stroke by real contact trials, without a virtual-win gate.

This is a separate complete workspace policy, not a modification of the ordinary
deep movement library. A coach starts practice in a geometric low contact pose
and initially sends physical feeds into that workspace, then broadens on scores.
"""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt51_low_curriculum as base
from . import tt37_opponent as opponent
from .tt4_transfer_trial import _read,_write,_hash
from .tt43_stance_native import direct_swing

ROOT=base.ROOT
PLAN=ROOT/'contact-plan.json'
FAMILIES=base.FAMILIES


class ContactMemory(base.RewardMemory):
    def choose(self,context,available=None):
        count=3*len(FAMILIES)
        good=np.ones(count);bad=np.ones(count)
        for r in self.rows:(good if r['landed'] else bad)[r['family']]+=1
        score=self.rng.beta(good,bad) if self.learn else good/(good+bad)
        mask=np.zeros(count,dtype=bool)
        for family in available:
            mask[3*family:3*family+3]=True
        if not mask.any():return None
        arm=int(np.argmax(np.where(mask,score,-np.inf)))
        self.pending=dict(family=arm,context=np.asarray(context).tolist(),plan=None)
        return arm


class ContactSkill(base.WorkspaceSkill):
    def begin(self):
        super().begin();self.arm=None;self.contact_eta=None

    def options(self,path,q):
        result=[]
        for depth in sorted({f[0] for f in FAMILIES}):
            if not len(path) or not path[:,2].min()<=depth<=path[:,2].max():continue
            p=path[np.argmin(abs(path[:,2]-depth))]
            eta=float(p[0])
            if not (.065<eta<.65 and .835<p[3]<1.22 and abs(p[1])<.46):continue
            # Warm-start from the actual current body, not just the first atlas neighbor.
            for row in self.chart.poses_at(p[1:4],p[4:7],4.3,.13,q[:5],quick=True):
                if row['travel']/4.16>eta-.012:continue
                if not self.chart.guard.route(q[:5],row['q'],steps=7):continue
                family=min((i for i,f in enumerate(FAMILIES) if f[1]==row['sign']),
                    key=lambda i:abs(FAMILIES[i][0]-depth))
                result.append(dict(row,family=family,eta=eta,p=p[1:4],v=p[4:7]))
        return result

    def step(self,tick,eye,x,q,reference,unused_deep_prior):
        self.engine.observe(tick,eye,q,reference.copy())
        if x is not None and x[4]>.9:
            self.arm=None;self.tape=None;self.contact_eta=None
            return self.hub
        incoming=x is not None and x[4]<-.9 and -1.58<x[1]<.76 and .76<x[2]<1.75
        frozen_contact=self.contact_eta is not None and self.contact_eta-tick<6
        if incoming and tick>=self.next and not frozen_contact and self.engine.sync():
            self.next=tick+6;twin=self.engine.twin
            if twin.launches or twin.card is not None:raise RuntimeError('Live state in private predictor')
            path=twin.partner.flight.path(twin.ball,twin.velocity,twin.spin,1,
                bounced=self.engine.bounced,horizon=.72)
            poses=self.options(path,q)
            available={p['family'] for p in poses}
            if self.arm is None and available:
                self.arm=self.priors.choose(np.r_[x,q[:5],reference[:5]-q[:5]],available)
            candidates=[p for p in poses if self.arm is not None and p['family']==self.arm//3]
            note=dict(tick=tick,arm=self.arm,poses=len(poses),available_families=sorted(available),
                optical=x.tolist(),body=q.tolist(),issued=False,private_win_filter=False)
            self.notes.append(note);self.diag['searches']+=1
            if candidates:
                p=min(candidates,key=lambda p:p['rank']+.08*p['travel']/p['eta'])
                mode=self.arm%3;params=np.zeros((1,13));dt=twin.control_dt
                goals=direct_swing(p['q'][None],p['dq'][None],np.array([p['eta']]),
                    params,np.array([mode]),dt,round((p['eta']+.17)/dt))[0]
                self.tape=goals;self.tape_tick=tick;self.contact_eta=tick+round(p['eta']/dt)
                self.priors.pending['plan']=dict(p=p['p'].tolist(),v=p['v'].tolist(),face=p['sign'],
                    parameters=params[0].tolist(),drive_mode=mode)
                note.update(issued=True,depth=float(p['p'][1]),face=p['sign'],drive_mode=mode,
                    contact_delay_s=p['eta'],target=p['q'].tolist())
                self.diag['issued_plans']+=1
        if self.tape is not None and 0<=tick-self.tape_tick<len(self.tape):
            self.diag['screened_ticks']+=1
            return self.tape[tick-self.tape_tick]
        return self.hub


class ReachableCoach(base.PracticeCoach):
    def draw(self,card,leg,protocol,rng):
        scores=sum(r['landed'] for r in self.memory.rows);level=min(2,scores//4)
        r=np.random.default_rng([510254,card['seed'],card['reply_seed'],leg])
        bounds=(((.11,.19),(1.012,1.040)),((.02,.25),(1.025,1.075)),
                ((-.28,.28),(1.010,1.110)))[level]
        if (len(self.calls)+1)%4==0:
            target,origin=base.STANDARD_DRAW(card,leg,protocol,rng)
        else:
            target=(float(r.uniform(*bounds[0])),float(r.uniform(-1.32,-1.24)),
                float(r.uniform(*bounds[1])),float(r.uniform(3.20,3.65)))
            origin='tt51_contact_workspace_curriculum'
        self.calls.append(dict(level=level,rewarded_attempts=scores,target=list(target)))
        return target,origin


def prepare():
    assert not PLAN.exists()
    p=_read(base.PLAN);atlas=_read(base.ATLAS)
    # Geometric body calibration, chosen before any new physical scores.
    # This is a local low contact pose, rather than a transition hub between old strokes.
    hub=np.r_[atlas['hubs'][0]['q'],p['ready'][-1]]
    p.update(hub=hub.tolist(),source_sha256=_hash(Path(base.__file__)),
        contact_source_sha256=_hash(Path(__file__)),original_plan_sha256=_hash(base.PLAN),
        learning='18 depth/face/drive arms. Real native legal landings/misses update Beta counts. No private virtual-win gate authorizes or labels an action.',
        practice_scaffold='Generic CAD low contact pose from unlabelled kinematics; initially narrower feeds into that physical workspace, broadened after real scores. Normal broad TT37 and normal initial arm for assessment.',
        causal_amendment='Two initial transition-hub pilots supplied zero scored strokes; current FK/atlas matched exactly. Use a true local contact pose, actual-encoder IK seeds and real trials rather than requiring a virtual win before learning.')
    _write(PLAN,p);print('prepared',_hash(PLAN),hub.tolist(),flush=True)


def run(args):
    p=_read(PLAN)
    if p['contact_source_sha256']!=_hash(Path(__file__)):raise ValueError('Contact learner source changed')
    with patch.object(base,'PLAN',PLAN),patch.object(base,'WorkspaceSkill',ContactSkill),\
         patch.object(base,'RewardMemory',ContactMemory),patch.object(base,'PracticeCoach',ReachableCoach):
        base.run(args)
    d=_read(args.out);d.update(contact_source_sha256=_hash(Path(__file__)),
        true_reward_action_selection=True,private_legal_prediction_not_used=True)
    _write(args.out,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','practice','frozen','online'))
    p.add_argument('--out',type=Path,default=ROOT/'contact-practice.json.gz');p.add_argument('--memory',type=Path)
    p.add_argument('--n',type=int,default=8);p.add_argument('--cap',type=int,default=4)
    p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    a=p.parse_args();prepare() if a.phase=='prepare' else run(a)

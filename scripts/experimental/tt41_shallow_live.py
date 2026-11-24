"""Test a frozen earned shallow strike bank on assigned physical random points."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash,_read,_write
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt37_joint_deep import DeepField
from .tt37_opponent import PROFILES,arena
from .tt38_motorcoach import AIM
from .tt39_carry_joints import Carried
from .tt41_shallow import OUT,touch_band
from .tt41_shallow_model import Chart,FEATURE,contact_preference,goals


BIAS=((-0.29,.29),(-1.31,-1.16),(.914,1.044),(3.25,4.21))
BANK=mpc.BANK
PRIOR=mpc.ROOT/'tt37/camera-time-school.npz'
OPTIC=mpc.ROOT/'tt37/rebound.npz'
LEGACY=mpc.MPC


class Shallow(LEGACY):
    def __init__(self,bank,raw,cal,practice):
        super().__init__(bank,raw,variant='elastic')
        self.cad=arena(detail='cad')
        self.chart=Chart(self.cad,cal['home'],cal['seeds'][0],atlas=cal)
        self.skill=tuple(practice['bank'])
        available=sorted({r.get('hub_index',0) for r in self.skill})
        self.hubs=[(i,np.r_[cal['hubs'][i]['q'],cal['home'][-1]],cal['hubs'][i]['position'][0]) for i in available]

    def begin(self):
        super().begin()
        self.recovery=False
        self.early_tape=None
        self.early_from=-200
        self.vote='none'
        self.recheck=-100
        self.outbound=-200
        self.armed=False
        self.hub=self.hubs[0][1]
        self.side_shift=False

    def close(self):
        self.cad.close()
        super().close()

    def key(self,b):
        x=np.r_[b['p'],b['v']]
        rows=[r for r in self.skill if r['sign']==b['sign']]
        dist=[float(np.linalg.norm((x-r['x'])/FEATURE)+.38*abs(b['pace']-r['pace'])+.9*abs(b['pitch']-r['pitch'])) for r in rows]
        return [(rows[i],dist[i]) for i in np.argsort(dist)[:3] if dist[i]<2.55]

    def test(self,reference,cutoff):
        f=np.zeros((len(reference),9))
        _,out=self.engine.trajectories(reference,AIM,cutoff,feedback=f)
        self.diag['shallow_native_candidates']+=len(reference)
        cost,legal,_=contact_preference(out,f)
        legal&=(f[:,6]>-1.445)&(f[:,6]<-1.018)&(out[:,5]>.018)
        return cost,legal,out,f

    def acquire(self,tick,q,path):
        t=self.engine.twin
        if t.launches or t.card is not None:
            raise ValueError('Need a camera-initialized private rollout')
        bases=self.chart.windows(path,q[:5],deployment=True)
        pairs=[(k,row,d) for k,b in enumerate(bases) for row,d in self.key(b)]
        pairs=sorted(pairs,key=lambda r:r[2]+.65*bases[r[0]]['rank'])[:25]
        note=dict(tick=tick,mode='shallow_workspace',profiles=len(bases),candidates=len(pairs),kind='no transferable proposal')
        self.notes.append(note)
        if not pairs:return None
        keys=[key for key,row,d in pairs]
        params=np.array([row['parameters'] for key,row,d in pairs])
        moves,cutoff=goals(bases,keys,params,t.control_dt)
        v,ok,out,f=self.test(moves,cutoff)
        note.update(nominal=int(ok.sum()))
        eligible=np.flatnonzero(ok)
        if not len(eligible):return None
        merit=v[eligible]-.13*np.array([pairs[i][2] for i in eligible]);ids=eligible[np.argsort(-merit)[:6]]
        score=v[ids].copy();count=np.ones(len(ids),int)
        p,vel=t.ball.copy(),t.velocity.copy()
        spin=t.data.qvel[t._ball_dof+3:t._ball_dof+6];w=spin.copy()
        try:
            for s in (-1.,1.):
                t.ball[:]=p+s*np.array([.0022,.004,.0035]);t.velocity[:]=vel+s*np.array([.026,.063,.085])
                spin[:]=w+s*np.array([17.,-8.,11.])
                values,okay,*_=self.test(moves[ids],cutoff)
                score+=np.where(okay,values,-2.8);count+=okay
        finally:
            t.ball[:]=p;t.velocity[:]=vel;spin[:]=w
        stable=np.flatnonzero(count>=2)
        if not len(stable):
            note['kind']='not robust';return None
        i=stable[np.argmax(score[stable])];j=int(ids[i]);row=bases[keys[j]]
        n=max(1,round((out[j,1]+.082)/t.control_dt))
        note.update(kind='issued',contact_eta=float(out[j,1]),sign=row['sign'],position=row['p'].tolist(),
                    incoming=row['v'].tolist(),support=int(count[i]),nominal_departure=f[j,2:5].tolist(),
                    historical_release=pairs[j][1]['practice_release'],estimated_distance=pairs[j][2])
        self.diag['shallow_issued']+=1
        return moves[j,:n].copy()

    def step(self,tick,eye,x,q,reference,base):
        self.engine.observe(tick,eye,q,reference.copy())
        t=self.engine.twin
        if self.early_tape is not None and 0<=tick-self.early_from<len(self.early_tape):
            return self.early_tape[tick-self.early_from]
        if x is not None and x[4]>.9:
            self.best=None
            if x[1]>-.36 and not self.armed:
                self.outbound=tick;self.vote='undecided';self.early_tape=None
                self.armed=True;self.side_shift=False
                ranked=sorted(self.hubs,key=lambda h:np.max(abs(q-h[1]))+.21*abs(x[0]-h[2]))
                move=next((h for h in ranked if self.chart.guard.route(q[:5],h[1][:5],steps=10)),None)
                self.recovery=move is not None
                if move is not None:
                    self.hub=move[1]
                    self.notes.append(dict(tick=tick,mode='shallow_recovery',requested_hub=move[0]))
                self.diag['outbound_attempted']+=1
            if self.recovery and self.vote=='undecided':
                self.diag['outbound_hub_ticks']+=1
                return self.hub
            return base
        if x is None:
            if self.recovery and self.vote=='undecided' and tick-self.outbound<66:return self.hub
        elif x[4]<-.9:
            self.armed=False
            # Incoming x provides only a public, coarse sign cue. The new
            # posture is reversible until the independent flight screen commits.
            if self.recovery and self.vote=='undecided' and x[1]>=.16:
                if x[1]>.29 and not self.side_shift:
                    lateral=x[0]+np.clip((x[1]+1.19)/max(.9,-x[4]),.12,.65)*x[3]
                    move=min(self.hubs,key=lambda h:abs(lateral-h[2]))
                    if abs(lateral)>.09 and np.max(abs(q-move[1]))<1.72 and self.chart.guard.route(q[:5],move[1][:5],steps=10):
                        self.hub=move[1];self.side_shift=True;self.diag['incoming_side_hub']+=1
                return self.hub
        if x is not None and x[4]<-.9 and -.86<x[1]<.16 and tick>=self.recheck:
            self.recheck=tick+6
            if self.engine.sync():
                path=t.partner.flight.path(t.ball,t.velocity,t.spin,1,bounced=self.engine.bounced,horizon=.67)
                prior=path[(path[:,2]<-1.47)&(path[:,2]>-1.63)] if len(path) else path
                early=path[(path[:,2]>-1.42)&(path[:,2]<-1.065)&(path[:,3]<1.075)&(path[:,3]>.833)&(path[:,6]>.22)] if len(path) else path
                if len(prior) and float(prior[np.argmin(abs(prior[:,2]+1.535)),3])>1.042:
                    self.vote='stock-high';self.recovery=False
                    self.diag['protected_high']+=1
                elif len(early) and early[0,0]>.049 and self.vote!='stock-high':
                    self.vote='shallow';self.recovery=False
                    tape=self.acquire(tick,q,path)
                    if tape is not None:
                        self.early_from=tick;self.early_tape=tape;self.best=None
                        return tape[0]
        if x is not None and x[4]<-.9 and x[1]>-.57 and self.recovery and self.vote=='undecided':
            return self.hub
        with patch.object(self.engine,'observe',lambda *args:None):
            return super().step(tick,eye,x,q,reference,base)


def measured_offers(row,flight):
    """Diagnose the actually produced balls after play, never inside the actor."""
    e=row['events'];final=[]
    for i,bounce in enumerate(e):
        if bounce['kind']!='legal_return' or bounce.get('side')!=-1:continue
        dep=next((a for a in e[i+1:i+13] if a['kind']=='separation' and a.get('collider') in ('table','table_right')),None)
        if dep is None:continue
        p=flight.path(dep['ball'],dep['velocity'],dep['spin'],1,bounced=True,horizon=.57)
        if not len(p):continue
        sliced=p[(p[:,2]>-1.42)&(p[:,2]<-1.055)&(p[:,3]>.833)&(p[:,3]<1.075)&(p[:,6]>.22)]
        nominal=p[(p[:,2]<-1.47)&(p[:,2]>-1.63)]
        height=None if not len(nominal) else float(nominal[np.argmin(abs(nominal[:,2]+1.535)),3])
        choice='high' if height is not None and height>1.042 else 'shallow' if len(sliced) else 'other'
        stop=next((a['time'] for a in e[i+1:] if a['kind']=='legal_return' and a.get('side')==-1),float('inf'))
        hit=next((a for a in e[i+1:] if a['kind']=='contact' and a.get('side')==1 and a['time']<stop),None)
        legal=any(a['kind']=='legal_return' and a.get('side')==1 and a['time']<stop for a in e[i+1:])
        final.append(dict(kind=choice,baseline_plane_height=height,time=bounce['time'],actual_legal=legal,
                          actual_early=bool(hit and legal and touch_band(hit)),actual_position=None if hit is None else hit['location']))
    return final


def games(role,manifest,atlas,bank,feeds,output):
    docket=_read(manifest);cal=_read(atlas) if atlas is not None else None
    memory=_read(bank) if bank is not None else None
    if role=='learned' and (cal is None or memory is None or len(memory['bank'])<2 or memory['atlas_sha256']!=_hash(atlas)):
        raise ValueError('Need a nonempty physically earned shallow bank')
    cards=[dict(c,profile='reach' if i%2==0 else 'shallow41') for i,c in enumerate(docket['held'])]
    held={c['id'] for c in cards}
    for path in (BANK,PRIOR):
        receipt=_read(path.with_suffix('.json'))
        if held & {c['card']['id'] for c in receipt['practice']}:raise ValueError('Held openings overlap stock practice')
    if memory is not None and held & {r['practice_release'] for r in memory['bank']}:
        raise ValueError('Same physical shallow release cannot be in training and testing')
    info=_read(BANK.with_suffix('.json'));ready=info['ready']
    with np.load(BANK) as z:raw=DeepField(z,'local')
    shots=measured_launches(feeds)
    out=dict(role=role,cards=cards,mode='live',ready=ready,cap=200,rows=[],
             public_inputs='Stereo RGB, encoders, own issued motor reference. No card or live solver in controller.',
             plan_sha256=_hash(manifest),bank_sha256=None if memory is None else _hash(bank),standard_strokes=_hash(BANK),
             standard_history=_hash(PRIOR),shallow_profile=BIAS)
    def fragments(file,engine):return Carried(file,engine,practice=[PRIOR],held=held)
    with centered(ready),patch.dict(PROFILES,{'shallow41':BIAS}),patch.object(mpc,'Fragments',fragments):
        a=arena();renderer,rig,stride,lo,hi=mpc.resources(a)
        actor=Shallow(BANK,raw,cal,memory) if role=='learned' else LEGACY(BANK,raw,variant='elastic')
        try:
            for card in cards:
                row=mpc.play(a,renderer,rig,stride,lo,hi,ready,card,shots[card['id']],raw,actor,OPTIC,'live',200)
                row['offers']=measured_offers(row,a.partner.flight)
                row['end_reason']=next((e.get('reason') for e in reversed(row['events']) if e['kind']=='point'),None)
                out['rows'].append(row)
                print(card['id'],card['profile'],row['score']['learner'],row['score']['returns'],
                    row['planner'].get('shallow_issued',0),[(r['kind'],r['actual_early']) for r in row['offers']],flush=True)
                _write(output,out)
        finally:
            actor.close();renderer.close();a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('role',choices=('stock','learned'));p.add_argument('out',type=Path)
    p.add_argument('--plan',type=Path,default=OUT/'shallow-plan.json');p.add_argument('--atlas',type=Path,default=OUT/'shallow-atlas.json.gz')
    p.add_argument('--bank',type=Path,default=OUT/'shallow-practice.json.gz');p.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    a=p.parse_args();games(a.role,a.plan,a.atlas if a.role=='learned' else None,a.bank if a.role=='learned' else None,a.feeds,a.out)

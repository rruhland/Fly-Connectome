"""Matched public paddle velocity study on the mechanically qualified mount."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt95_patch_lessons as bank
from . import tt88_public_inverse as pub
from . import tt90_installation_learning as tt90
from . import tt88_mount_transfer as tilted
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt40_long_low import same_trace

mpc=tt90.BASE
ROOT=bank.ROOT
PLAN=ROOT/'public.protocol.json'
D90=tt90.path('practice-dose16')
H=pub.H
WEIGHT=np.array([.115,.185,.088,.69,.86,1.12])
CASES={'hard':(3,2),'whole':2}


class PatchRoots:
    """No tape, true ball state or held label enters this public goal generator."""

    def __init__(self,twin,mode,exclude):
        self.mode=mode
        self.inverse=bank.inv.Inverse(bank.inv.Body(twin),twin,tilted.MODEL)
        with np.load(bank.MODEL) as f:
            keep=~np.isin(f['id'],list(exclude))
            self.data={k:f[k][keep] for k in f.files}
        self.index=cKDTree(self.data['optical']/WEIGHT)

    def target_options(self,engine):
        d=engine.twin
        path=engine.nominal.flight.path(d.ball,d.velocity,d.spin,1,bounced=engine.bounced,horizon=.45)
        if not len(path):
            return []
        f=self.data
        proposals=[]
        prev=engine.previous[:5]
        for frames in range(7,H-2):
            j=int(np.argmin(abs(path[:,0]-frames*bank.DT)))
            if abs(path[j,0]-frames*bank.DT)>.012:
                continue
            p,v=path[j,1:4],path[j,4:7]
            if not (-1.765<p[1]<-.98 and .834<p[2]<1.155 and abs(p[0])<.38 and v[1]<-1.0):
                continue
            distances,ranks=self.index.query(np.r_[p,v]/WEIGHT,min(14,len(f['id'])))
            for distance,i in zip(np.atleast_1d(distances),np.atleast_1d(ranks),strict=True):
                if distance>2.60:
                    break
                n=f['normal'][i]
                uv=f['patch_coordinates'][i]
                relative=f['patch_relative'][i]
                vel=f['velocity'][i]
                # Same context and patch geometry for both variants. Ignore
                # tangent only after the candidate and horizon are selected.
                for sign in (1.,-1.):
                    solid=np.r_[p+relative,sign*n,vel]
                    travel=np.max(abs(f['joint'][i]-prev))
                    # Prefer true same-mount teaching without excluding valid stock forms.
                    cost=float(distance+.13*travel+.21*(1-f['body'][i])+.05*(sign!=f['signed_face'][i])
                               +1.8*max(0.,np.linalg.norm(uv)-.051))
                    proposals.append((cost,frames,solid,uv*np.array([sign,1.]),int(i)))
        result=[]
        for candidate in sorted(proposals,key=lambda v:v[0]):
            score,k,x,uv,i=candidate
            if any(abs(k-r[1])<2 and np.dot(x[3:6],r[2][3:6])>.936 for r in result):
                continue
            result.append(candidate)
            if len(result)>=6:
                break
        return result

    def paths(self,engine,ref):
        state=np.r_[engine.previous[:5],engine.rate[:5],ref[:5]]
        out,notes=[],[]
        for score,n,target,uv,i in self.target_options(engine):
            target=target.copy()
            normal=target[3:6]
            tangential=bank.tangent(normal,target[6:])
            if self.mode=='normal':
                target[6:]-=tangential
            motor,_=self.inverse.solve(state,target,uv,n,rounds=2)
            planner=self.inverse.motor
            cur=ref.copy()
            refs=[]
            for u in motor:
                cur[:5]=np.clip(cur[:5]+planner.stride*u,planner.lo,planner.hi)
                refs.append(cur.copy())
            for j in range(H-len(refs)):
                cur[:5]=np.clip(cur[:5]+.38*planner.stride*motor[-1]/(1.+j),planner.lo,planner.hi)
                refs.append(cur.copy())
            out.append(np.asarray(refs)-ref)
            f=self.data
            notes.append(dict(frames=n,lesson_id=int(f['id'][i]),same_mount=bool(f['body'][i]),
                              patch_uv=uv.tolist(),velocity=target[6:].tolist(),
                              tangent_mps=float(np.linalg.norm(tangential)),distance=score))
        return np.asarray(out),notes


class Pupil(tt90.ObservedPlayer):
    def __init__(self,*args,mode,exclude,**kw):
        super().__init__(*args,**kw)
        self.mode=mode
        self.patch_ik=None if mode=='frozen' else PatchRoots(self.engine.twin,mode,exclude)

    def begin(self):
        super().begin()
        self.patches=[]
        self.patch_left=2
        self.patch_last=-100
        self.patch_protected=0

    def step(self,tick,eye,x,q,ref,base):
        if x is not None and x[4]>1.:
            self.patch_left=2
        return super().step(tick,eye,x,q,ref,base)

    def search(self,tick,x,q,ref,base):
        super().search(tick,x,q,ref,base)
        if self.patch_ik is None or not (-1.36<x[1]<-.38 and x[4]<-1.05 and abs(x[0])<.38):
            return
        if not self.notes or self.notes[-1]['tick']!=tick or tick-self.patch_last<5:
            return
        current=self.notes[-1]
        if current.get('robust',0):
            self.patch_protected+=1
            return
        if current.get('chosen') or not self.patch_left or not self.engine.sync():
            return
        self.patch_last=tick
        self.patch_left-=1
        paths,memo=self.patch_ik.paths(self.engine,ref)
        entry=dict(tick=tick,x=x.tolist(),candidates=len(memo),chosen=None)
        self.patches.append(entry)
        if not len(memo):
            return
        ranks,summary=pub.population.few.rank_paths(self.engine,paths,ref,memo)
        entry['gate']=summary
        solid=[(u,x) for u,x in ranks if x['legal'] and x['robust']]
        if not solid:
            # Allow an informative real attempt from either ablation on a
            # visibly hard low only when the nominal actually touches and its
            # outgoing may be legal, or a nominal legal lacks a stress vote.
            from .tt59_online_corrections import low_envelope
            if self.patch_left==1 and .80<low_envelope(x)<1.028:
                solid=[(u,x) for u,x in ranks if x['legal'] or (x['robust']==2 and x['separated'])]
        if not solid:
            return
        u,d=min(solid,key=lambda y:y[1]['score'])
        a=self.engine.twin
        step,lo,hi=tt90.library.limits(a)
        self.best=tt90.library.references(u,ref,step,lo,hi)[1:]
        self.tick=tick
        self.lease=H
        self.patch_left=0
        entry['chosen']=d


def freeze():
    data=_read(tt90.PLAN)
    hard=_read(tt90.HARD)
    roles=[r['kind'] for r in hard['cases']]
    selected=[]
    for idx,role in enumerate(('low','ordinary')):
        selected.extend([r for r in hard['cases'] if r['kind']==role][:CASES['hard'][idx]])
    whole=[dict(card=c,kind='unfiltered') for c in data['assess'][:CASES['whole']]]
    seen={r['card']['id'] for r in selected+whole}
    practice={c['id'] for c in _read(bank.PRACTICE)['cards']}
    if seen&practice or len(seen)!=len(selected)+len(whole) or len(roles)<len(selected):
        raise ValueError('The native training/evaluation release boundary changed')
    plan=dict(schema='tt95-public-velocity-1',train=_hash(bank.MODEL),lessons=_hash(ROOT/'lesson-receipt.json'),
              stock_dynamics=_hash(bank.inv.MODEL),mount_dynamics=_hash(tilted.MODEL),stock_origin=_hash(tt90.SEASON.CARRY),
              body=_hash(tt90.system.BUILD),frozen_carried_practice=_hash(D90),checkpoint=_hash(tt90.bank('practice-dose16')),
              direct=selected,whole=whole,arms=['frozen','normal','full'],cap_hard=3,cap_whole=240,
              controls='Both fitted arms keep the same trained inverse, patch, pose and six target selection budget. '
                       'Both start from the same fitted stock and mounted memories and freeze those. Only the '
                       'normal ablation erases velocity perpendicular to the face normal. Fixed same-body mounts, '
                       'ordinary native releases and the full randomized physical opponent. Score real rises only.')
    if not PLAN.exists():
        _write(PLAN,plan)
    if _read(PLAN)!=plan:
        raise ValueError('Changed matched public velocity evaluation')
    return plan


def played(r,ref):
    delivered=r['placements']['delivered']
    first=delivered[0] if delivered else None
    matched=bool(first and all(same_trace(first[k],ref[k]) for k in ('far_contact','near_bounce'))) if ref else None
    return dict(first_landed=bool(first and first['near_landed']),
                first_answered=bool(first and first['near_landed'] and len(delivered)>1 and delivered[1]['near_bounce']),
                matched=matched)


def run(group):
    p=freeze()
    m,ready,reg=tt90.material()
    ready=np.asarray(ready)
    schedule=p[group]
    dest=ROOT/(group+'-matched.json.gz')
    keep={x['card']['id'] for x in [*p['direct'],*p['whole']]}
    snapshot=_read(D90)
    shots=mpc.measured_launches(tt90.FEEDS)
    d=_read(dest) if dest.exists() else dict(protocol=_hash(PLAN),rows=[],complete=False)
    if d['protocol']!=_hash(PLAN):
        raise ValueError('Public velocity adjudication contract changed')
    if d['complete']:
        return
    with np.load(mpc.BANK) as f:
        raw=tt90.mount.LocalField(mpc.DeepField(f,'local'),reg)
    rank=tt90.SEASON.fresh.shot.Coverage(_read(tt90.SEASON.fresh.RISK))
    cap=p['cap_hard'] if group=='direct' else p['cap_whole']
    with tt90.school.enabled(),tt90.system.installed(m['v'],m['mount']),mpc.centered(ready):
        a=mpc.arena()
        cad=mpc.arena(width=64,height=48,detail='cad')
        cam,rig,stride,lo,hi=mpc.resources(a)
        clear=tt90.school.physical.GripClearance(cad.model,stock=True)
        try:
            with patch.object(mpc,'Fragments',lambda b,e:tt90.mount.Carried(b,e,practice=[tt90.COURSE.PRACTICE],held=keep)):
                arms={k:Pupil(mpc.BANK,raw,register=reg,variant='elastic',budget=28,coverage=rank,
                              mode=k,exclude=keep) for k in p['arms']}
            for k,a0 in arms.items():
                a0.priors.home=ready.copy()
            try:
                for entry in schedule[len(d['rows']):]:
                    c=entry['card']
                    results={}
                    for k,a0 in arms.items():
                        origin=tt90.prior(a0)
                        tt90.COURSE.resume(a0,tt90.bank('practice-dose16'),dict(snapshot,rows=origin['rows']+snapshot['rows']))
                        run=tt90.checked_play(a,cam,rig,stride,lo,hi,ready,c,shots[c['id']],raw,a0,'frozen',cap)
                        r0=played(run,entry.get('offer'))
                        r0.update(patch_choices=a0.patches,protected=a0.patch_protected,
                                  exact_event_sha256=digest_json(run['events']))
                        run.update(r0)
                        add=any(x['chosen'] for x in a0.patches)
                        if add or group=='direct':
                            run['cad']=tt90.validate_game(a,c,shots[c['id']],run,clear)
                        results[k]=run
                    landing=[]
                    for k in p['arms']:
                        r=results[k]
                        landing.append((k,r['score']['learner'],r['first_landed'],r['first_answered'],
                                       sum(bool(n['chosen']) for n in r['patch_choices']),r['matched']))
                    d['rows'].append(dict(case=entry,results=results))
                    d['complete']=len(d['rows'])==len(schedule)
                    _write(dest,d)
                    print('TT95',group,c['id'],entry['kind'],landing,flush=True)
            finally:
                for a0 in arms.values():
                    a0.close()
        finally:
            cam.close();cad.close();a.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('plan','direct','whole'))
    s=cli.parse_args().stage
    freeze() if s=='plan' else run(s)

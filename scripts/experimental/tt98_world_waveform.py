"""Physically teach the canted SO-101 whole ordinary world-paddle waveforms."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import tt63_fixture_qualification as qualify
from . import tt72_retarget_trajectory as transport
from . import tt84_installation_system as body
from . import tt84_installation_play as play
from . import tt57_motor_school_transfer as optical
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt31_curriculum import card, opening_candidates
from .tt37_optical_clinic import OpticalClinic
from .tt37_opponent import draw
from .tt37_opponent_audit import guarded, verify_clearance
from .tt39_carry_joints import Carried


ROOT=Path('artifacts/tt98_world_waveform')
PLAN=ROOT/'protocol.json'
TEACH=ROOT/'source-strokes.json.gz'
PRACTICE=ROOT/'transported-practice.json.gz'
MODEL=ROOT/'routine-motions.npz'
HELD=ROOT/'held-forks.json.gz'
GAME_PLAN=ROOT/'games.protocol.json'
SOURCE=body.TRAIN
ASSESS=body.ROOT/'season-prefix32.json.gz'
MP=body.MP
SCHOOL=body.fixture.school
H=SCHOOL.H
SCALE=np.array([.13,.09,.072,.49,.66,.69,.42,.47,.48,.39,.51])
MIN_PROJECTED=1.052
IN_Y=(-1.20,-.56)


def registration():
    g=_read(body.BUILD); r=g['rows'][g['chosen']]
    return r['candidate'],r['commission']['ready']


def freeze():
    trainer=_read(SOURCE); test=_read(ASSESS); exclusions={c['id'] for c in test['cards']}
    exclusions.update(c['id'] for c in body.contract()['physical'])
    pool=[r for r in body.dataset(trainer,first=True) if r['stock_landed'] and r['height']>=1.055 and r['id'] not in exclusions]
    chosen=[]; seen=set()
    for lane in (-.27,.24,-.09,.08,-.34,.32,.0,.0,-.20,.20,-.12,.15):
        ready=[r for r in pool if r['id'] not in seen]
        if not ready:break
        v=min(ready,key=lambda s:abs(s['x']-lane)+.26*abs(s['height']-1.085));seen.add(v['id']);chosen.append(dict(v,source=str(SOURCE)))
    teststat=[]
    for i,(c,r) in enumerate(zip(test['cards'],test['rows'],strict=True)):
        for g,o in enumerate(r['placements']['delivered']):
            if not all(o.get(k) for k in ('far_contact','near_bounce','receive_line')):continue
            y=o['receive_line']; h=body.fixture.projected_height(y)
            last=o['near_contact']['step'] if o['near_contact'] else o['near_bounce']['step']+25
            options=[k for k in r['searches'] if k.get('p') and not k.get('robust',0) and k['bounced']
                         and IN_Y[0]<k['p'][1]<IN_Y[1] and o['near_bounce']['step']<=k['tick']<last-3]
            if not options:continue
            role=('miss' if not o['near_landed'] and h>1.048 else 'routine' if o['near_landed'] and 1.054<h<1.14
                  else 'low' if h<1.018 else None)
            if role:teststat.append(dict(i=i,id=c['id'],leg=g,tick=options[0]['tick'],height=h,source=str(ASSESS),role=role,
                                          x=y['xyz'][0],scored=bool(o['near_landed']),checksum=digest_json([o['far_contact'],o['near_bounce']])))
    held=[]; excluded=set()
    for role in ('miss','routine','low'):
        group=[r for r in teststat if r['role']==role]
        nodes=[]
        for lane in (-.18,.18,0.,-.24) if role!='miss' else (-.23,.23,0.,.16,.24):
            choices=[r for r in group if r['id'] not in excluded]
            if not choices:break
            found=min(choices,key=lambda r:(abs(r['x']-lane),r['tick']));nodes.append(found);excluded.add(found['id'])
        held.extend(nodes)
    if len(chosen)<6 or len({r['id'] for r in held}&seen) or not {'miss','routine','low'}<={r['role'] for r in held}:
        raise ValueError('Need disjoint ordinary practice and powered routine and low controls')
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema='tt98-world-waveform-transfer-1',sources={str(p):_hash(p) for p in (SOURCE,ASSESS,body.BUILD,SCHOOL.course.OPTIC)},
          practice=chosen,held=held,gate=dict(height=MIN_PROJECTED,y=IN_Y),
          boundary='First fit complete old-body world paddle trajectories and their local angular/translation velocity in private training only. Calibrate with a finite actual mounted native search from the same physical receive input. Keep only repeatably CAD-clear, answered returns. Frozen test inputs and low controls use only prior RGB and encoders. Observer cannot alter robust stock moves or predicted low interactions. Ordinary game actions are signed native joint rates.'))
    print('disjoint docket',[(x['id'],round(x['height'],3)) for x in chosen],[(x['role'],x['id'],x['leg']) for x in held],flush=True)


def freeze_games(feeds):
    if GAME_PLAN.exists():raise ValueError('Do not overwrite the independent whole-point roster')
    p=plan(); tt90=Path('artifacts/tt90_installation_learning/protocol.json')
    sources=(play.SEASON.CARRY,play.TT80,play.PLAN,PLAN,body.BUILD,SCHOOL.course.OPTIC,tt90)
    contract=body.contract(); excluded={r['id'] for r in p['practice']+p['held']+contract['physical']+contract['archive']}
    for path,key in ((play.TT80,'practice'),(play.PLAN,'cards'),(play.SEASON.CARRY,'cards'),
                     *[(tt90,k) for k in ('learn','assess','hard_pool')]):
        z=_read(path); excluded.update(c['id'] for c in (z[key][:len(z['rows'])] if path==play.SEASON.CARRY else z[key]))
    rng=np.random.default_rng(9838751)
    pool=[r['id'] for r in opening_candidates(MP.measured_launches(feeds)) if r['id'] not in excluded]
    cards=[dict(card(int(i),'full',j,int(rng.integers(1,2**30))),profile='reach',opening='random',reply_seed=int(rng.integers(1,2**30)))
           for j,i in enumerate(rng.permutation(pool)[:8])]
    if len(cards)!=8:raise ValueError('Not enough independent whole-point opponents')
    _write(GAME_PLAN,dict(schema='tt98-paired-normal-1',sources={str(f):_hash(f) for f in sources},feeds=_hash(feeds),
                          cards=cards,cap=240,excluded=sorted(excluded),
                          scope='Same rigid canted body, ready, 48-point carry, sampled releases, standard broad reactive opponent and assessable online credit. Only the physically trained routine corrective mode differs. Release-disjoint from physical design, the TT84 season, TT90 and TT98 lessons. Every normal point starts once and runs to physical end or benchmark cap.'))
    print('registered whole points',[c['id'] for c in cards],flush=True)


def plan():
    p=_read(PLAN)
    if any(_hash(Path(f))!=h for f,h in p['sources'].items()):raise ValueError('Changed installed body or physical waveforms')
    return p


def harvest():
    p=plan(); old=_read(SOURCE); g=dict(protocol=_hash(PLAN),rows=[],complete=False)
    if TEACH.exists():
        g=_read(TEACH)
        if g['protocol']!=_hash(PLAN):raise ValueError('Changed stock training roster')
    with SCHOOL.enabled(),MP.centered(old['ready']):
        a=MP.arena(); cam,rig,*_=MP.resources(a)
        try:
            for spec in p['practice'][len(g['rows']):]:
                c,r,obs=qualify.get(old,spec); beginning=next(e for e in r['events'] if e['kind']=='separation' and e.get('collider')=='rival_racket')
                start=int(np.ceil(beginning['time']/MP.DT+1e-7))+1
                frames,impacts=transport.capture(a,cam,rig,c,r,obs,r['actions'])
                h=next(x for x in impacts if abs(x['time']-obs['offer']['near_contact']['time'])<1e-7)
                if digest_json(a.events)!=digest_json(r['events'][:len(a.events)]):raise ValueError('Different original physical stock lesson')
                if not a.legal_returns or len(frames)<start+20:raise ValueError('Stock lesson ended before its known paid waveform')
                g['rows'].append(dict(case=spec,capture=frames,impact=h,fork=start,stock_event=digest_json(a.events)))
                g['complete']=len(g['rows'])==len(p['practice']);_write(TEACH,g)
                print('physical source',c['id'],len(frames),start,round(h['time'],3),flush=True)
        finally:cam.close();a.close()


def targeted(ik,r,ready):
    d=transport.Course(r); hit=d.impact; fits=[]
    for phase in (-90.,0.,90.,180.):
        for clock in (-1,0,1):
            a=clock*MP.DT; p,R,T=(v[0] for v in d.target(np.array([hit['time']+a]),a,1.))
            R=R@Rotation.from_euler('z',phase,degrees=True).as_matrix()
            starts=[ready[:5],np.array(hit['q'][:5]),np.array(hit['q'][:5])+np.array([.34,0,.17,0.,-.31])]
            q,_=min((ik.fit_frame(p,R,j,tangent=.014,stay=.0001,n=26) for j in starts),key=lambda z:z[1]);
            e=ik.errors(q,p,R,T); quality=8.*e[0]+.010*e[1]+.17*e[3]+.020*e[4]
            fits.append(dict(offset_ticks=clock,phase_deg=phase,q=q.tolist(),tangent=.014,error=e,quality=quality))
    return d,sorted(fits,key=lambda z:z['quality'])[:6]


def mounted_lessons():
    p=plan(); old=_read(SOURCE); dat=_read(TEACH); m,home=registration(); home=np.asarray(home)
    if not dat['complete'] or dat['protocol']!=_hash(PLAN):raise ValueError('Physically replay stock lessons before calibration')
    rec=dict(protocol=_hash(PLAN),true_sources=_hash(TEACH),rows=[],complete=False)
    if PRACTICE.exists():
        rec=_read(PRACTICE)
        if rec['protocol']!=_hash(PLAN) or rec['true_sources']!=_hash(TEACH):raise ValueError('Changed calibration sources')
    with SCHOOL.enabled(),body.installed(m['v'],m['mount']),MP.centered(home):
        a=MP.arena(); cam,rig,stride,lo,hi=MP.resources(a); cad=MP.arena(detail='cad')
        try:
            clear=qualify.GripClearance(cad.model,stock=True); kin=transport.FullPose(a); oracle=Rollout(a,1)
            for spec in dat['rows'][len(rec['rows']):]:
                c,row,anchor=qualify.get(old,spec['case']); prefix=[transport.ZERO]*spec['fork']
                qualify.tape.setup(a,c,row,anchor)
                for u in prefix:a.exchange(u,MP.FAR_COMMAND)
                if digest_json(a.events)!=digest_json([e for e in row['events'] if e['time']<=a.data.time+1e-8]):
                    raise ValueError('Calibration changed the original powered far strike')
                course,fits=targeted(kin,spec,home)
                goals,meta=transport.goals_for(kin,course,a,fits,gains=(.78,1.04),leads=(.023,.057,.088))
                f=np.zeros((len(goals),9));u,y=oracle.trajectories(goals,qualify.AIM,course.contact_time-a.data.time+.21,feedback=f)
                value,legal=MP.cost(y,f); picks=[]; issued=[]
                for i in np.argsort(value):
                    if f[i,1]<.5:continue
                    path=qtrim(u[i],meta[i]['hit']+19)
                    if any(np.linalg.norm(path[:min(len(path),len(x))]-x[:min(len(path),len(x))])<.18 for x in issued):continue
                    if len(picks)>=3:break
                    issued.append(path); tape=prefix+path.tolist()+[transport.ZERO]*139
                    hit=qualify.tape.matched_motion(a,c,row,anchor,tape,clear)
                    hit['model']=dict(legal=bool(legal[i]),radius=float(f[i,8]),loss=float(value[i]),fit=meta[i])
                    picks.append(hit)
                    if qualify.received(hit):
                        trace,impacts=transport.capture(a,cam,rig,c,row,anchor,hit['commands'])
                        if digest_json(a.events)!=digest_json(hit['events']):raise ValueError('Earned mounted trial failed optical replay')
                        hit['trace']=trace
                        hit['world_patch']=next(j for j in impacts if abs(j['time']-hit['contact']['time_s'])<1e-7)
                        break
                rec['rows'].append(dict(case=spec['case'],fits=fits,modeled_legal=int(legal.sum()),modeled_contact=int(np.sum(f[:,1]>.5)),actual=picks))
                rec['complete']=len(rec['rows'])==len(dat['rows']);_write(PRACTICE,rec)
                print('mounted',c['id'],int(legal.sum()),[(v['same'],v['landed'],v['answered'],v['clearance']['penetration_samples']) for v in picks],flush=True)
        finally:cam.close();cad.close();a.close()


def qtrim(u,n):return np.array(u[:n],copy=True)


def horizon_height(x):
    t=np.clip((-1.53-x[1])/(.98*min(-1.1,x[4])),0.,.37)
    return float(x[2]+t*x[5]-4.905*t*t)


def gate(x,bounce):
    return bool(x is not None and bounce and x[4]<-.91 and IN_Y[0]<x[1]<IN_Y[1] and horizon_height(x)>=MIN_PROJECTED)


def window(x,q):return np.r_[x,q[:5]]/SCALE


def fit():
    d=_read(PRACTICE)
    if not d['complete'] or d['protocol']!=_hash(PLAN):raise ValueError('Calibrate actual native lessons before distilling')
    rows=[]
    for x in d['rows']:
        for arm in x['actual']:
            if not qualify.received(arm) or 'trace' not in arm:continue
            impact=arm['contact']['time_s']/MP.DT
            for k,w in enumerate(arm['trace']):
                p=w['rgb_pv'];
                if p is None or not 5.<impact-k<23. or w['time']<x['case']['first_bounce']:continue
                if not gate(np.array(p),True):continue
                r=np.asarray(w['ref']); path=MP.tail(np.asarray([n['ref'] for n in arm['trace'][k+1:k+1+H]]),H)-r
                hit=arm['world_patch']; rows.append((window(p,np.asarray(w['q'])),p,r,path,x['case']['id'],
                                                    hit['q'],hit['ball'][1],impact-k))
    if len({x[-1] for x in rows})<3:
        raise ValueError(f'Insufficient independent paid routine motions: {len(rows)} clocks, {len({x[-1] for x in rows})} releases')
    x,p,ref,u,release,hit_q,hit_y,clocks=zip(*rows,strict=True)
    np.savez_compressed(MODEL,keys=x,x=p,ref=ref,paths=u,ids=release,hit_q=hit_q,hit_y=hit_y,clocks=clocks,
                        practice=_hash(PRACTICE),protocol=_hash(PLAN))
    print('routine montage',len(x),len(set(release)),flush=True)


class Archive:
    def __init__(self,venue):
        with np.load(MODEL) as f:self.layout={k:f[k].copy() for k in f.files}
        if str(self.layout['practice'])!=_hash(PRACTICE):raise ValueError('Changed paid mounted action bank')
        self.index=cKDTree(self.layout['keys']); self.geom=transport.FullPose(venue)

    @staticmethod
    def flight(x,y):
        t=np.clip((y-x[1])/(.98*min(-1.1,x[4])),.0,.40)
        return x[:3]+t*x[3:]+np.array([0.,0.,-4.905*t*t]),t

    def moved(self,x,px,i):
        z=self.layout; want,clock=self.flight(x,z['hit_y'][i]); old,_=self.flight(px,z['hit_y'][i])
        motion=np.clip(want-old,[-.086,-.028,-.064],[.086,.028,.064])
        if np.linalg.norm(motion)<.007:return None
        q=z['hit_q'][i][:5]; _,frame,linear,ang=self.geom.frame(q,True); n=frame[:,2]
        d,_resid,_rank,_sv=np.linalg.lstsq(np.vstack([linear,.12*(np.eye(3)-np.outer(n,n))@ang,.013*np.eye(5)]),
                                            np.r_[motion,np.zeros(8)],rcond=None)
        d=np.clip(d,np.maximum(-.24,self.geom.low-q),np.minimum(.24,self.geom.high-q))
        if np.linalg.norm(linear@d-motion)>np.linalg.norm(motion)*.72:return None
        # Move the entire local stroke onto this ball line. Constant shift
        # through contact retains the original first-order joint twist there.
        progress=np.clip(np.arange(1,H+1)/np.clip(.57*clock/MP.DT,3.,12.),0.,1.)
        blend=progress*progress*(3.-2.*progress)
        return blend[:,None]*np.r_[d,0.]

    def propose(self,x,q,r):
        z=self.layout; out=[];meta=[];ids=set()
        distances,ix=self.index.query(window(x,q),min(len(z['ids']),48))
        for d,i in zip(np.atleast_1d(distances),np.atleast_1d(ix),strict=True):
            ident=int(z['ids'][i]); px=z['x'][i]
            if d>3.3:break
            if ident in ids or abs(px[0]-x[0])>.16 or abs(horizon_height(px)-horizon_height(x))>.091:continue
            speed=lambda v:np.clip((-1.52-v[1])/min(-1.4,v[4]),.04,.37)
            velocity=np.clip(speed(px)/speed(x),.77,1.28)
            path=SCHOOL.warp(z['paths'][i],velocity,0.)+.52*np.minimum(np.arange(1,H+1)/10.,1)[:,None]*np.clip(z['ref'][i]-r,-.30,.30)
            out.append(path);meta.append(dict(release=ident,distance=float(d),jacobian=False)); ids.add(ident)
            shift=self.moved(x,px,i)
            if shift is not None:out.append(path+shift);meta.append(dict(release=ident,distance=float(d),jacobian=True))
            if len(ids)==4:break
        return out,meta


def held(n):
    p=plan(); m,ready=registration(); old=_read(ASSESS); out=dict(protocol=_hash(PLAN),weights=_hash(MODEL),rows=[],complete=False)
    if HELD.exists():
        o=_read(HELD)
        if any(o[k]!=v for k,v in out.items() if k not in ('rows','complete')):raise ValueError('Changed matched routine test')
        out=o
    with SCHOOL.enabled(),body.installed(m['v'],m['mount']),MP.centered(ready):
        a=MP.arena(); cam,rig,*_=MP.resources(a); cad=MP.arena(detail='cad');
        try:
            clear=qualify.GripClearance(cad.model,stock=True); library=Archive(a)
            if {v['id'] for v in p['held']}&set(library.layout['ids'].tolist()):raise ValueError('Waveform evidence leaked a held release')
            for s in p['held'][len(out['rows']):n]:
                i=s['i']; t=s['tick']; c,row=old['cards'][i],old['rows'][i]; target=row['placements']['delivered'][s['leg']]
                if digest_json([target['far_contact'],target['near_bounce']])!=s['checksum']:raise ValueError('Changed held matched offer')
                eye,x,q,ref,_,feat,hist=SCHOOL.observed_prefix(a,cam,rig,c,row,t,a._arm_ids[1][2]); point=SCHOOL.State(a);
                baseline=SCHOOL.physical(a,point,c,row,s,row['actions'][t:],clear); n0=baseline['stock_prefix_events']
                if digest_json(baseline['events'])!=digest_json(row['events'][n0:n0+len(baseline['events'])]) or bool(baseline['landed'])!=s['scored']:
                    raise ValueError('Original ordinary/low held tape did not reproduce')
                clinic=OpticalClinic(body=True)
                try:
                    allow=gate(x,bool(eye.eye.latest is not None and eye.eye.latest.bounced)); pick=None; evidence=None; ceiling=[]
                    if allow:
                        optical.warm_clinic(clinic,rig,a.control_dt,hist,eye,t,q,ref)
                        paths,notes=library.propose(x,q,ref);pick,evidence=optical.choose(clinic,paths,notes,ref)
                        if paths and pick is None and clinic.sync():
                            # Diagnostic only: no hidden evidence or action
                            # from this counterfactual can enter whole games.
                            u,_,f,loss,_=SCHOOL.native(clinic.predict,np.asarray(paths),ref)
                            for j in np.argsort(loss)[:4]:
                                tape=np.vstack([u[j,:H],np.zeros((138-H,6))])
                                oracle=SCHOOL.physical(a,point,c,row,s,tape,clear)
                                if oracle.get('unmatched'):raise ValueError('Candidate changed the matched offer')
                                ceiling.append(dict(proposal=notes[j],nominal_contact=bool(f[j,1]>.5),physical=oracle))
                    actual=baseline
                    if pick is not None:
                        cmds=np.vstack([pick,np.zeros((138-H,6))]); actual=SCHOOL.physical(a,point,c,row,s,cmds,clear)
                        if actual.get('unmatched'):raise ValueError('Issued routine correction altered the exact incoming ball')
                    out['rows'].append(dict(case=s,admitted=allow,projected=None if x is None else horizon_height(x),issued=pick is not None,
                                            original=baseline,transport=actual,decision=evidence,unissued_diagnostic=ceiling))
                    out['complete']=len(out['rows'])==len(p['held']);_write(HELD,out)
                    print('matched',s['role'],s['id'],allow,pick is not None,baseline['landed'],actual['landed'],actual['answered'],flush=True)
                finally:clinic.twin.close()
        finally:cam.close();cad.close();a.close()


class WorldPlayer(play.Registered):
    """A disjoint high post-bounce recall, secondary to the existing normal policy."""
    def __init__(self,*args,library,actuator,**kwargs):
        super().__init__(*args,**kwargs)
        self.wave=library
        self.wave_stride,self.wave_low,self.wave_high=actuator

    def begin(self):
        super().begin()
        self.wave_path=None; self.wave_tick=-H; self.wave_log=[]

    def search(self,tick,x,q,reference,base):
        super().search(tick,x,q,reference,base)
        if not self.notes or self.notes[-1]['tick']!=tick or self.notes[-1].get('robust',0)>0:
            return
        if not gate(x,self.engine.bounced):return
        paths,metadata=self.wave.propose(x,q,reference)
        choose,evidence=optical.choose(self.engine,paths,metadata,reference)
        self.wave_log.append(dict(tick=tick,height=horizon_height(x),issued=choose is not None,**evidence))
        if choose is not None:
            self.wave_path=np.clip(reference+np.cumsum(choose,axis=0)*self.wave_stride,self.wave_low,self.wave_high)
            self.wave_tick=tick
            self.best=self.wave_path; self.tick=tick; self.lease=H

    def step(self,tick,eye,x,q,reference,base):
        if self.wave_path is not None and 0<=tick-self.wave_tick<H and (x is None or x[4]<.9):
            self.engine.observe(tick,eye,q,reference.copy())
            return self.wave_path[tick-self.wave_tick]
        self.wave_path=None
        return super().step(tick,eye,x,q,reference,base)


def games(mode,feeds):
    p=_read(GAME_PLAN)
    if any(_hash(Path(f))!=h for f,h in p['sources'].items()) or p['feeds']!=_hash(feeds):raise ValueError('Changed whole-point conditions')
    if mode=='transport':
        held_cases=_read(HELD)
        if not held_cases['complete'] or held_cases['weights']!=_hash(MODEL):raise ValueError('Finish the release-disjoint whole-waveform assay')
    build=_read(body.BUILD); m,ready=registration()
    reg=play.Registration(build['rows'][0]['candidate']['mount']['base'],m['mount']['base'],m['mount']['rotation'])
    ids={c['id'] for c in p['cards']}
    with np.load(MP.BANK) as f:raw=play.LocalField(MP.DeepField(f,'local'),reg)
    rank=play.SEASON.fresh.shot.Coverage(_read(play.SEASON.fresh.RISK)); shots=MP.measured_launches(feeds)
    out=ROOT/f'{mode}.json.gz'; bank=ROOT/f'{mode}.npz'; carry=_read(play.SEASON.CARRY)
    result=dict(protocol=_hash(GAME_PLAN),match=_hash(HELD) if mode=='transport' else None,params=_hash(MODEL) if mode=='transport' else None,
                mode=mode,cards=p['cards'],cap=p['cap'],
                mounting=m,ready=ready,rows=[])
    if out.exists():
        previous=_read(out)
        if any(previous[k]!=v for k,v in result.items() if k!='rows'):raise ValueError('Changed game comparison')
        result=previous
    with SCHOOL.enabled(),body.installed(m['v'],m['mount']),MP.centered(ready):
        a=MP.arena(); cad=MP.arena(detail='cad'); cam,rig,stride,lo,hi=MP.resources(a)
        clearance=qualify.GripClearance(cad.model,stock=True)
        try:
            lib=Archive(a) if mode=='transport' else None
            if lib is not None and ids&set(lib.layout['ids'].tolist()):raise ValueError('The grounded motion library used an assessment release')
            with patch.object(MP,'Fragments',lambda b,e:Carried(b,e,practice=[play.COURSE.PRACTICE],held=ids)):
                shared=dict(register=reg,variant='elastic',budget=28,coverage=rank)
                actor=(WorldPlayer(MP.BANK,raw,library=lib,actuator=(stride,lo,hi),**shared) if mode=='transport'
                       else play.Registered(MP.BANK,raw,**shared))
            try:
                actor.priors.home=np.array(ready); play.COURSE.resume(actor,play.SEASON.CARRY.with_suffix('').with_suffix('.npz'),carry)
                if result['rows']:
                    if _hash(bank)!=result['checkpoint_sha256']:raise ValueError('Changed learned-game checkpoint')
                    play.COURSE.resume(actor,bank,dict(result,rows=carry['rows']+result['rows']))
                for c in p['cards'][len(result['rows']):]:
                    guarded(a,clearance); launch=a.launches
                    r=MP.play(a,cam,rig,stride,lo,hi,ready,c,shots[c['id']],raw,actor,play.COURSE.OPTIC,'live',p['cap'],progress=20)
                    if a.launches!=launch+1 or any(o['draw']!='random' or o['target']!=list(draw(c,j,'random',None)[0])
                                                 for j,o in enumerate(r['placements']['requested'])):
                        raise ValueError('Changed standard continuously random reactive opponent')
                    verify_clearance(a);r['clearance']=a.clearance_result()
                    if r['score']['learner']>=100:
                        witness=play.audit_witness(r,c,a,clearance,mode,len(result['rows']),'tt98-'+mode,ready,m)
                        r['verified_witness']=dict(path=str(witness),sha256=_hash(witness))
                    r['corrective_mode']=getattr(actor,'wave_log',[])
                    result['rows'].append(r);result['checkpoint_sha256']=play.COURSE.save(actor,bank);_write(out,result)
                    print('ordinary game',mode,c['id'],r['score']['learner'],sum(m['issued'] for m in r['corrective_mode']),flush=True)
            finally:actor.close()
        finally:cam.close();cad.close();a.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('mode',choices=('plan','games-plan','harvest','learn','fit','held','normal','transport'))
    a.add_argument('--n',type=int,default=64);a.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    p=a.parse_args()
    if p.mode in ('normal','transport'):games(p.mode,p.feeds)
    else:{'plan':freeze,'games-plan':lambda:freeze_games(p.feeds),'harvest':harvest,'learn':mounted_lessons,'fit':fit,'held':lambda:held(p.n)}[p.mode]()

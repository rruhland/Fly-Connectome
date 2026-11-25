"""Reanchor native learned ordinary strokes by their absolute mounted contact impulse."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import tt98_world_waveform as world
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json


ROOT=Path('artifacts/tt104_reanchor_impulse')
PREFIX=world.body.ROOT/'season-prefix64.json.gz'
PLAN=ROOT/'protocol.json'
OUT=ROOT/'held.json.gz'
MP=world.MP
SCHOOL=world.SCHOOL
H=world.H
CLOCK=np.arange(1,H+1)


def gate(x,bounced):
    return bool(x is not None and bounced and x[4]<-.91 and -1.295<x[1]<-.53 and world.horizon_height(x)>=1.051)


def freeze():
    if PLAN.exists():raise ValueError('Do not overwrite the new release-disjoint test')
    d=_read(PREFIX); p=world.plan(); excluded={x['id'] for x in p['held']+p['practice']+world.body.contract()['physical']}
    candidates=[]
    for i,(c,r) in enumerate(zip(d['cards'],d['rows'],strict=True)):
        if c['id'] in excluded:continue
        for j,o in enumerate(r['placements']['delivered']):
            if not all(o.get(k) for k in ('far_contact','near_bounce','receive_line')):continue
            line=o['receive_line']; h=world.body.fixture.projected_height(line)
            role=('miss' if not o['near_landed'] and h>1.045 else 'routine' if o['near_landed'] and h>1.052
                  else 'low' if h<1.02 else None)
            if role is None:continue
            end=o['near_contact']['step'] if o.get('near_contact') else o['near_bounce']['step']+22
            notes=[z for z in r['searches'] if z.get('p') and z['bounced'] and not z.get('robust',0)
                   and -1.295<z['p'][1]<-.53 and o['near_bounce']['step']<=z['tick']<end-3]
            if not notes:continue
            candidates.append(dict(role=role,i=i,leg=j,id=c['id'],tick=notes[0]['tick'],height=h,x=line['xyz'][0],
                                   scored=bool(o['near_landed']),checksum=digest_json([o['far_contact'],o['near_bounce']])))
    groups={s:[r for r in candidates if r['role']==s] for s in ('miss','routine','low')}
    seen=set(); held={k:[] for k in groups}
    misses={r['id'] for r in groups['miss']}
    searches={'routine':(-.15,.18),'miss':(-.24,.24,0.,-.12,.14),'low':(-.21,.21,0.,.05)}
    for role in ('routine','miss','low'):
        for x in searches[role]:
            choices=[r for r in groups[role] if r['id'] not in seen]
            if not choices:break
            v=min(choices,key=lambda k:(role=='routine' and k['id'] in misses,abs(k['x']-x),k['i'],k['tick']))
            seen.add(v['id']);held[role].append(v)
    if len(held['miss'])<3 or len(held['routine'])!=2 or len(held['low'])<3:
        raise ValueError('Not enough distinct release groups on the new ordinary/low roster')
    ROOT.mkdir(parents=True,exist_ok=True)
    sources=(PREFIX,world.PLAN,world.PRACTICE,world.MODEL,world.body.BUILD,SCHOOL.course.OPTIC)
    _write(PLAN,dict(schema='tt104-retime-world-impulse-1',sources={str(f):_hash(f) for f in sources},
                     held=[r for k in ('miss','routine','low') for r in held[k]],excluded=sorted(excluded),
                     actor='Only previously paid TT98 mounted movement labels and public RGB/encoder state. Retimes at the actually paid hit plane, re-establishes absolute arm references, adapts face and tangent speed at the new ball, then verifies candidate control through the optical twin. Never routes low balls or overrides a robust existing ordinary plan.'))
    print('absolute retarget roster',[(r['role'],r['id'],r['leg']) for k in held for r in held[k]],flush=True)


def smooth(t):
    z=np.clip(t,0.,1.)
    return z*z*(3-2*z)


def normal_rotation(n,target):
    # Smallest proper rotation, preserving the winding of the learned stroke.
    cross=np.cross(n,target); s=float(np.linalg.norm(cross))
    if s<1e-10:return np.eye(3)
    angle=np.clip(np.arctan2(s,n@target),0.,.20)
    return Rotation.from_rotvec(cross*angle/s).as_matrix()


class Anchor(world.Archive):
    def __init__(self,venue):
        super().__init__(venue)
        self.paid={}
        for row in _read(world.PRACTICE)['rows']:
            for act in row['actual']:
                if not world.qualify.received(act) or 'world_patch' not in act:continue
                hit=act['world_patch']
                end=next(e for e in act['events'] if e['kind']=='separation' and e.get('collider')=='learner_racket'
                         and e['time']>=hit['time'])
                self.paid[row['case']['id']]=dict(x=np.array(hit['ball']),p=np.array(hit['p']),R=np.array(hit['R']),
                    twist=np.array(hit['twist']),q=np.array(hit['q'])[:5],qd=np.array(hit['qd'])[:5],
                    incoming=np.array(hit['incoming']),out=np.array(end['velocity']))
        self.step=venue.control_dt
        self.speed=np.minimum(venue._joint_increments[venue._arm_ids[1][2][:5]]*venue.rate_scale/venue.model.opt.timestep,
                              world.transport.NOMINAL_NO_LOAD)

    def relabel(self, paid, p, v, blend, radius):
        raw=paid['R'][:,2]; sign=1 if raw[1]>0 else -1; old=raw*sign
        integral=paid['out']-paid['incoming']; target=paid['out']-v
        u=integral/max(np.linalg.norm(integral),.01); w=target/max(np.linalg.norm(target),.01)
        n=old+blend*(w-u)
        R=normal_rotation(old,n/np.linalg.norm(n))@paid['R']
        offset=paid['R'].T@(paid['p']-paid['x'])
        center=p+R@(offset*np.array([radius,radius,1.]))
        desired=np.array(paid['twist'],copy=True)
        desired[3:]+=R[:,2]*(R[:,2]@(paid['incoming']-v))*.43*blend
        return center,R,desired

    def absolute(self,i,paid,row,ref,mix,rad,phase):
        z=self.layout
        time=max(.05,float(row[0])); hit_ticks=time/self.step
        alpha=float(np.clip(z['clocks'][i]/hit_ticks,.59,1.68))
        prior=z['ref'][i]
        shifted=self.relabel(paid,row[1:4],row[4:7],mix,rad)
        p,R,T=shifted
        candidate,error=self.geom.fit_frame(p,R,paid['q'],tangent=.025,stay=.005,n=25)
        if error>.065:return None
        # Fit the shifted frame's translation and angular rate jointly.
        _,_,L,_=self.geom.frame(candidate,True)
        new_velocity=self.geom.velocity(candidate,T,self.speed)
        physical_speed=float(np.linalg.norm(L@new_velocity-T[3:]))
        if physical_speed>.98:return None
        # Absolute old-action references are recovered as the learned arc is
        # approached. The initial offset is exactly zero in command space.
        w=smooth(CLOCK/(.82*hit_ticks))
        rebound=np.clip(1.-((CLOCK-hit_ticks)/10.)**2,0.,1.)
        base=prior+SCHOOL.warp(z['paths'][i],alpha,phase)
        base+=(ref-prior)*(1-w[:,None])
        base[:,:5]+=w[:,None]*(candidate-paid['q'])
        rate=np.clip(new_velocity-alpha*paid['qd'],-3.2,3.2)
        base[:,:5]+=((CLOCK-hit_ticks)*self.step*rebound*w)[:,None]*rate
        base[:,:5]+=(rebound*w)[:,None]*.053*rate
        result=base-ref
        return result,dict(eta=round(time,4),mix=mix,radius=rad,phase=float(phase),
                           frame_error_m=float(error),velocity_error=float(physical_speed))

    def propose(self,x,q,ref,engine):
        if not engine.sync():return [],[]
        a=engine.twin; arc=a.partner.flight.path(a.ball.copy(),a.velocity.copy(),a.spin.copy(),1,bounced=True,horizon=.42)
        if not len(arc):return [],[]
        z=self.layout; nearby=[]
        for ident,paid in self.paid.items():
            around=np.flatnonzero(z['ids']==ident)
            j=int(np.argmin(np.abs(arc[:,2]-paid['x'][1]))); t,p=arc[j,0],arc[j,1:4]
            if not .065<t<.395 or abs(p[1]-paid['x'][1])>.035:continue
            delta=p-paid['x']; metric=np.linalg.norm(delta/np.array([.19,.13,.10]))
            if metric>1.75:continue
            k=min(around,key=lambda i:abs(z['clocks'][i]*self.step/t-1.)+.12*np.linalg.norm(world.window(x,q)[6:]-z['keys'][i,6:]))
            nearby.append((metric,int(ident),int(k),j))
        paths=[]; meta=[]
        for distance,ident,i,j in sorted(nearby)[:4]:
            for clock in (-2,0,2):
                row=arc[np.clip(j+int(round(clock*self.step/a.model.opt.timestep)),0,len(arc)-1)]
                if not .052<row[0]<.411:continue
                for mix,rad,phase in ((.0,1.,0.),(.65,.65,0.),(1.,.44,1.1)):
                    new=self.absolute(i,self.paid[ident],row,ref,mix,rad,phase)
                    if new is None:continue
                    goal,info=new
                    paths.append(goal);meta.append(dict(release=ident,distance=float(distance),**info))
        return paths,meta


def screen(engine,paths,meta,ref):
    if not paths or not engine.sync():return None,dict(proposed=len(paths),extended=0),[]
    out=np.asarray(paths); desc=list(meta); e=engine
    u,_,f,loss,legal=SCHOOL.native(e.predict,out,ref)
    # Preserve pose-family structure while exploring bounded clock/velocity
    # uncertainty of only those seeds whose native face comes close.
    ix=np.argsort(loss)[:min(5,len(out))]
    new=[]; notes=[]
    rng=np.random.default_rng(104614)
    for i in ix:
        for scale,phase in ((.89,0.),(1.12,0.),(1.,-1.4),(1.,1.4)):
            for jitter in (False,True):
                path=SCHOOL.warp(out[i],scale,phase)
                n=dict(desc[i],stretch=scale,warp_start=phase,jitter=jitter)
                if jitter:
                    kick=np.r_[rng.normal(0.,[.063,.054,.052,.068,.076]),0.]
                    path+=smooth(CLOCK/11.)[:,None]*kick
                new.append(path);notes.append(n)
    full=np.vstack([out,new]); names=desc+notes
    pick,evidence=world.optical.choose(e,list(full),names,ref)
    evidence.update(proposed=len(paths),extended=len(new),anchor_contact=int(np.sum(f[:,1]>.5)),anchor_legal=int(np.sum(legal)))
    return pick,evidence,[(u[j,:H],meta[j],bool(f[j,1]>.5)) for j in ix[:3]]


def run():
    p=_read(PLAN); d=_read(PREFIX); m,ready=world.registration()
    if any(_hash(Path(f))!=h for f,h in p['sources'].items()):raise ValueError('Changed protocol commitment')
    out=dict(protocol=_hash(PLAN),rows=[],complete=False)
    if OUT.exists():
        out=_read(OUT)
        if out['protocol']!=_hash(PLAN):raise ValueError('Changed exact qualification')
    with SCHOOL.enabled(),world.body.installed(m['v'],m['mount']),MP.centered(ready):
        a=MP.arena(); cam,rig,*_=MP.resources(a); cad=MP.arena(detail='cad'); planner=Anchor(a)
        if set(planner.paid)&{r['id'] for r in p['held']}:raise ValueError('A held release entered the controller')
        try:
            clear=world.qualify.GripClearance(cad.model,stock=True)
            for s in p['held'][len(out['rows']):]:
                i,t=s['i'],s['tick']; c,r=d['cards'][i],d['rows'][i]; o=r['placements']['delivered'][s['leg']]
                if c['id']!=s['id'] or digest_json([o['far_contact'],o['near_bounce']])!=s['checksum']:
                    raise ValueError('The held reactive offer changed')
                eye,x,q,ref,_,_,hist=SCHOOL.observed_prefix(a,cam,rig,c,r,t,a._arm_ids[1][2]); point=SCHOOL.State(a)
                orig=SCHOOL.physical(a,point,c,r,s,r['actions'][t:],clear); k=orig['stock_prefix_events']
                if (digest_json(orig['events'])!=digest_json(r['events'][k:k+len(orig['events'])])
                        or orig['landed']!=s['scored']):raise ValueError('The unmodified prefix should replay exactly')
                yes=gate(x,eye.eye.latest is not None and eye.eye.latest.bounced)
                trial=orig; log=None; pick=None; diagnostics=[]; clinic=world.OpticalClinic(body=True)
                try:
                    if yes:
                        world.optical.warm_clinic(clinic,rig,a.control_dt,hist,eye,t,q,ref)
                        paths,meta=planner.propose(x,q,ref,clinic)
                        pick,log,unissued=screen(clinic,paths,meta,ref)
                        for u,annotation,model_hit in (unissued if pick is None else [(pick,log,True)]):
                            v=SCHOOL.physical(a,point,c,r,s,np.vstack([u,np.zeros((138-H,6))]),clear)
                            if v.get('unmatched'):raise ValueError('A proposed stroke altered an already received offer')
                            if pick is None:diagnostics.append(dict(model_hit=model_hit,plan=annotation,trial=v))
                            else:trial=v
                finally:clinic.twin.close()
                out['rows'].append(dict(case=s,admitted=yes,projected=None if x is None else world.horizon_height(x),
                                        issued=pick is not None,original=orig,absolute=trial,decision=log,diagnostic=diagnostics))
                out['complete']=len(out['rows'])==len(p['held']);_write(OUT,out)
                print('reanchor',s['role'],s['id'],yes,pick is not None,orig['landed'],trial['landed'],trial['answered'],log,
                      [(x['trial']['landed'],x['trial']['actual'] is not None) for x in diagnostics],flush=True)
        finally:cam.close();a.close();cad.close()


class Reanchored(world.WorldPlayer):
    def search(self,tick,x,q,reference,base):
        world.play.Registered.search(self,tick,x,q,reference,base)
        if not self.notes or self.notes[-1]['tick']!=tick or self.notes[-1].get('robust',0)>0:return
        if not gate(x,self.engine.latest is not None and self.engine.latest.bounced):return
        paths,metadata=self.wave.propose(x,q,reference,self.engine)
        choice,evidence,_=screen(self.engine,paths,metadata,reference)
        self.wave_log.append(dict(tick=tick,height=world.horizon_height(x),issued=choice is not None,**evidence))
        if choice is not None:
            self.wave_path=np.clip(reference+np.cumsum(choice,axis=0)*self.wave_stride,self.wave_low,self.wave_high)
            self.wave_tick=tick
            self.best=self.wave_path;self.tick=tick;self.lease=H


def games(feeds):
    p=_read(PLAN); h=_read(OUT); fresh=_read(world.GAME_PLAN)
    if h['protocol']!=_hash(PLAN) or not h['complete'] or any(_hash(Path(f))!=d for f,d in p['sources'].items()):
        raise ValueError('The physical reanchoring qualification must be complete and unchanged')
    if {r['id'] for r in p['held']}&{c['id'] for c in fresh['cards']}:
        raise ValueError('An exact assessment ball may not enter a fresh normal point')
    world.games('transport',feeds,root=ROOT,qualification=OUT,actor_class=Reanchored,archive_class=Anchor)


def paired():
    world.paired(ROOT)
    paths=(world.ROOT/'normal.json.gz',ROOT/'transport.json.gz')
    base,treat=(_read(p) for p in paths)
    effects=[]
    for c,a,b in zip(treat['cards'],base['rows'],treat['rows'],strict=True):
        for issue in (u for u in b['corrective_mode'] if u['issued']):
            t=issue['tick']; out=[]
            for x in (a,b):
                offered=x['placements']['delivered']
                j=max(i for i,o in enumerate(offered) if o.get('near_bounce') and o['near_bounce']['step']<=t)
                k=offered[j]; own=k.get('near_contact')
                departure=next((e for e in x['events'] if own is not None and e['kind']=='separation'
                                and e.get('collider')=='learner_racket' and e['time']>=own['time']),None)
                next_ball=offered[j+1] if j+1<len(offered) else None
                far=next_ball['far_contact'] if next_ball else None
                land=next((e for e in x['events'] if departure and e['kind']=='legal_return' and e['side']==1
                           and e['time']>=departure['time'] and (not far or e['time']<far['time'])),None)
                prior=world.play.audit([e for e in x['events'] if e['time']<=MP.DT*t+1e-8],opening_side=1)['learner']
                last=offered[-1]; crossing=last.get('receive_line')
                out.append(dict(same_ball_key=digest_json([k['far_contact'],k['near_bounce']]),leg=j,own_before=prior,
                                outgoing=departure,landing=land,answered=bool(next_ball and next_ball['near_bounce']),
                                following_offer=next_ball,terminal=x['events'][-1],final_ball_touched=bool(last.get('near_contact')),
                                final_height=world.body.fixture.projected_height(crossing) if crossing else None,
                                score=x['score']['learner'],clearance=x['clearance']))
            effects.append(dict(release=c['id'],tick=t,features=issue,
                                same_matched_incoming=out[0]['same_ball_key']==out[1]['same_ball_key'],
                                condition_order=['normal','reanchored'],result=out))
    _write(ROOT/'issued-effects.json',dict(sources={str(p):_hash(p) for p in paths},effects=effects))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=('plan','held','games','paired'))
    a.add_argument('--feeds',type=Path,default=Path('/tmp/tt25-rallies.json'))
    args=a.parse_args()
    {'plan':freeze,'held':run,'games':lambda:games(args.feeds),'paired':paired}[args.command]()

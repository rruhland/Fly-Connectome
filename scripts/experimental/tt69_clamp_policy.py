"""Fit and physically qualify a same-body RGB/encoder receiving motor field."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt69_clamp_school as school
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BallFlight, BladeIK
from .tt37_opponent import draw
from .tt37_opponent_audit import guarded, verify_clearance
from .tt39_opportunity_map import projected_height
from .tt40_long_low import same_trace


ROOT=school.ROOT
MODEL=ROOT/'motor-field.json.gz'
QUAL=ROOT/'qualification.protocol.json'
OUT=ROOT/'first-receives.json.gz'
GAMES=ROOT/'games.json.gz'
mpc=school.mpc
course=school.course
fixture=school.fixture
register=school.registered
season=register.season
AXIS=-1.45
SCALE=np.array([.154,.089,.63,.86,.88])
MIN_DISTANCE=1.25


def roster():
    seed=school.checked(); prior=_read(season.CARRY)
    avoid={c['id'] for c in prior['cards'][:len(prior['rows'])]}
    avoid.update(c['id'] for c in seed['ordered_practice'])
    avoid.update(c['id'] for c in _read(register.PLAN)['cards'])
    p=_read(fixture.PLAN); avoid.update(c['id'] for k in ('development','held') for c in p[k])
    meta=_read(mpc.BANK.with_suffix('.json'))
    avoid.update(r['card']['id'] for r in meta.get('practice',[]) if 'card' in r)
    old=_read(course.SOURCE)
    pool={c['id']:c for c in old['assessment'] if c['profile']=='reach' and c['id'] not in avoid}
    rng=np.random.default_rng(690204); order=rng.permutation(list(pool))
    cards=[dict(pool[int(i)],seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))) for i in order]
    if len(cards)<12:raise ValueError('Insufficient measured releases disjoint from parent and teacher')
    _write(QUAL,dict(practice=_hash(school.PLAN),carry=_hash(season.CARRY),fixture=_hash(fixture.GEOMETRY),
                     ready=seed['ready'],dx=school.DX,first=cards[:12],whole=cards[12:18],portable_full_game=cards[18:24],
                     preserve=dict(maximum_consecutively_unsupported=5,expected_relative_margin_rad=.30),
                     law='New independent broad draw on each whole table opening. The new receiving field remains frozen and may not choose a stream. No release in new school, body fixture teacher, carried bank or TT68 comparison may enter.'))
    print('qualification',[c['id'] for c in cards[:12]], 'whole',[c['id'] for c in cards[12:18]],flush=True)


def p_arc(flight,eye,tick,x):
    latest=eye.latest
    if latest is None or tick*mpc.DT-latest.time>.079 or x[4]>=-.90:
        return np.empty((0,10))
    use=getattr(eye,'filter',None)
    spin=use.rotation(tick*mpc.DT) if hasattr(use,'rotation') else np.zeros(3)
    return flight.path(x[:3],x[3:6],spin,1,bounced=bool(latest.bounced),horizon=.98)


def point_at(path,y):
    if not len(path) or not path[:,2].min()-.023<y<path[:,2].max()+.023:
        return None
    return path[int(np.argmin(abs(path[:,2]-y)))].copy()


def marker(row):
    return np.r_[row[1],row[3:7]]


def physical_paid():
    stages=school.dev_tapes(); practice=_read(school.OUT)
    if practice['protocol_sha256']!=_hash(school.PLAN):raise ValueError('Wrong shifted private school')
    for r in practice['rows']:
        a=school.paid(r)
        if a:
            stages.append(dict(card=r['card'],shot=r['shot'],leg=r['leg'],controls=a['controls'],events=a['events'],
                              placements=a['placements'],contact=a['contact'],source='TT69 practice real wait'))
    if _read(QUAL)['practice']!=_hash(school.PLAN):raise ValueError('Separate optical docket changed')
    held={c['id'] for k in ('first','whole','portable_full_game') for c in _read(QUAL)[k]}
    if held & {r['card']['id'] for r in stages}:raise ValueError('Held launch in practice pupil')
    return stages


def capture(a,cam,rig,ik,flight,raw):
    c=raw['card']; L=raw['leg']; o=raw['placements']['delivered'][L]
    hit=o['near_contact']['time']; far=o['far_contact']['step']; close=int(np.ceil(hit/a.control_dt))
    axis=a._arm_ids[1][2]; stride=a._joint_increments[axis]*(a.control_dt/a.model.opt.timestep)
    lo,hi=a._joint_low[axis],a._joint_high[axis]
    a.launch(raw['shot'],card=c); a.partner.reset(seed=c['seed']); eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    observations=[]; archive=[]
    for t,u in enumerate(raw['controls']):
        q=a.joints.copy(); r=a.data.ctrl[axis].copy()
        im=cam.capture() if t%2==0 and far-28<=t<=close+7 else None
        x=mpc.observed(eye,t,im,q)
        if x is not None and far+2<=t<=close+3 and x[4]<-.88:
            center,n=ik.pose(q[:5]); off=x[:3]-center
            dist=float(np.linalg.norm(off))
            aim=np.clip(r+stride*np.asarray(u),lo,hi)
            observations.append(dict(t=t,x=x.tolist(),q=q.tolist(),ref=r.tolist(),goal=aim.tolist(),fk_gap=dist))
            if -.27<x[1]<.78 and t%2==0:
                path=p_arc(flight,eye.eye,t,x)
                row=point_at(path,AXIS)
                if row is not None:archive.append(dict(t=t,feature=marker(row).tolist(),path=path.tolist(),now=x.tolist(),q=q.tolist()))
        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
        if done or cap:break
    if digest_json(raw['events'])!=digest_json(a.events) or digest_json(raw['placements'])!=digest_json(a.point_placements()):
        raise ValueError(f'Shifted paid source did not replay bit-identically for {c["id"]}:{L}')
    if not archive or len(observations)<9:return None
    # RGB/encoder chooses defended phase by minimum observed ball/blade gap
    # among pre-outbound readings of the acquired motion. Referee merely
    # labels the practice stroke and its causal observation interval.
    end=min((r for r in observations if r['t']>=close-8),key=lambda p:p['fk_gap'],default=None)
    if end is None:return None
    depth=float(end['x'][1]); rows=[r for r in observations if r['t']<=end['t']]
    if len(rows)<8 or not -1.78<depth<-1.04:return None
    early=sorted(archive,key=lambda r:abs(r['now'][1]-.23))[:min(5,len(archive))]
    at=[point_at(np.array(r['path']),depth) for r in early]; at=[p for p in at if p is not None]
    if not at:return None
    depth_coord=np.maximum.accumulate([-r['x'][1] for r in rows])
    _,unique=np.unique(depth_coord,return_index=True); use=[rows[int(i)] for i in unique]
    if len(use)<7:return None
    return dict(id=c['id'],leg=L,source=raw['source'],radius=raw['contact']['radial_m'],key=np.median([r['feature'] for r in early],axis=0).tolist(),
                depth=depth,anchor=marker(np.median(at,axis=0)).tolist(),gate_columns=[r['now'][1] for r in early],
                y=depth_coord[unique].tolist(),x=[r['x'] for r in use],goal=[r['goal'] for r in use],q=[r['q'] for r in use],
                contact_q=end['q'],contact_goal=end['goal'],contact_observation=end['x'],fit_gap=end['fk_gap'],clock_count=len(archive))


def fit():
    p=school.checked(); data=physical_paid(); ready=p['ready']
    bank=dict(protocol=_hash(school.PLAN),qualification=_hash(QUAL),source_practice=_hash(school.OUT),rows=[],missing=[])
    with fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cam,rig,*_=mpc.resources(a); ik=BladeIK(a); flight=BallFlight(a)
        try:
            for t in data:
                use=capture(a,cam,rig,ik,flight,t)
                if use is None:bank['missing'].append(dict(id=t['card']['id'],leg=t['leg']))
                else:bank['rows'].append(use)
                _write(MODEL,bank)
                print('capture',t['card']['id'],t['leg'],None if use is None else (len(use['y']),use['depth'],use['radius']),flush=True)
        finally:cam.close();a.close()


class Field:
    def __init__(self,doc):
        self.rows=[]
        for x in doc['rows']:
            self.rows.append({k:(np.asarray(v) if k in ('key','anchor','y','goal','q','contact_q','contact_goal') else v) for k,v in x.items()})

    def proposals(self,path,x,q,ref):
        key=point_at(path,AXIS)
        if key is None or not .925 < key[3]< 1.055 or not -.32<key[1]<.13:return []
        maps=[]
        data=marker(key)
        for k in self.rows:
            d=data-k['key']; phase=point_at(path,k['depth'])
            if phase is None or not .18<phase[0]<.74 or abs(d[0])>.15 or abs(d[1])>.095:
                continue
            align=abs(np.array([phase[1],phase[3]])-k['anchor'][:2]);
            if align[0]>.11 or align[1]>.078:continue
            # There must be actual actuator time to the taught contact.
            if np.max(abs(k['contact_goal'][:5]-ref[:5]))>phase[0]*4.16-.042:
                continue
            distance=float(np.linalg.norm(d/SCALE)+.15*max(0.,k['radius']-.045)/.03)
            distance+=.18*np.linalg.norm((k['q'][0,:5]-q[:5])/[.55,.60,.54,.70,1.15])
            maps.append((distance,k,phase))
        return sorted(maps,key=lambda r:r[0])


class Receiver(register.Registered):
    def __init__(self,*args,school_field,policy,**kwargs):
        super().__init__(*args,**kwargs)
        self.school_field=school_field; self.policy=policy; self.moves=[]

    def begin(self):
        super().begin()
        self.move=None; self.moves=[]; self.no_more=False; self.motion_clock=None
        self.prev_rgb_tick=-99; self.late_tuning=False; self.intervened_tick=-99

    def adaptation(self,donor,phase):
        home=donor['contact_q'][:5]; _,n,J,D=self.geometry.pose(home,jacobian=True)
        old=donor['anchor']; world=np.array([phase[1]-old[0],0.,phase[3]-old[1]])
        world=np.clip(world,[-.067,0.,-.043],[.067,0.,.043])
        M=np.r_[J,.081*D,.026*np.eye(5)]
        d=np.linalg.lstsq(M,np.r_[world,np.zeros(8)],rcond=None)[0]
        return np.clip(d,-.26,.26)

    def step(self,tick,eye,x,q,ref,base):
        backup=super().step(tick,eye,x,q,ref,base)
        if self.policy=='registered':return backup
        if x is not None and x[4]>.90:
            self.move=None; self.no_more=False; return backup
        if self.move is None and not self.no_more:
            if x is None or x[4]>=-.90 or not -.31<x[1]<.75 or tick-self.prev_rgb_tick<3:
                return backup
            self.prev_rgb_tick=tick
            path=p_arc(self.engine.twin.partner.flight,eye,tick,x)
            hits=self.school_field.proposals(path,x,q,ref)
            if not hits:return backup
            d,lesson,phase=hits[0]
            if d>MIN_DISTANCE:return backup
            correction=self.adaptation(lesson,phase) if self.policy=='transport' else np.zeros(5)
            self.move=dict(lesson=lesson,correction=correction,start=tick,at=tick+float(phase[0]/mpc.DT),last=x.copy(),
                           progress=float(-x[1]),errors=0)
            self.moves.append(dict(tick=tick,donor=lesson['id'],donor_leg=lesson['leg'],distance=float(d),kind=self.policy,
                                   achieved=q.tolist(),key=marker(point_at(path,AXIS)).tolist(),coupling=correction.tolist()))
            self.late_tuning=False
        if self.move is None:return backup
        m=self.move; k=m['lesson']
        # Planned contact clock never comes from a prerecorded game clock.
        if x is None or x[4]>-.7:
            m['errors']+=1
            if m['errors']>5 or tick>m['at']+8:
                self.move=None; self.no_more=True; return backup
            yy=m['progress']-float(m['last'][4])*mpc.DT
        else:
            m['errors']=0; m['last']=x.copy()
            d=float(-x[1])-m['progress']; yy=m['progress']+float(np.clip(d,.003,.111))
        m['progress']=yy
        if tick>m['at']+11 or yy>abs(k['depth'])+.15:
            self.move=None; self.no_more=True; return backup
        if self.policy=='transport' and not self.late_tuning and x is not None and eye.latest is not None and eye.latest.bounced and .56<yy<1.02:
            shift=point_at(p_arc(self.engine.twin.partner.flight,eye,tick,x),k['depth'])
            self.late_tuning=True
            if shift is not None:
                m['correction']=.25*m['correction']+.75*self.adaptation(k,shift)
        goal=np.array([np.interp(yy,k['y'],col) for col in k['goal'].T]);
        gate=np.clip((yy+.26)/.99,0.,1.)
        goal[:5]+=gate*m['correction']; self.intervened_tick=tick
        return goal


def play_actor(a,renderer,rig,stride,lo,hi,actor,raw,ready,c,shot,first):
    a.launch(shot,card=c); a.partner.reset(seed=c['seed']); eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC); actor.begin()
    ref=np.array(ready); home=ref.copy(); actions=[]; visible=[]; engaged=[]; on=False
    for t in range(a.max_steps):
        q=a.joints.copy(); x=mpc.observed(eye,t,renderer.capture() if t%2==0 else None,q)
        goal=home
        if x is not None and x[4]<-.9 and x[1]<-.90:on=True
        if x is not None and (x[4]<-.9 or (on and x[1]<.50)):
            new,w=raw.get(x,q)
            if new is not None:goal=home+w*(new-home)
        g=actor.step(t,eye.eye,x,q,ref,goal)
        u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8); ref=np.clip(ref+stride*u,lo,hi)
        if actor.intervened_tick==t:engaged.append(t)
        if x is not None and x[4]<-.9 and x[1]<.8:
            visible.append((t,x.tolist()))
        actions.append(u.tolist()); _,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND)
        if end or cap or (first and a.rival_returns>=2) or (not first and a.legal_returns+a.rival_returns-1>=240):break
    from .tt29_endurance import audit
    placements=a.point_placements(); score=audit(a.events,opening_side=1)
    if any(v['draw']!='random' or v['target']!=list(draw(c,i,'random',None)[0]) for i,v in enumerate(placements['requested'])):
        raise ValueError('Unmodified random opponent target changed')
    return dict(controls=actions,events=a.events.copy(),placements=placements,score=score,school=actor.moves,
                engaged_ticks=engaged,visible=visible if first else len(visible),shot=shot)


def qualify(n,whole=False):
    if whole:return integrate(n)
    p=_read(QUAL); packet=_read(MODEL); compiled=Field(packet)
    if p['carry']!=_hash(season.CARRY) or packet['qualification']!=_hash(QUAL) or packet['protocol']!=_hash(school.PLAN):
        raise ValueError('Changed optical-only qualification')
    if len(compiled.rows)<8:raise ValueError('Insufficient optical motor evidence')
    cards=p['whole'] if whole else p['first']; path=GAMES if whole else OUT; cards=cards[:n]
    ids={c['id'] for c in cards}; ready=p['ready']; modes=('registered','transport') if whole else ('registered','phase','transport')
    if ids & {c['id'] for c in compiled.rows}:raise ValueError('Teacher learned a held release')
    doc=dict(protocol=_hash(QUAL),model=_hash(MODEL),rows=[],complete=False)
    shots=mpc.measured_launches(school.FEEDS)
    if path.exists():
        old=_read(path)
        if any(old[k]!=doc[k] for k in ('protocol','model')):raise ValueError('Changed frozen pupil')
        doc=old
    with np.load(mpc.BANK) as f:raw=fixture.BodyRelativeRaw(mpc.DeepField(f,'local'),school.DX)
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cad=mpc.arena(width=64,height=48,detail='cad'); cam,rig,stride,lo,hi=mpc.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True)
        with patch.object(mpc,'Fragments',lambda b,e:course.Carried(b,e,practice=[course.PRACTICE],held=ids)):
            actor=Receiver(mpc.BANK,raw,variant='elastic',budget=28,coverage=rank,school_field=compiled,policy='registered')
        actor.priors.home=np.asarray(ready)
        try:
            course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),_read(season.CARRY))
            for c in cards[len(doc['rows']):]:
                results={}
                for mode in modes:
                    actor.policy=mode
                    v=play_actor(a,cam,rig,stride,lo,hi,actor,raw,ready,c,shots[c['id']],not whole)
                    if v['engaged_ticks']:
                        v['clearance']=school.verify(a,c,shots[c['id']],v,guard)
                    results[mode]=v
                base=results['registered']['placements']['delivered']
                for k,v in results.items():
                    came=v['placements']['delivered']; same=bool(base and came and all(same_trace(base[0][s],came[0][s]) for s in ('far_contact','near_bounce')))
                    v['exact_first_ball']=same
                obs=base[0]['receive_line'] if base else None
                doc['rows'].append(dict(card=c,grade=None if obs is None else projected_height(obs),x=None if obs is None else obs['xyz'][0],arms=results))
                doc['complete']=len(doc['rows'])==len(p['whole'] if whole else p['first']);_write(path,doc)
                print('game' if whole else 'first',c['id'],None if obs is None else projected_height(obs),
                      [(name,v['score']['learner'],v['exact_first_ball'],len(v['school']),len(v['engaged_ticks']),v.get('clearance',{}).get('penetration_samples')) for name,v in results.items()],flush=True)
        finally:actor.close();cam.close();cad.close();a.close()


def earned_promotion():
    d=_read(OUT)
    if not d['complete'] or d['protocol']!=_hash(QUAL) or d['model']!=_hash(MODEL):
        raise ValueError('Complete the matched first-offer test before any whole-point use')
    tally={}
    for label in ('phase','transport'):
        gains=[]; damage=[]
        for r in d['rows']:
            base=r['arms']['registered']; other=r['arms'][label]; live=other['placements']['delivered']
            old=bool(base['placements']['delivered'] and base['placements']['delivered'][0]['near_landed'])
            new=bool(live and live[0]['near_landed'])
            if old and not new:damage.append(r['card']['id'])
            if not old and new and other['exact_first_ball'] and len(live)>1 and live[1]['near_bounce'] and other.get('clearance',{}).get('penetration_samples')==0:
                gains.append(r['card']['id'])
        tally[label]=dict(gains=gains,damage=damage)
    eligible=[k for k in tally if tally[k]['gains'] and not tally[k]['damage']]
    return max(eligible,key=lambda k:(len(tally[k]['gains']),k=='transport')) if eligible else None,tally


def integrate(n):
    p=_read(QUAL); compiled=Field(_read(MODEL)); chosen,tally=earned_promotion()
    if chosen is None:
        print('Not promoted',tally,flush=True)
        return
    cards=p['whole']; ids={c['id'] for c in cards}; ready=np.asarray(p['ready']); shot=mpc.measured_launches(school.FEEDS)
    doc=dict(protocol=_hash(QUAL),model=_hash(MODEL),qualified_first_sha256=_hash(OUT),chosen=chosen,comparison=tally,
             rows=[],complete=False,mode='live both arms; direct additions frozen, ordinary carried bank learns each real score')
    names=('registered',chosen)
    if GAMES.exists():
        old=_read(GAMES)
        if any(old[k]!=doc[k] for k in ('protocol','model','qualified_first_sha256','chosen')):
            raise ValueError('Changed matched live experiment')
        doc=old
    with np.load(mpc.BANK) as f:raw=fixture.BodyRelativeRaw(mpc.DeepField(f,'local'),school.DX)
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK)); saved={k:ROOT/f'live-{k}.npz' for k in names}
    parent=_read(season.CARRY)
    with fixture.slid(school.DX),mpc.centered(ready),school.enabled():
        a=mpc.arena(); cad=mpc.arena(width=64,height=48,detail='cad'); cam,rig,stride,lo,hi=mpc.resources(a)
        guard=school.physical.GripClearance(cad.model,stock=True)
        with patch.object(mpc,'Fragments',lambda b,e:course.Carried(b,e,practice=[course.PRACTICE],held=ids)):
            actors={k:Receiver(mpc.BANK,raw,variant='elastic',budget=28,coverage=rank,school_field=compiled,policy=k) for k in names}
        try:
            for name,actor in actors.items():
                actor.priors.home=ready.copy()
                course.resume(actor,season.CARRY.with_suffix('').with_suffix('.npz'),parent)
                if doc['rows']:
                    if _hash(saved[name])!=doc['checkpoints'][name]:raise ValueError('Shifted live bank changed')
                    checkpoint=dict(checkpoint_sha256=doc['checkpoints'][name],rows=[None]*len(parent['rows'])+
                        [r['arms'][name] for r in doc['rows']])
                    course.resume(actor,saved[name],checkpoint)
            for card in cards[len(doc['rows']):n]:
                results={}
                for name,actor in actors.items():
                    r=mpc.play(a,cam,rig,stride,lo,hi,ready,card,shot[card['id']],raw,actor,course.OPTIC,'live',240,20)
                    if any(v['draw']!='random' or v['target']!=list(draw(card,i,'random',None)[0]) for i,v in enumerate(r['placements']['requested'])):
                        raise ValueError('Ordinary opponent law changed')
                    r['school']=actor.moves
                    if r['school']:
                        r['clearance']=school.verify(a,card,shot[card['id']],dict(r,controls=r['actions']),guard)
                    results[name]=r
                m=[]
                for leg,(x,y) in enumerate(zip(*(results[k]['placements']['delivered'] for k in names))):
                    if all(same_trace(x[s],y[s]) for s in ('far_contact','near_bounce')):
                        m.append(dict(leg=leg,old=bool(x['near_landed']),new=bool(y['near_landed'])))
                doc['rows'].append(dict(card=card,arms=results,exact_offers=m))
                doc['checkpoints']={k:course.save(v,saved[k]) for k,v in actors.items()}
                doc['complete']=len(doc['rows'])==len(cards); _write(GAMES,doc)
                print('registered full',card['id'],[(k,v['score']['learner'],len(v['school'])) for k,v in results.items()],flush=True)
        finally:
            for actor in actors.values():actor.close()
            cam.close();cad.close();a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('plan','fit','first','games'));p.add_argument('--n',type=int,default=12)
    a=p.parse_args()
    if a.mode=='plan':roster()
    elif a.mode=='fit':fit()
    else:qualify(a.n,whole=a.mode=='games')

"""Practice bounded motor alternatives on materially distinct, actually observed central replies."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from . import tt51_low_reverse_curriculum as optical
from . import tt54_earned_low_flights as bank
from . import tt54_midline_course as cams
from . import tt40_low_feasibility as physical
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, guarded
from .tt40_long_low import same_trace
from .tt44_paired_credit import State
from .tt54_training_speedups import enabled

ROOT=Path('artifacts/tt56_central_training')
PLAN=ROOT/'protocol.json'
TRAINING=ROOT/'training.json.gz'
HELD=ROOT/'held.json.gz'
SOURCES=bank.ROOT/'central-templates.json'
EARNED=bank.ROOT/'central-earned-replay.json.gz'
TRAIN=(14000,15603,6928,14071)
VALID=(12259,3178,6061,6324)
SCOUTS=12
BANK_GUARD=np.array([.056,.042,.83,.76])


def deep(x):
    x=np.asarray(x)
    t=(-1.53-x[1])/(.98*x[4]);return np.r_[x[0]+t*x[3],x[2]+t*x[5]-4.905*t*t,-x[4],x[5]-9.81*t]


def changed(c,leg,seed):
    return dict(c,**({'seed':int(seed)} if leg==0 else {'reply_seed':int(seed)}))


def plan():
    doc=_read(SOURCES); extra=_read(EARNED)
    source={t['case']['release']:t for t in doc['rows']}
    disallowed=set(_read(bank.ROOT/'central-sources.json')['reserved'])|{4214}
    if (not doc['complete'] or any(not v['actual']['answered'] or v['clearance']['penetration_samples'] for v in extra['rows'])
            or set(TRAIN+VALID)&disallowed or set(TRAIN)&set(VALID)):
        raise ValueError('Unverified or non-disjoint central training and validation')
    rng=np.random.default_rng(562009)
    rows=[]
    for subset,roster in (('train',TRAIN),('held',VALID)):
        for idx in roster:
            s=source[idx];case=s['case'];path=Path(case['source']);old=_read(path)
            c,r=old['cards'][case['index']],old['rows'][case['index']];leg=case['leg']
            request=r['placements']['requested'][leg];p=np.array(request['target'])
            anchor=int(np.floor(request['time_s']/mpc.DT+1e-7))
            proposed=[]
            while len(proposed)<SCOUTS:
                seed=int(rng.integers(1,2**30)); goal,_=draw(changed(c,leg,seed),leg,'random',None)
                if abs(goal[0]-p[0])>.105 or abs(goal[2]-p[2])>.046 or abs(goal[3]-p[3])>.38:
                    continue
                proposed.append(dict(seed=seed,target=list(goal)))
            rows.append(dict(id=idx,split=subset,case=case,card=c,anchor=anchor,
                             original_goal=request['target'],original_phase=s['template']['source_phase_tick'],draws=proposed))
    ROOT.mkdir(parents=True,exist_ok=True)
    optical.save(PLAN,dict(sources_sha256=_hash(SOURCES),earned_sha256=_hash(EARNED),rows=rows,
        limits=dict(batch=72,rounds=2,maximum_variants=2,max_physical_candidates=2),
        role='Different releases for new coaching and holdout. Whole prior stroke reproduced. Only standard TT37 independent goal draws, admitted by a fixed generous request vicinity for scout compute. Branch choice and gap are based on later realized uninterrupted RGB, never an assumed request shift or scored response. Scalar legal-and-answered governs practice storage. No official held release consumed.'))
    print([(r['id'],r['split'],r['anchor']) for r in rows],flush=True)


def photo(eye,a,cam,t):
    q=a.joints.copy();x=mpc.observed(eye,t,cam.capture() if t%2==0 else None,q)
    b=bool(x is not None and x[4]<-.9 and x[1]<-.09 and (eye.eye.latest.bounced or x[5]>.58))
    return q,x,b


def peek(a,actor,cam,c,old,case,anchor,root,eye0,body0,want):
    a.card=dict(c);root.restore(a);leg=case['leg'];a.card=changed(c,leg,want['seed'])
    axes=a._arm_ids[1][2];eye=deepcopy(eye0);cams.restore_body(actor.engine,body0,eye)
    gate=None;rgb=[];end=False
    for t in range(anchor,min(anchor+112,len(old['actions']))):
        q,x,b=photo(eye,a,cam,t);ref=a.data.ctrl[axes].copy()
        if gate is None and b and -.91<x[1]<-.60 and x[2]>.774:
            gate=dict(time=t,eye=deepcopy(eye),body=cams.body_snapshot(actor.engine),native=State(a),
                      first=(q,x,b),ref=ref.tolist())
        if x is not None and b and -1.31<x[1]<-.98:
            rgb.append((t,deep(x).tolist()))
        # Keep the archived normal attained body. The later unexecuted arm
        # outcome must not affect whether this becomes an evaluation input.
        actor.engine.observe(t,eye.eye,q,ref)
        _,_,end,cap,_=a.exchange(old['actions'][t],mpc.FAR_COMMAND)
        if end or cap or (len(rgb)>=3 and x is not None and x[1]<-1.16):
            break
    actual=a.point_placements(); req=actual['requested'][leg:leg+1]
    if req and req[0]['target']!=want['target']:
        raise ValueError('Standard continuous opponent target differs during scout')
    note=dict(want,rgb=rgb,preparing_gate=None if gate is None else gate['time'],
              received=actual['delivered'][leg] if len(actual['delivered'])>leg else None)
    if len(rgb)>=3 and gate is not None:
        cloud=np.array([x for _,x in rgb]);t=cloud[:,:2]
        if np.max(np.ptp(t,axis=0))<.075:
            note['visible_deep']=np.median(cloud,axis=0).tolist()
    return note,gate


def choose(scouts,c0,limit):
    old=np.r_[c0['optical_x'],c0['optical_z']]
    eligible=[r for r in scouts if 'visible_deep' in r and abs(r['visible_deep'][0])<.116
          and .969<r['visible_deep'][1]<1.034 and np.linalg.norm(np.asarray(r['visible_deep'][:2])-old)>.008]
    if not eligible:
        return []
    eligible.sort(key=lambda r: abs(r['visible_deep'][1]-1.000)+.18*abs(r['visible_deep'][0]))
    first=eligible[0]; picked=[first]
    for row in sorted(eligible[1:],key=lambda r:-np.linalg.norm((np.asarray(r['visible_deep'])-first['visible_deep'])/BANK_GUARD)):
        ax=np.asarray(row['visible_deep'])-np.asarray(first['visible_deep'])
        if abs(ax[0])<.008 and abs(ax[1])<.008 and abs(ax[2])<.15:
            continue
        picked.append(row)
        if len(picked)>=limit:break
    return picked[:limit]


def record(a,q,x,b,t,ref,u,res):
    _,_,stride,lo,hi=res
    return dict(tick=t,q=q.tolist(),x=None if x is None else x.tolist(),bounce=b,ref=ref.tolist(),
                next_ref=np.clip(ref+u*stride,lo,hi).tolist())


def issue(a,t,x,q,ref,actor,eye,raw,home):
    desired=home
    if x is not None:
        w,conf=raw.get(x,q)
        if w is not None:desired=home+conf*(w-home)
    return actor.step(t,eye.eye,x,q,ref,desired)


def play(a,actor,res,gate,want,side,card,case,home,check=None,commands=None):
    cam,_,stride,lo,hi=res; axes=a._arm_ids[1][2]; leg=case['leg']
    gate['native'].restore(a);eye=deepcopy(gate['eye']);actor.begin();cams.restore_body(actor.engine,gate['body'],eye)
    if check:
        guarded(a,check)
    else:
        a.sample_clearance(False)
    before=a.data.time; controls=[];observations=[]
    try:
        for tick in range(gate['time'],gate['time']+150):
            q,x,b=gate['first'] if tick==gate['time'] else photo(eye,a,cam,tick)
            ref=a.data.ctrl[axes].copy(); k=tick-gate['time']
            if commands is None:
                g=issue(a,tick,x,q,ref,actor,eye,actor.raw,home)
                u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8)
            else:
                u=np.asarray(commands[k] if k<len(commands) else [0.]*6)
            observations.append(record(a,q,x,b,tick,ref,u,res));controls.append(u.tolist())
            _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
            if done or cap or a.rival_returns>=leg+2:
                break
        c=a.clearance_result();p=a.point_placements()
        if check and c['samples']!=round((a.data.time-before)/a.model.opt.timestep):
            raise ValueError('Physical suffix skipped a strict CAD sample')
        if p['requested'][leg]['target']!=want['target']:
            raise ValueError('Physical return target law changed')
        q=p['delivered'][leg] if len(p['delivered'])>leg else None
        new=bool(q and len(p['delivered'])>leg+1 and p['delivered'][leg+1]['near_bounce'] and q['near_landed'])
        return dict(offer=q,answered=new,landed=bool(q and q['near_landed']),events=a.events,placements=p,controls=controls,
                    log=observations,stock_searches=actor.notes,clearance=c)
    finally:
        a.sample_clearance(False)


def proof(a,ready,card,seed,case,anchor,old,gate,earned,clear):
    arena=mpc.arena(width=64,height=48)
    try:
        arena.launch(old['shot'],card=card);arena.partner.reset(seed=card['seed']);guarded(arena,clear)
        tape=old['actions'][:gate]+earned['controls']
        for t,u in enumerate(tape):
            if t==anchor:arena.card=changed(card,case['leg'],seed)
            if max(abs(np.asarray(u)))>.8+1e-10:
                raise ValueError('Signed rated bounds changed')
            arena.exchange(u,mpc.FAR_COMMAND)
        if any(digest_json(a)!=digest_json(b) for a,b in ((earned['events'],arena.events),
                                                          (earned['placements'],arena.point_placements()))):
            raise ValueError('Independent actual train conversion failed exact replay')
        c=arena.clearance_result()
        if c['samples']!=round(arena.data.time/arena.model.opt.timestep):
            raise ValueError('Full witness skipped physical states')
        return c
    finally:
        arena.close()


def same(a,b):
    return bool(a and b and all(same_trace(a[k],b[k]) for k in ('far_contact','near_bounce')))


def variants(actor,examples,tick,x,q,ref):
    if not actor.engine.sync():return []
    twin=actor.engine.twin
    track=twin.partner.flight.path(twin.ball,twin.velocity,twin.spin,1,bounced=True,horizon=.49)
    if not len(track):return []
    point=track[np.argmin(abs(track[:,2]+1.10))]
    sensed=np.r_[point[1],point[3:7]]
    if abs(sensed[0])>.135 or not .964<deep(np.r_[point[1:4],point[4:7]])[1]<1.043:
        return []
    ranked=[]
    for e in examples:
        spec=e['template']; rows=e['approach']; launch=spec['source_phase_tick']
        if not rows:continue
        e0=min(rows,key=lambda r:abs(r['tick']-launch))
        if e0['x'] is None:continue
        body=min(rows,key=lambda r:abs(r['tick']-launch+max(1,int((spec['depth']-x[1])/x[4]/mpc.DT))))
        gap=np.linalg.norm((np.asarray(spec['p'])-sensed)/[.093,.048,.8,1.04,.83])
        gap+=.41*np.linalg.norm((q[:5]-np.asarray(body['q'][:5]))/[.38,.48,.51,.64,.63])
        if gap<2.4:
            ranked.append((gap,e,body,e0))
    picks=[];seen=set()
    for gap,e,start,final in sorted(ranked,key=lambda a:a[0]):
        if e['release'] in seen:continue
        seen.add(e['release']);s=e['template']
        eta, *later=track[np.argmin(abs(track[:,2]-s['depth']))]
        if not .046<eta<.33:continue
        qc=np.asarray(final['q'][:5]); _,normal,J,N=actor.geometry.pose(qc,jacobian=True)
        error=np.asarray(later[:3])-np.asarray(final['x'][:3]);error=np.clip(error,-.084,.084)
        swing=-(np.asarray(later[3:6])-np.asarray(final['x'][3:6]))/8.0
        input_=np.r_[error,.10*np.clip(swing,-.14,.14)]
        A=np.r_[J,.10*N];dq=A.T@np.linalg.solve(A@A.T+.035**2*np.eye(6),input_)
        dq=np.clip(dq,-.205,.205)
        timeline=(np.arange(mpc.HORIZON)+1)
        relative=np.asarray(s['relative']);refs=np.asarray(s['refs'])
        for offset in (0.,-1.,1.):
            index=timeline-eta/mpc.DT+offset
            nominal=np.column_stack([np.interp(index,relative,s) for s in refs.T])
            ramp=np.clip(timeline/(.68*eta/mpc.DT),0,1)
            motors=nominal.copy();motors[:,:5]+=.79*ramp[:,None]*dq
            if np.max(abs(motors[0,:5]-ref[:5]))<.53:
                picks.append((e['release'],motors,gap))
        if len(seen)>=3:break
    return picks


class Viewer(season.Player):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw);self.repertoire=[]

    def begin(self):
        super().begin();self.augmentation=[]

    def search(self,tick,x,q,reference,base):
        super().search(tick,x,q,reference,base)
        if not self.repertoire or not self.engine.bounced or x[1]>-.64:
            return
        existing=self.best is not None and 0<=tick-self.tick<self.lease
        if existing and len(self.notes) and self.notes[-1].get('robust',0)>0:
            return
        v=variants(self,self.repertoire,tick,x,q,reference)
        if not v:return
        goals=[r[1] for r in v]
        loss,ok,_,y,_=self.trial(goals,long=True)
        keep=np.flatnonzero(ok&(y[:,5]>.028))
        frame=dict(tick=tick,n_tried=len(goals),eligible=len(keep),issued=False)
        self.augmentation.append(frame)
        if not len(keep):return
        keep=keep[np.argsort(loss[keep])[:4]]
        support,quality=self.uncertainty([goals[j] for j in keep],loss[keep])
        if not np.any(support>0):return
        choice=int(keep[np.argmin(np.where(support>0,quality,np.inf))])
        self.best=goals[choice];self.tick=tick;self.lease=mpc.LEASE
        frame.update(issued=True,donor=v[choice][0],distance=v[choice][2],score=float(loss[choice]))


def experiment(stage,out):
    schedule=_read(PLAN); seeds=_read(SOURCES); earned=_read(EARNED)
    if schedule['sources_sha256']!=_hash(SOURCES) or schedule['earned_sha256']!=_hash(EARNED):
        raise ValueError('Reward source bundle changed')
    coach_library=None
    if stage=='held':
        prior=_read(TRAINING)
        if not prior['complete']:
            raise ValueError('Do not open prospective hold until practice concludes')
        novel=prior['templates']; releases={r['release'] for r in novel}
        if len(releases)<2:
            optical.save(out,dict(complete=True,rows=[],status='Fewer than two independently coached release conversions. Held draw set unopened.'))
            return
        base=[dict(t,release=t['case']['release']) for t in seeds['rows'] if t['case']['release'] not in VALID]
        lots=[dict(t) for t in earned['templates'] if t['release'] not in VALID]
        coach_library=base+lots+novel
        if {r['release'] for r in coach_library}&set(VALID):
            raise ValueError('Held sources contributed a trained motor')
    receipt=_read(out) if out.exists() else dict(plan_sha256=_hash(PLAN),stage=stage,rows=[],templates=[],complete=False)
    if receipt.get('stage')!=stage or receipt.get('plan_sha256')!=_hash(PLAN):
        raise ValueError('Stopped branch has a different prospective schedule')
    if stage=='held':receipt['trained_sha256']=_hash(TRAINING)
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,'local')
    for item in [r for r in schedule['rows'] if r['split']==stage][len(receipt['rows']):]:
        case=item['case'];d=_read(Path(case['source']));c,old=d['cards'][case['index']],d['rows'][case['index']]
        if c!=item['card'] or _hash(Path(case['source']))!=case['source_sha256']:
            raise ValueError('Source motion changed')
        home=np.array(d['ready']);leg=case['leg'];anchor=item['anchor'];report=dict(id=c['id'],scouts=[],branches=[])
        with mpc.centered(home),enabled(opponent=False):
            a=mpc.arena();cad=mpc.arena(detail='cad',width=64,height=48);clear=CourtClearance(cad.model,articulated=True)
            res=mpc.resources(a);cam,rig,*_=res; axes=a._arm_ids[1][2]
            actor=Viewer(mpc.BANK,raw,variant='elastic',budget=28,coverage=fresh.shot.Coverage(_read(fresh.RISK)))
            teacher=physical.TapeProbe(a,steering=True) if stage=='train' else None
            try:
                a.launch(old['shot'],card=c);a.partner.reset(seed=c['seed']);actor.begin()
                eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
                for t,u in enumerate(old['actions'][:anchor]):
                    q=a.joints.copy();mpc.observed(eye,t,cam.capture() if t%2==0 and t>=anchor-36 else None,q)
                    actor.engine.observe(t,eye.eye,q,a.data.ctrl[axes].copy());a.exchange(u,mpc.FAR_COMMAND)
                if anchor==0:
                    actor.engine.tick=0
                    actor.engine.reference=a.data.ctrl[axes].copy()
                if len(a.partner.requests)!=leg or digest_json(a.events)!=digest_json([e for e in old['events'] if e['time']<=a.data.time+1e-8]):
                    raise ValueError('Previous physical stroke or opponent boundary changed')
                root=State(a);engine=cams.body_snapshot(actor.engine);starts={}
                for draw_id,p in enumerate(item['draws']):
                    got,g=peek(a,actor,cam,c,old,case,anchor,root,eye,engine,p)
                    if g is not None: starts[p['seed']]=g
                    report['scouts'].append(got|{'draw_index':draw_id})
                picked=choose(report['scouts'],case,schedule['limits']['maximum_variants'])
                print('scout',stage,c['id'],len(picked),[(r['visible_deep'][:2],r['seed']) for r in picked],flush=True)
                for p in picked:
                    gate=starts[p['seed']]
                    actor.repertoire=[]
                    stock=play(a,actor,res,gate,p,'stock',c,case,home)
                    result=dict(scout=p,stock=stock,trained=None)
                    report['branches'].append(result)
                    if stage=='held':
                        actor.repertoire=coach_library
                        test=play(a,actor,res,gate,p,'learned',c,case,home)
                        result['learned']=test|{'candidate_decisions':actor.augmentation}
                        if not same(stock['offer'],test['offer']):raise ValueError('Held comparison changed actual delivered opportunity')
                        if test['answered']:
                            test['full_cad']=proof(a,home,c,p['seed'],case,anchor,old,gate['time'],test,clear)
                        print('held',c['id'],stock['answered'],test['answered'],len(actor.augmentation),flush=True)
                        continue
                    if stock['answered'] or stock['offer'] is None:
                        continue
                    gate['native'].restore(a)
                    q,x,b=gate['first']; tick=gate['time']
                    # The altered real ball's optical phase, not the source's
                    # native hit timestamp, centers the privileged coach's search.
                    arrival=max(.065,min(.35,(-1.58-x[1])/x[4]))
                    guide=dict(old,placements=deepcopy(old['placements']))
                    guide['placements']['delivered'][leg]['near_contact']={'time':float(a.data.time)+arrival}
                    try:
                        a.clearance_margin=.0006
                        options,stats=teacher.search([560321,c['seed'],p['seed']],schedule['limits']['batch'],schedule['limits']['rounds'],guide,tick,leg)
                    finally:a.clearance_margin=0.
                    options.sort(key=lambda z:(z['feedback'][8]>.065,z['cost']+8*z['feedback'][8]))
                    result['trained']=dict(stats=stats,actual=[])
                    for option in options[:schedule['limits']['max_physical_candidates']]:
                        actual=play(a,actor,res,gate,p,'coach',c,case,home,check=clear,commands=option['suffix'])
                        actual['coach_record']={k:v for k,v in option.items() if k!='suffix'}
                        result['trained']['actual'].append(actual)
                        if not same(stock['offer'],actual['offer']):
                            raise ValueError('Coached stroke changed the actual physical offer')
                        if not actual['answered'] or actual['clearance']['penetration_samples']:
                            continue
                        actual['full_cad']=proof(a,home,c,p['seed'],case,anchor,old,tick,actual,clear)
                        if actual['full_cad']['penetration_samples']:continue
                        sample=optical.extract(actual['log'],BladeIK(a),f'actual-{c["id"]}-{p["seed"]}',tick)
                        if sample is None:continue
                        try:sample['p']=cams.cue_prototype(actual['log'],sample)
                        except ValueError:pass
                        receipt['templates'].append(optical.plain(dict(release=c['id'],template=sample,approach=actual['log'])))
                        actual['acquired']=True
                        break
                    print('coach',c['id'],[z['legal'] for z in stats['rounds']],[(x['answered'],x.get('acquired')) for x in result['trained']['actual']],flush=True)
            finally:
                actor.close();cam.close();a.close();cad.close()
        receipt['rows'].append(report)
        optical.save(out,receipt)
    receipt.update(complete=True,status='Released frozen matched physical cases only')
    optical.save(out,receipt)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('plan','train','held'));ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.phase=='plan':plan()
    else:experiment(args.phase,args.out or (TRAINING if args.phase=='train' else HELD))

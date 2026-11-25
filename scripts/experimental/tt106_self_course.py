"""Scalable early trajectory teaching from the canted student's own encountered errors."""

import argparse
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from . import tt90_installation_learning as tt
from . import tt88_inverse_body as motor
from . import tt88_mount_transfer as planted
from .tt29_endurance import audit
from .tt31_curriculum import card,opening_candidates
from .tt38_motorcoach import AIM
from .tt40_long_low import same_trace
from .tt44_paired_credit import State
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=Path('artifacts/tt106_self_course')
PLAN=ROOT/'protocol.json'
PRACTICE=ROOT/'practice.json.gz'
MODEL=ROOT/'student.npz'
HELD=ROOT/'held.json.gz'
mpc=tt.BASE
DT=mpc.DT
ZERO=[0.]*6
N_TRAIN=28
N_HELD=6
TRAIN_CAP=6
DENSITY=2
AMPLIFY=ROOT/'teacher-residual.protocol.json'
QUICK=ROOT/'pilot.protocol.json'
EXPLORE=ROOT/'practice-exploration.protocol.json'
FEATURE_SCALE=np.array([.18,.31,.20,.72,1.00,1.15,.85,.79,.95,.80,.85,
                        .20,.22,.20,.19,.25,2.4,2.4,2.4,2.4,2.4,.17,.11,1.25])


def freeze():
    ROOT.mkdir(exist_ok=True,parents=True)
    plays=tt.contract(tt.FEEDS)
    known=_read(tt.path('practice'))
    if len(known['rows'])!=len(plays['learn']) or known['checkpoint_sha256']!=_hash(tt.bank('practice')):
        raise ValueError('Use the final immutable 40-point baseline carry')
    reserve={c['id'] for s in ('learn','assess','hard_pool') for c in plays[s]} | set(plays['excluded'])
    reserve|={c['id'] for c in _read(tt.COURSE.SOURCE)['assessment']}
    for p,key in ((Path('artifacts/tt101_body_practice/assessment.json'),'cards'),
                  (Path('artifacts/tt73_partner_rehearsal/course/course.json'),'cards')):
        if p.exists():
            reserve|={c['id'] for c in _read(p)[key]}
    # Explicit reservations from the full-waveform/contact impulse study.
    reserve|={12866,12144,3527,10339,9508,12116,5335,14540,12265,4317,9060}
    shots=mpc.measured_launches(tt.FEEDS)
    valid=sorted({c['id'] for c in opening_candidates(shots)}-reserve)
    if len(valid)<N_TRAIN+N_HELD:
        raise ValueError('Insufficient independently reserved measured shots')
    rng=np.random.default_rng(1065301)
    order=rng.permutation(valid)[:N_TRAIN+N_HELD]
    roster=[dict(card(int(i),'full',j,int(rng.integers(1,2**30))),profile='reach',opening='random',
                 reply_seed=int(rng.integers(1,2**30))) for j,i in enumerate(order)]
    d=dict(schema='tt106-on-policy-early-full-motion-1',launch=_hash(tt.FEEDS),tt90_plan=_hash(tt.PLAN),
           carry=_hash(tt.path('practice')),checkpoint=_hash(tt.bank('practice')),build=_hash(tt.system.BUILD),
           train=roster[:N_TRAIN],held=roster[N_TRAIN:],excluded=sorted(reserve),
           rehearsal=dict(epoch=7,cap=TRAIN_CAP,lesson_interval=DENSITY,earliest_camera_y=.80,latest_camera_y=-.47,
                          whole_trajectory_native_screen=84,followup_native_screen=56,actual_candidates=3),
           scope='Independent full measured launches. Student acts without teacher during source practice; '
                 'capture real states before its first near bounce, then show corrected full movement '
                 'from that state only for training. Keep the pupils genuine successes, failed motions and '
                 'physically certified coaching successes. Refit on the pupil’s encountered places every seven points. '
                 'One causal continuous direct learner plus ordinary robust fallback on independent full games.')
    if not PLAN.exists():
        _write(PLAN,d)
    if _read(PLAN)!=d:
        raise ValueError('Changed frozen practice or held course')
    print('TT106 scheduled',len(valid),[x['id'] for x in d['train']],[x['id'] for x in d['held']],flush=True)
    return d


def cue(x,q,qd,ref):
    # Causal trajectory phase and approximate lateral arrival; neither a true
    # intercept nor future bounce/contact is supplied.
    x=np.asarray(x)
    t=float(np.clip((-1.42-x[1])/min(-.92,x[4]),-.05,.65))
    return np.r_[x,np.asarray(q)[:5],(np.asarray(ref)-q)[:5],np.asarray(qd)[:5],
                 x[0]+t*x[3],t,x[5]+9.81*t]/FEATURE_SCALE


class Policy:
    """Local non-backprop joint-rate regression with explicit failed-state support."""

    def __init__(self, path=None):
        self.live=path is not None and path.exists()
        if not self.live:
            return
        with np.load(path) as d:
            self.x,self.u,self.y,self.ep,self.phase=(d[k] for k in ('x','u','y','ep','phase'))
        self.tree=cKDTree(self.x)
        self.good_idx=np.flatnonzero(self.y>0)
        self.good=cKDTree(self.x[self.good_idx])

    def predict(self,x,q,qd,ref):
        if not self.live or x is None or x[4]>-1.:
            return None,0.
        g=cue(x,q,qd,ref)
        d,i=self.tree.query(g,min(70,len(self.y)))
        d,i=np.atleast_1d(d),np.atleast_1d(i)
        keep=d<2.65
        if keep.sum()<7:
            return None,0.
        d,i=d[keep],i[keep]
        w=np.exp(-1.6*d*d)
        fraction=(w@self.y[i]/w.sum())
        z,j=self.good.query(g,min(62,len(self.good_idx)))
        z,j=np.atleast_1d(z),self.good_idx[np.atleast_1d(j)]
        # Fit one signed mode rather than interpolate unrelated forehands
        # and backhands encountered at the same optical position.
        modal=self.u[j[0]]
        keep=(z<2.55)&(np.linalg.norm(self.u[j]-modal,axis=1)<1.25)
        if keep.sum()<6 or len(set(self.ep[j[keep]]))<2:
            return None,0.
        j,z=j[keep],z[keep]
        w=np.exp(-1.8*z*z)
        w/=w.sum()
        mix=self.x[j]-g
        design=np.c_[np.ones(len(j)),mix]
        reg=np.diag(np.r_[.008,np.full(mix.shape[1],.11)])
        fixed=np.linalg.solve(design.T@(w[:,None]*design)+reg,design.T@(w[:,None]*self.u[j]))[0]
        confidence=np.exp(-.38*np.median(z[:min(10,len(z))])**2)*np.clip(.22+.85*fraction,0.,1.)
        return np.clip(fixed,-.8,.8),float(confidence)


def fit(doc,destination):
    out=[]
    for r in doc['rows']:
        for block in r['lessons']:
            for kind in ('on_policy','teacher','close'):
                part=block.get(kind)
                if part is None:
                    continue
                paid=int(part['landed'])
                for x,q,qd,ref,u in part['dense']:
                    out.append((cue(x,q,qd,ref),u[:5],paid,
                                r['id']*20+block['leg'],int(kind=='teacher')))
    if not out or sum(r[2] for r in out)<34:
        return None
    x,u,y,ep,phase=map(np.asarray,zip(*out,strict=True))
    np.savez_compressed(destination,x=x,u=u,y=y,ep=ep,phase=phase)
    return dict(examples=len(x),paid=int(y.sum()),corrections=int(phase.sum()),
                groups=len(set(ep)),model=_hash(destination))


class Coach:
    """Independent training-only search over complete preparation and strike references."""
    def __init__(self,a):
        self.a=a
        self.p=tt.school.physical.low.Probe(a)

    def paths(self,states,table,n,design):
        a=self.a
        q0=a.joints[:5].copy()
        qd0=a.data.qvel[a._near_dofs][:5].copy()
        c=np.vstack([b['q'] for b in states])[table]
        v=np.vstack([b['dq'] for b in states])[table]
        et=np.asarray([b['time'] for b in states])[table]
        move=(np.arange(n)+1)*a.control_dt
        span=np.maximum(.15,et+design[:,15]*.060)
        t=np.clip(move[None,:]/span[:,None],0.,1.14)
        endpoint=c+.16*design[:,:5]
        v=v+1.40*design[:,5:10]
        # An independently conditioned whole early preparation, not merely a late
        # endpoint perturbation. A bounded native roll selects its attainable phase.
        tc=np.minimum(t,1.)
        lift=4.4*tc*(1.-tc)*np.maximum(0,1.-.58*tc)
        q=((2*tc**3-3*tc**2+1)[:,:,None]*q0 + (3*tc**2-2*tc**3)[:,:,None]*endpoint[:,None]
           +span[:,None,None]*((tc**3-2*tc**2+tc)[:,:,None]*qd0+(tc**3-tc**2)[:,:,None]*v[:,None]))
        q += .26*lift[:,:,None]*design[:,None,10:15]
        q += np.clip(move[None,:]-span[:,None],0.,.095)[:,:,None]*v[:,None]
        tau=.026+.033*(.5+.5*np.tanh(design[:,16]))
        q+=tau[:,None,None]*np.gradient(q,a.control_dt,axis=1)
        return np.dstack([q,np.broadcast_to(a.data.ctrl[a._arm_ids[1][2][-1]],q.shape[:2])])

    def propose(self,seed):
        a=self.a
        roots=self.p.basis()
        if not roots:
            return [],[],None
        r=[]
        for b in roots:
            if sum(x['face']==b['face'] and abs(x['depth']-b['depth'])<.015 for x in r)<3:
                r.append(b)
        roots=r[:40]
        n=int(np.ceil((max(b['time'] for b in roots)+1.19)/DT))
        rng=np.random.default_rng(seed)
        choices=[];report=[];near=[]
        n0=84
        ids=np.arange(n0)%len(roots)
        raw=qmc.Sobol(17,scramble=True,seed=int(rng.integers(1,2**30))).random_base2(7)[:n0]
        d=2.*raw-1.
        d[:len(roots)]=0.
        for stage in range(2):
            refs=self.paths(roots,ids,n,d)
            f=np.zeros((len(d),9))
            rates,y=self.p.core.trajectories(refs,AIM,max(b['time'] for b in roots)+.18,feedback=f)
            w,ready=mpc.cost(y,f)
            w+=np.where(f[:,1]>.50,36.*np.maximum(f[:,8]-.052,0),0.)
            p=np.argsort(w)
            report.append(dict(tested=len(d),physically_modeled_legal=int(ready.sum()),contact=int(np.sum(f[:,1]>.5))))
            for k in p[:8]:
                if ready[k]:
                    use=min(n,int(np.ceil((y[k,1]+.14)/DT)))
                    choices.append((float(w[k]),rates[k,:use].tolist(),float(y[k,12])))
                elif f[k,1]>.50 and f[k,3]>1.6 and .009<f[k,8]<.072:
                    use=min(n,int(np.ceil((roots[ids[k]]['time']+.22)/DT)))
                    near.append((float(w[k]),rates[k,:use].tolist(),float(f[k,8])))
            if stage==0:
                elite=p[:14]
                u=np.r_[elite,rng.choice(elite,42)]
                ids=ids[u].copy()
                d=d[u].copy()
                d[14:]+=rng.normal(0,.57,size=(len(d)-14,17))
                d=np.clip(d,-2.1,2.1)
        result=[]
        for w,c,r in sorted(choices,key=lambda z:(z[2]>.058,z[0])):
            if any(np.mean(abs(np.asarray(c[:min(len(c),len(y))])-np.asarray(y[:min(len(c),len(y))])))<.035
                   for _,y,_ in result):
                continue
            result.append((w,c,r))
            if len(result)==3:break
        return result,report,min(near,key=lambda v:v[0]) if near else None


    def residual(self,seed,own,tick,leg):
        """Retrieve an actual missing pupil movement only in the training lab."""
        a=self.a
        original=tt.school.physical.low.TapeProbe(a,steering=True)
        options,history=original.search(seed,72,2,own,tick,leg)
        stages=[dict(tested=r['trials'],physically_modeled_legal=r['legal'],contact=r['contacts'],
                     family='own-motion-correction') for r in history['rounds']]
        arms=[(z['cost'],z['suffix'],z['quality'][12]) for z in options[:2]]
        return arms,stages


def low(c):
    """An inexpensive causal low forecast including a possible near table bounce."""
    x=np.asarray(c)
    time=np.clip((-1.53-x[1])/(.97*min(-.9,x[4])),0.,.72)
    v=x[5]
    power=v*v+19.62*(x[2]-.779)
    b=(v+np.sqrt(max(0.,power)))/9.81
    if 0.<b<time:
        dz=-(v-9.81*b)*.79
        t=time-b
        return .779+t*dz-4.905*t*t
    return x[2]+time*v-4.905*time*time


class Pupil(tt.mount.Registered):
    def __init__(self,*a,policy,weight,trust_gate=.48,**kw):
        super().__init__(*a,**kw)
        self.policy=policy
        self.weight=weight
        self.trust_gate=trust_gate
        self.plant=None
        if self.policy.live:
            self.plant=motor.PlantFit(motor.Body(self.engine.twin),planted.MODEL)

    def begin(self):
        super().begin()
        self.previous_q=None
        self.early=[]
        self.written=[]
        self.recent_try=-1000
        self.tried=0

    def step(self,tick,eye,x,q,ref,base):
        qd=np.zeros(6) if self.previous_q is None else (q-self.previous_q)/DT
        self.previous_q=q.copy()
        if x is not None and x[4]>.9:
            self.tried=0
        if self.plant is not None and x is not None and x[4]<-1. and -.60<x[1]<.70 and .87<low(x)<1.033:
            u,trust=self.policy.predict(x,q,qd,ref)
            if u is not None and trust>=self.trust_gate:
                want=np.r_[ref[:5]+2.2*self.plant.stride*u,ref[5]]
                base=base.copy()
                base[:5]+=self.weight*trust*np.clip(want[:5]-base[:5],-.38,.38)
                self.early.append(tick)
        return super().step(tick,eye,x,q,ref,base)

    def search(self,tick,x,q,ref,base):
        ans=super().search(tick,x,q,ref,base)
        if (self.plant is None or not self.notes or self.notes[-1]['tick']!=tick
                or self.notes[-1].get('chosen') or not (-1.15<x[1]<-.38)
                or tick-self.recent_try<6 or self.tried>=2 or not self.engine.sync()):
            return ans
        self.recent_try=tick;self.tried+=1
        a=self.engine.twin
        path=self.engine.nominal.flight.path(a.ball,a.velocity,a.spin,1,bounced=self.engine.bounced,horizon=.48)
        if not len(path):
            return ans
        from .tt59_fewshot_joints import rank_paths
        lead,lower,upper=tt.library.limits(a)
        want=[]; meta=[]
        for weights in (.76,1.0):
            flat=np.r_[q[:5],self.engine.rate[:5],ref[:5]]
            controls=[]
            confidence=[]
            for i in range(26):
                j=int(np.argmin(abs(path[:,0]-DT*i)))
                velocity=path[j,1:7]
                rates,trust=self.policy.predict(velocity,np.r_[flat[:5],q[5]],np.r_[flat[5:10],0.],np.r_[flat[10:],ref[5]])
                if rates is None:
                    rates=np.clip((base[:5]-flat[10:])/lead[:5],-.62,.62)
                else:
                    confidence.append(trust)
                    aim=np.clip((base[:5]-flat[10:])/lead[:5],-.62,.62)
                    rates=(.34+.56*weights)*rates+(.66-.56*weights)*aim
                flat=self.plant.next(flat[None],rates[None])[0]
                controls.append(np.r_[flat[10:],ref[5]]-ref)
            if len(confidence)>=5:
                want.append(controls)
                meta.append(dict(mode='dense-closed-loop',trust=float(np.mean(confidence)),weight=weights))
        if not want:
            return ans
        result,diag=rank_paths(self.engine,np.asarray(want),ref,meta)
        picked=[(u,v) for u,v in result if v['legal'] and v['robust']]
        logged=dict(tick=tick,issued=bool(picked),**diag)
        if picked:
            u,description=min(picked,key=lambda w:w[1]['score'])
            self.best=tt.library.references(u,ref,lead,lower,upper)[1:]
            self.tick,self.lease=tick,16
            logged['candidate']=description
        self.written.append(logged)
        return ans


def context(x,q,qd,ref,u):
    if x is None or x[4]>-.90 or not -.09<low(x)<1.45 or not -1.77<x[1]<.85:
        return None
    return [x.tolist(),q.tolist(),qd.tolist(),ref.tolist(),list(u)]


def first_choice(a):
    n=a.rival_contacts
    if not n:return None
    for event in reversed(a.events):
        if event['kind']=='legal_return' and event.get('side')==-1:
            # That actual incoming already bounced. Do not claim early prep.
            return None
        if event['kind']=='separation' and event.get('collider')=='rival_racket':
            return n-1,dict(far_contact=True,near_bounce=False)
        if event['kind']=='contact' and event.get('side')==-1:
            return None
    return None


def practice_point(a,cam,rig,ready,c,shot,raw,actor,capture,*,deeper=False):
    a.launch(shot,card=c)
    a.partner.reset(seed=c['seed'])
    a.sample_clearance(False)
    step,lo,hi=tt.library.limits(a)
    eye=mpc.Vision(rig,a.control_dt,raw=True); eye.eye=mpc.LearnedOptics(eye.eye,tt.COURSE.OPTIC)
    actor.begin()
    ref=ready.copy(); prev=None
    attempted={}; commands=[]; readings=[]; visible=False
    for tick in range(a.max_steps):
        q=a.joints.copy()
        qd=(q-prev)/DT if prev is not None else np.zeros_like(q)
        prev=q.copy()
        x=mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
        if x is not None and x[4]<-.9 and x[1]<-.9:
            visible=True
        selected=None
        if capture and x is not None and x[4]<-.96 and -.47<x[1]<.80:
            selected=first_choice(a)
            if selected:
                leg,z=selected
                if leg not in attempted and (len(attempted)<3 or deeper) and z.get('far_contact') and not z.get('near_bounce'):
                    if deeper and len(attempted)>=4:
                        old=sorted(attempted)[2]
                        del attempted[old]
                    attempted[leg]=dict(state=State(a),eye=deepcopy(eye),tick=tick,first=x.copy(),
                                        previous=None if tick==0 else oldq.copy(),prefix=commands.copy(),
                                        observed_y=float(x[1]))
        base=ready
        if x is not None and (x[4]<-.9 or (visible and x[1]<.50)):
            p,conf=raw.get(x,q)
            if p is not None:
                base=ready+conf*(p-ready)
        desired=actor.step(tick,eye.eye,x,q,ref,base)
        u=np.clip((np.clip(desired,lo,hi)-ref)/step,-.8,.8).tolist()
        readings.append(context(x,q,qd,ref,u) if tick%DENSITY==0 else None)
        commands.append(u)
        oldq=q.copy()
        ref=np.clip(ref+step*u,lo,hi)
        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
        if done or cap or (capture and a.legal_returns+a.rival_returns-1>=(12 if deeper else TRAIN_CAP)):
            break
    places=a.point_placements()
    if any(k['draw']!='random' or k['target']!=list(tt.draw(c,i,'random',None)[0])
           for i,k in enumerate(places['requested'])):
        raise ValueError('Changed ordinary playing opponent')
    result=dict(actions=commands,events=a.events.copy(),placements=places,score=audit(a.events,opening_side=1),
                terminal=a.events[-1]['kind'],interventions=len(actor.early),screens=actor.written,
                memory=actor.priors.digest(),launch=shot)
    return result,readings,attempted


def physical_lesson(a,guard,cam,card,entry,leg,commands,target,collect=True):
    from .tt37_opponent_audit import guarded
    b=entry['state']; b.restore(a); guarded(a,guard)
    eye=deepcopy(entry['eye'])
    step,lo,hi=tt.library.limits(a)
    ref=a.data.ctrl[a._arm_ids[1][2]].copy()
    prev=entry['previous']
    hits=[]; data=[]; moved=[]; first=float(a.data.time)
    try:
        with tt.school.contact.physical_contacts(a,hits):
            for j,u in enumerate(commands[:160]):
                tick=entry['tick']+j
                q=a.joints.copy()
                qd=(q-prev)/DT if prev is not None else np.zeros_like(q)
                prev=q.copy()
                x=entry['first'] if not j else (mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q) if collect else None)
                if collect and j%DENSITY==0:
                    v=context(x,q,qd,ref,u)
                    if v is not None:
                        data.append((tick,v))
                ref=np.clip(ref+step*u,lo,hi)
                _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                moved.append(u)
                if done or cap or a.rival_returns>=leg+2:
                    break
        p=a.point_placements()
        offer=p['delivered'][leg] if leg<len(p['delivered']) else None
        if not offer or not all(same_trace(target[k],offer[k]) for k in ('far_contact','near_bounce')):
            return None
        hit=offer.get('near_contact')
        physical=next((z for z in hits if z['side']==1 and hit and abs(z['time_s']-hit['time'])<1e-8),None)
        clear=a.clearance_result()
        if clear['samples']!=round((a.data.time-first)/a.model.opt.timestep):
            raise ValueError('Teacher clearance audit omitted an actual substep')
        if any(k['draw']!='random' or k['target']!=list(tt.draw(card,i,'random',None)[0])
               for i,k in enumerate(p['requested'])):
            raise ValueError('Matched physical teaching changed independent random goals')
        return dict(landed=bool(offer['near_landed']),answered=bool(offer['near_landed'] and len(p['delivered'])>leg+1
                               and p['delivered'][leg+1]['near_bounce']),
                    motion=moved,offer=offer,placements=p,clearance=clear,contact=physical,events=a.events.copy(),
                    dense=[v for tick,v in data if tick<=((hit or {}).get('step',entry['tick']+55))])
    finally:
        a.sample_clearance(False)


def acquire(a,cam,guide,coach,c,path,readings,entries,*,amplify=False,probe=False):
    offers=path['placements']['delivered']
    lessons=[]
    for leg,e in entries.items():
        z=offers[leg] if leg<len(offers) else None
        if not z or not z.get('near_bounce'):
            continue
        cut=e['tick']
        touch=z.get('near_contact')
        last=min(len(readings),touch['step'] if touch else int(z['near_bounce']['step']+32))
        dense=[v for v in readings[cut:last] if v is not None]
        r=dict(leg=leg,tick=cut,opening_y=e['observed_y'],height=None if not z.get('receive_line')
               else float(tt.optics.projected_height(z['receive_line'])),
               on_policy=dict(landed=bool(z['near_landed']),dense=dense),
               target=z,coach=[],teacher=None,close=None,attempts=[])
        if not z['near_landed']:
            e['state'].restore(a);a.sample_clearance(False)
            candidates,report,close=coach.propose([10637,c['seed'],leg])
            r['coach']=report
            def check(trials,family):
                for count,(score,u,radius) in enumerate(trials):
                    branch=physical_lesson(a,guide,cam,c,e,leg,u+[ZERO]*140,z,collect=False)
                    if branch is None:
                        r['attempts'].append(dict(matched=False,family=family))
                        continue
                    r['attempts'].append(dict(matched=True,landed=branch['landed'],answered=branch['answered'],
                                              contact=branch['contact'],clearance=branch['clearance'],family=family))
                    if branch['answered'] and not branch['clearance']['penetration_samples'] and branch['contact']:
                        obs=physical_lesson(a,guide,cam,c,e,leg,u+[ZERO]*140,z,collect=True)
                        if obs is None or digest_json(obs['events'])!=digest_json(branch['events']):
                            raise ValueError('Camera/encoder acquisition changed the real scored response')
                        if len(obs['dense'])>=5:
                            obs.update(prior_controls=e['prefix'],model_radius=radius,lesson_attempt=count,family=family)
                            return obs
                return None
            r['teacher']=check(candidates,'independent-full-spline')
            if amplify and r['teacher'] is None:
                e['state'].restore(a);a.sample_clearance(False)
                other,extra=coach.residual([10664,c['seed'],leg],path,cut,leg)
                r['coach'].extend(extra)
                r['teacher']=check(other,'own-motion-correction')
            if probe and r['teacher'] is None and close is not None:
                v=physical_lesson(a,guide,cam,c,e,leg,close[1]+[ZERO]*140,z,collect=True)
                r['attempts'].append(dict(matched=v is not None,family='tracked-close-practice',
                                          landed=False if v is None else v['landed'],
                                          answered=False if v is None else v['answered']))
                if v is not None and not v['clearance']['penetration_samples'] and v['contact']:
                    v.update(prior_controls=e['prefix'],model_radius=close[2],family='tracked-close-practice')
                    if v['answered'] and len(v['dense'])>=5:
                        r['teacher']=v
                    else:
                        v['physical_landed']=v['landed']
                        v['landed']=False
                        r['close']=v
        lessons.append(r)
    return lessons


def restored(actor):
    initial=tt.prior(actor)
    data=_read(tt.path('practice'))
    tt.COURSE.resume(actor,tt.bank('practice'),dict(data,rows=initial['rows']+data['rows']))


def go(stage,n,partial=False,amend=False):
    p=freeze()
    m,ready,reg=tt.material()
    ready=np.asarray(ready)
    shots=mpc.measured_launches(tt.FEEDS)
    practice=stage=='practice'
    pilot=stage=='pilot'
    dest=PRACTICE if practice else ROOT/'pilot-held.json.gz' if pilot else HELD
    p0=_read(PRACTICE) if PRACTICE.exists() else None
    if not practice and (not p0 or not p0.get('model') or not p0['complete'] and not (partial or pilot)):
        raise ValueError('Finish the student encounter course before independent full games')
    pilot_spec=None
    use_model=MODEL
    seq=p['train'] if practice else p['held']
    if pilot:
        use_model=ROOT/'pilot-student.npz'
        if not QUICK.exists():
            excluded=set(p['excluded']) | {x['id'] for x in p['train']+p['held']}
            possible=sorted({c['id'] for c in opening_candidates(shots)}-excluded)
            rng=np.random.default_rng(106524842)
            order=rng.permutation(possible)[:2]
            picks=[dict(card(int(i),'full',j,int(rng.integers(1,2**30))),profile='reach',opening='random',
                        reply_seed=int(rng.integers(1,2**30))) for j,i in enumerate(order)]
            use_model.write_bytes(MODEL.read_bytes())
            _write(QUICK,dict(protocol=_hash(PLAN),cards=picks,model=_hash(use_model),points=len(p0['rows']),
                              scope='Independent directional pilot, not folded back into training. Final audit uses separate new cards.'))
        pilot_spec=_read(QUICK)
        if pilot_spec['protocol']!=_hash(PLAN) or pilot_spec['model']!=_hash(use_model):
            raise ValueError('Pilot physical policy version changed')
        seq=pilot_spec['cards']
    n=min(len(seq),n)
    signature=dict(protocol=_hash(PLAN),stage=stage,partial=partial,practice_model=None if practice else _hash(use_model))
    if pilot_spec is not None:
        signature['pilot']=_hash(QUICK)
    doc=_read(dest) if dest.exists() else dict(**signature,rows=[],complete=False)
    if any(doc.get(k)!=v for k,v in signature.items()):
        raise ValueError('Changed carried student or course during resume')
    if len(doc['rows'])>=n:
        return
    amend_onset=None
    if practice and amend:
        if not AMPLIFY.exists():
            epoch=len(doc['rows'])
            if epoch<7:
                raise ValueError('Finish the first plain spline epoch before extending the practice teacher')
            _write(AMPLIFY,dict(base=_hash(PLAN),onset=epoch,
                trigger='Only on new actual pupil misses with no paid early full spline. Correct the complete enacted '
                        'trajectory from the pre-bounce fork in the training lab.',
                first_epoch_sha256=digest_json(doc['rows'][:epoch]),batch=72,rounds=2))
        m1=_read(AMPLIFY)
        amend_onset=m1['onset']
        if m1['base']!=_hash(PLAN) or m1['first_epoch_sha256']!=digest_json(doc['rows'][:amend_onset]):
            raise ValueError('Changed the earlier teacher evidence')
    if practice and n>14 and len(doc['rows'])>=14:
        meta=dict(base=_hash(PLAN),first_14_sha256=digest_json(doc['rows'][:14]),start=14,
                  ordinary_held_gate=.48,practice_gate=.28,practice_cap=12,actual_close_probes_per_error=1,
                  rule='Exercise learned low preparation on moderately supported own practice contexts only, '
                       'to collect the learner-induced state distribution. Keep first two and last two early snapshots on '
                       'the longer practice point, giving actual late failures a teaching chance. No evaluation change.')
        if not EXPLORE.exists():
            _write(EXPLORE,meta)
        if _read(EXPLORE)!=meta:
            raise ValueError('Exploration has changed the frozen first-stage phases')
    with np.load(mpc.BANK) as f:
        raw=tt.mount.LocalField(mpc.DeepField(f,'local'),reg)
    coverage=tt.SEASON.fresh.shot.Coverage(_read(tt.SEASON.fresh.RISK))
    exclude={c['id'] for c in p['train']+p['held']+seq}
    with tt.school.enabled(),tt.system.installed(m['v'],m['mount']),mpc.centered(ready):
        a=mpc.arena();cam,rig,*_=mpc.resources(a)
        cad=mpc.arena(width=64,height=48,detail='cad')
        guard=tt.school.physical.GripClearance(cad.model,stock=True)
        coach=Coach(a) if practice else None
        try:
            for j,c in enumerate(seq[len(doc['rows']):n],start=len(doc['rows'])):
                if practice:
                    modes=('encountered',)
                    weight=min(.80,.25+.16*(j//7))
                else:
                    modes=('frozen','student'); weight=.66
                outcomes={}
                for arm in modes:
                    using=((not practice and arm=='student') or (practice and j>=7 and doc.get('model')))
                    with patch.object(mpc,'Fragments',lambda b,e:tt.mount.Carried(b,e,practice=[tt.COURSE.PRACTICE],held=exclude)):
                        actor=Pupil(mpc.BANK,raw,register=reg,variant='elastic',budget=28,coverage=coverage,
                                    policy=Policy(use_model if using else None),weight=weight,
                                    trust_gate=.28 if practice and j>=14 else .48)
                    try:
                        actor.priors.home=ready.copy()
                        restored(actor)
                        digest0=actor.priors.digest()
                        r,trace,entries=practice_point(a,cam,rig,ready,c,shots[c['id']],raw,actor,practice,
                                                       deeper=practice and j>=14)
                        if r['memory']!=digest0:
                            raise ValueError('The ordinary shared carry was no longer frozen')
                        if practice:
                            r['lessons']=acquire(a,cam,guard,coach,c,r,trace,entries,
                                                 amplify=amend_onset is not None and j>=amend_onset,probe=j>=14)
                            r['id']=c['id']
                            doc['rows'].append(r)
                            if (j+1)%7==0 or j+1==len(seq):
                                prior=fit(doc,MODEL)
                                if prior:
                                    doc['model']=prior
                        else:
                            outcomes[arm]=r
                    finally:
                        actor.close()
                if practice:
                    lt=r['lessons']
                    print('TT106 practice',j+1,c['id'],r['score']['learner'],len(lt),
                          [(z['leg'],z['height'],z['on_policy']['landed'],z['teacher'] is not None,
                            sum(k['tested'] for k in z['coach'])) for z in lt],r['interventions'],doc.get('model'),flush=True)
                else:
                    pair=dict(id=c['id'],case=c,results=outcomes)
                    doc['rows'].append(pair)
                    print('TT106 full',j+1,c['id'],[(k,y['score']['learner'],y['interventions'],
                                                   sum(z['issued'] for z in y['screens'])) for k,y in outcomes.items()],flush=True)
                doc['complete']=len(doc['rows'])==len(seq)
                _write(dest,doc)
        finally:
            cam.close();cad.close();a.close()


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('stage',choices=('plan','practice','held','pilot'))
    c.add_argument('--n',type=int,default=28)
    c.add_argument('--partial',action='store_true')
    c.add_argument('--amend',action='store_true')
    a=c.parse_args()
    freeze() if a.stage=='plan' else go(a.stage,a.n,a.partial,a.amend)

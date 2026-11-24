"""A shorter intermediate-depth correction from the original moving stock arm."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import Rollout
from .tt31_ready import centered
from .tt37_bodycoach_windows import Reach
from .tt37_joint_capture import observed
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_motorcoach import AIM,SIZES,answerable,schedules
from .tt39_opportunity_map import projected_height
from .tt40_low_feasibility import Faces
from .tt41_shallow import ROOT
from .tt41_shallow_model import contact_preference,impulse,unit

CASES={
    7038:ROOT/'tt38/joint-mpc-development-elastic.json.gz',
    5864:ROOT/'tt38/joint-mpc-development-elastic.json.gz',
    4563:ROOT/'tt39/credit-novel.json.gz',
}
DEPTHS=(-1.325,-1.365,-1.405,-1.445,-1.48)
STRENGTH=SIZES*np.r_[[.54,.62,.68,.70,.69],[.50,.66,.58,.64,.68], [.66,.78,.80]]
OPTIC=ROOT/'tt37/rebound.npz'


def impact_window(approaches):
    # Native approach records carry ball center at the first contact. The
    # surface contact point is about one ball radius farther toward the blade.
    # Separation positions can already have moved >3 cm and are not a depth check.
    return (approaches[:,18]>.5)&(approaches[:,2]>-1.464)&(approaches[:,2]<-1.297)


def move_center(guard,kin,path,q,reference,change):
    before,normal0=kin.pose(q)
    sign0=1 if normal0[1]>0 else -1
    # This seed follows the *moving* signed branch instead of installing a
    # separate starting posture or selecting another global workspace hub.
    _,normalref=kin.pose(reference)
    options=[]
    for y in DEPTHS:
        choice=path[(path[:,0]>=.052)&(path[:,0]<.36)&(path[:,3]>.854)&(path[:,3]<1.16)]
        if not len(choice) or not choice[:,2].min()-.010<y<choice[:,2].max()+.010:continue
        target=choice[np.argmin(abs(choice[:,2]-y))]
        eta,p,v=float(target[0]),target[1:4],target[4:7]
        for pace,pitch,mix in ((4.12,.04,.10),(4.56,.08,.18),(3.63,.18,.12),
                               (4.26,.02,.42),(4.65,-.045,.29),(3.92,.16,.33)):
            n,push=impulse(p,v,pace,pitch)
            if abs(normalref[1])>.48:
                n=unit((1-mix)*n+mix*np.sign(normalref[1])*normalref)
            wish=p-.024*n
            for start in (q,reference,np.clip(q+.047*change,kin.low,kin.high)):
                qs,(actual,signed,J,G)=kin.fit(wish,sign0*n,start)
                face=sign0*signed;delta=p-actual
                axial=float(delta@face);tang=float(np.linalg.norm(delta-axial*face))
                travel=float(np.max(abs(q-qs)))
                if not (-.009<axial<.062 and tang<.077 and n@face>.89 and travel<eta*4.45+.12):continue
                if any(abs(r['eta']-eta)<.012 and np.max(abs(r['q']-qs))<.10 for r in options):continue
                if not guard.route(q,qs,steps=6):continue
                dq=np.linalg.lstsq(np.r_[J,.076*G,.022*np.eye(5)],np.r_[push,np.zeros(8)],rcond=None)[0].clip(-6.1,6.1)
                options.append(dict(q=qs,dq=dq,eta=eta,position=p,velocity=v,normal=face,sign=sign0,
                    metric=float(5*abs(axial-.024)+3*tang+.15*travel-.04*eta),travel=travel,pose_gap=tang))
    return sorted(options,key=lambda x:x['metric'])[:23]


def solve(engine,pose,reference,dt,seed,*,optical_state,strict):
    if not pose:return None,dict(poses=0,stages=[])
    r=np.random.default_rng(seed)
    n=104;sketch=qmc.Sobol(len(SIZES),scramble=True,seed=int(r.integers(1,2**31))).random_base2(7)[:n]
    params=(2*sketch-1)*STRENGTH
    keys=np.arange(n)%len(pose);params[:len(pose)]=0
    q=np.array([b['q'] for b in pose]);dq=np.array([b['dq'] for b in pose]);eta=np.array([b['eta'] for b in pose])
    end=float(eta.max()+.12);frames=round((end+1.07)/dt);t=(1+np.arange(frames))*dt
    history=[];batch=[];notes=[]
    for index in range(3):
        g=schedules(q[keys],dq[keys],eta[keys],params,dt,frames)
        # Smooth the position/reference handoff during the original reach;
        # the controller’s rate clamp remains active on every solver step.
        blend=np.clip(t[None,:]/np.maximum(.025,.32*eta[keys,None]),0.,1.)
        blend=blend**2*(3.-2.*blend)
        g=reference+blend[:,:,None]*(g-reference)
        f=np.zeros((len(g),9));impact=np.full((len(g),19),np.nan)
        native=engine.predict if optical_state is not None else engine
        rate,y=native.trajectories(g,AIM,end,feedback=f,approaches=impact)
        value,legal,touched=contact_preference(y,f)
        if strict:
            inside=impact_window(impact)
            # A legal stock-depth return isn't useful feedback for moving the
            # real contact earlier. Protect in-band contacts separately.
            value=np.where(inside,3.+value,np.where(impact[:,18]>.5,
                -1.2+np.clip(value,-5.,2.)-7.*np.abs(impact[:,2]+1.395),value))
        else:
            inside=(f[:,6]>-1.525)&(f[:,6]<-1.28)
        legal_in_band=legal&inside
        remains=legal_in_band&(y[:,5]>.014)
        for j in np.flatnonzero(remains):
            b=pose[keys[j]]
            quality=float(answerable(y[j])+.17*b['travel'])
            batch.append((quality,g[j],y[j],f[j],b,impact[j].tolist()))
        notes.append(dict(index=index,trials=len(g),touched=int(touched.sum()),nominal=int(legal.sum()),mid=int(remains.sum()),
                          first_impact_in_band=int(impact_window(impact).sum()),legal_in_band=int(legal_in_band.sum()),
                          best_in_band_clearance=float(y[legal_in_band,5].max()) if legal_in_band.any() else None,
                          minimum_gap=float(f[:,0].min())))
        if len(batch)>=6 or index==2:break
        history.extend(zip(value,keys,params,strict=True))
        elite=sorted(history,key=lambda x:-x[0])[:15]
        n=64;choice=r.integers(len(elite),size=n)
        keys=np.array([elite[i][1] for i in choice])
        params=np.array([elite[i][2] for i in choice])+r.normal(size=(n,13))*STRENGTH*(.57 if index==0 else .38)
        params=np.clip(params,-2.1*STRENGTH,2.1*STRENGTH)
        for k,(_,branch,previous) in enumerate(elite[:5]):keys[k]=branch;params[k]=previous
    note=dict(poses=len(pose),stages=notes)
    if not batch:return None,note
    final=sorted(batch,key=lambda x:x[0])[:7]
    keep=np.ones(len(final),int);rank=np.array([f[0] for f in final])
    if optical_state is not None:
        t0=optical_state;x,v=t0.ball.copy(),t0.velocity.copy()
        spin=t0.data.qvel[t0._ball_dof+3:t0._ball_dof+6];w=spin.copy()
        list_goals=np.array([xx[1] for xx in final])
        try:
            for sign in (-1.,1.):
                t0.ball[:]=x+sign*np.array([.0024,.0045,.003]);t0.velocity[:]=v+sign*np.array([.012,.046,.075])
                spin[:]=w+sign*np.array([17.,8.,11.])
                fb=np.zeros((len(final),9));contact=np.full((len(final),19),np.nan)
                _,out=engine.predict.trajectories(list_goals,AIM,end,feedback=fb,approaches=contact)
                okay=np.isfinite(out[:,0])&(impact_window(contact) if strict else (fb[:,6]>-1.53)&(fb[:,6]<-1.27))
                rank[okay]+=.28*answerable(out[okay]);rank[~okay]+=2.1
                keep+=okay
        finally:
            t0.ball[:]=x;t0.velocity[:]=v;spin[:]=w
    legal=np.flatnonzero(keep>=(2 if optical_state is not None else 1))
    if not len(legal):return None,note|dict(candidate_settings=keep.tolist())
    j=legal[np.argmin(rank[legal])];score,move,out,fb,b,first=final[j]
    note.update(settings=int(keep[j]),cost=score,contact_delay=float(out[1]),outcome=out.tolist(),
                feedback=fb.tolist(),first_impact=first,
                shape={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()})
    return move[:max(1,round((out[1]+.09)/dt))],note


def case(release,output,*,capacity=True,strict=True):
    source=CASES[release];d=_read(source)
    card,row=next((c,r) for c,r in zip(d['cards'],d['rows'],strict=True) if c['id']==release)
    final=row['placements']['delivered'][-1]
    if final['near_contact'] is None or final['near_landed'] or not .99<=projected_height(final['receive_line'])<=1.04:
        raise ValueError('Case is not a physical moderate touched terminal miss')
    home=np.array(d['ready'])
    out=dict(id=release,ready=d['ready'],source=source.name,source_sha256=_hash(source),card=card,shot=row['shot'],
             stock_own=row['score']['learner'],stock_total=row['score']['returns'],projected_height=projected_height(final['receive_line']),
             source_touch=final['near_contact'],surface_depth=list(DEPTHS),depth_basis='first physical impact' if strict else 'separation (obsolete)',modes=[])
    with centered(home):
        a=arena();cad=arena(detail='cad');eye_engine=OpticalClinic(body=True)
        native=Rollout(a,1);reader,rig,stride,lo,hi=joint.resources(a)
        guard=Reach(cad,home,home);kin=Faces(cad)
        try:
            start_tick=round((final['near_bounce']['time']-.036)*60)
            end_tick=min(round((final['near_contact']['time']-.059)*60),start_tick+13)
            source_start=max(0,round((final['far_contact']['time']-.58)*60))
            for mode in ('optical','capacity') if capacity else ('optical',):
                a.launch(row['shot'],card=card);a.partner.reset(seed=card['seed'])
                eye=joint.Vision(rig,a.control_dt,raw=True);eye.eye=LearnedOptics(eye.eye,OPTIC)
                eye_engine.previous=None;eye_engine.bounced=False
                ref=home.copy()
                start=process_time();actions=[];notes=[];tape=None;from_tick=None
                base_key=386102
                for tick in range(len(row['actions'])+100):
                    q=a.joints.copy()
                    if tick<=end_tick and mode=='optical':
                        raw=observed(eye,tick,reader.capture() if tick%2==0 and tick>=source_start else None,q)
                        eye_engine.observe(tick,eye.eye,q,ref.copy())
                    if tape is None and start_tick<=tick<=end_tick and (tick-start_tick)%5==0:
                        old=[e for e in row['events'] if e['time']<=a.data.time+1e-8]
                        if digest_json(a.events)!=digest_json(old):
                            raise ValueError('Actual stock motor and event prefix changed')
                        if mode=='optical':
                            if raw is None or raw[4]>=-.9 or not eye_engine.sync():continue_trial=False
                            else:
                                t=eye_engine.twin
                                p=t.partner.flight.path(t.ball,t.velocity,t.spin,1,bounced=t._received_bounce,horizon=.52)
                                engine=eye_engine;change=eye_engine.rate[:5];continue_trial=True
                        else:
                            p=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=.52)
                            engine=native;t=None;change=(q-previous_q)[:5]/a.control_dt;continue_trial=True
                        if continue_trial:
                            frames=move_center(guard,kin,p,q[:5],ref[:5],change)
                            tape,metrics=solve(engine,frames,ref,a.control_dt,base_key+(tick-start_tick),optical_state=t,strict=strict)
                            notes.append(dict(tick=tick,encoder=q.tolist(),stock_reference=ref.tolist(),**metrics))
                            if tape is not None:from_tick=tick
                    if tape is None and tick<len(row['actions']):
                        u=np.asarray(row['actions'][tick]);ref=np.clip(ref+stride*u,lo,hi)
                    elif tape is None:break
                    else:
                        age=tick-from_tick
                        target=tape[age] if age<len(tape) else home
                        u=np.clip((np.clip(target,lo,hi)-ref)/stride,-.8,.8);ref=np.clip(ref+stride*u,lo,hi)
                    a.exchange(u,joint.FAR_COMMAND);actions.append(u.tolist());previous_q=q
                    if a.terminated or a.truncated:break
                    if tape is not None and a.legal_returns-1>row['score']['learner'] and a.rival_returns>row['score']['opponent']:break
                score=audit(a.events,opening_side=1);contact=[h for h in score['shots'] if h['side']==1]
                exact_prefix=from_tick is None and digest_json(row['events'])==digest_json(a.events) or from_tick is not None
                if not exact_prefix:raise ValueError('No-action branch changed archived physical point')
                out['modes'].append(dict(name=mode,notes=notes,issued_tick=from_tick,score=score,events=a.events,actions=actions,
                    placements=a.point_placements(),cpu=process_time()-start,exact_prefix=exact_prefix,
                    reached_intermediate=bool(len(contact)>row['score']['learner'] and -1.50<contact[-1]['contact_xyz'][1]<-1.30)))
                print(release,mode,from_tick,score['learner'],score['opponent'],[(r['tick'],r['poses']) for r in notes],flush=True)
                _write(output,out)
        finally:
            reader.close();eye_engine.twin.close();cad.close();a.close()


def verify(saved,output):
    from .tt37_opponent_audit import CourtClearance,contact_observer,guarded,verify_clearance
    d=_read(saved);reports=[]
    with centered(d['ready']):
        a=arena();cad=arena(detail='cad')
        try:
            for r in d['modes']:
                if r['score']['learner']<=d['stock_own']:continue
                a.launch(d['shot'],card=d['card']);a.partner.reset(seed=d['card']['seed'])
                guarded(a,CourtClearance(cad.model,articulated=True));touches=[]
                with contact_observer(a,touches):
                    for u in r['actions']:a.exchange(u,joint.FAR_COMMAND)
                if any(digest_json(actual)!=digest_json(r[label]) for label,actual in (
                    ('events',a.events),('score',audit(a.events,opening_side=1)),('placements',a.point_placements())
                )):raise ValueError('Physical replay changed')
                verify_clearance(a)
                selected=[t for t in touches if t['time_s']>=r['issued_tick']/60]
                reports.append(dict(mode=r['name'],source_prefix_tick=r['issued_tick'],native_contacts=selected,
                    clearance=a.clearance_result(),own=r['score']['learner'],opponent=r['score']['opponent']))
        finally:a.close();cad.close()
    _write(output,dict(source_sha256=_hash(saved),replays=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=int,choices=tuple(CASES));p.add_argument('out',type=Path)
    p.add_argument('--camera-only',action='store_true');p.add_argument('--separation',action='store_true')
    p.add_argument('--audit',type=Path)
    a=p.parse_args()
    if a.audit:verify(a.audit,a.out)
    else:case(a.release,a.out,capacity=not a.camera_only,strict=not a.separation)

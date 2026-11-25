"""Separately test blade rotation around the existing handle, with stock mechanics and actual opens."""
import argparse
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt40_low_feasibility as low
from . import tt51_low_reverse_curriculum as optical
from . import tt56_central_training as central
from . import tt55_midline_margin as impact
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, guarded
from .tt40_long_low import same_trace

ROOT=Path('artifacts/tt57_handle_rotation')
PLAN=ROOT/'protocol.json'
OUT=ROOT/'results.json.gz'
AUX=Path('artifacts/tt54_midline/protocol.json')
ANGLES=(0.,-45.,45.)
INVARIANTS=('geom_pos','geom_size','geom_friction','geom_contype','geom_conaffinity',
            'jnt_range','jnt_pos','jnt_axis','actuator_ctrlrange','actuator_forcerange',
            'actuator_trnid','actuator_gainprm','body_mass','body_inertia','body_ipos','body_iquat','eq_data')


def proof_model(a):
    return {key:sha256(np.asarray(getattr(a.model,key)).tobytes()).hexdigest() for key in INVARIANTS}


def rotate(a,angle):
    blade=a.model.geom('learner_racket').id
    handle=a.model.geom('learner_handle').id
    before=proof_model(a); q=a.model.geom_quat.copy();center=a.data.geom_xpos[blade].copy()
    shaft=a.data.geom_xpos[handle].copy()
    if angle:
        roll=np.deg2rad(angle)*.5;new=np.empty(4)
        mujoco.mju_mulQuat(new,np.array([np.cos(roll),0.,0.,np.sin(roll)]),q[blade])
        a.model.geom_quat[blade]=new
        mujoco.mj_forward(a.model,a.data)
    if (before!=proof_model(a) or np.linalg.norm(center-a.data.geom_xpos[blade])>1e-12
            or np.linalg.norm(shaft-a.data.geom_xpos[handle])>1e-12
            or any(not np.array_equal(q[i],a.model.geom_quat[i]) for i in range(a.model.ngeom) if i!=blade)):
        raise ValueError('Blade-only motion unexpectedly changed the body or handle')
    rel=a.model.geom_pos[blade]-a.model.geom_pos[handle]
    if np.linalg.norm(rel[:2])>1e-12:
        raise ValueError('Blade center not on the declared existing handle axis')
    return dict(angle=angle,unchanged_arrays=before,quaternion_before=q[blade].tolist(),
                quaternion_after=a.model.geom_quat[blade].tolist(),mass_kg=float(a.model.body('learner_paddle').mass[0]),
                geom_unchanged_center_local=a.model.geom_pos[blade].tolist())


def make_plan():
    archival=next(x for x in _read(AUX)['rows'] if x['id']==11846)
    earlier=_read(Path(archival['source'])); r=earlier['rows'][archival['index']]; c=earlier['cards'][archival['index']]
    t0=next(v for v in _read(central.ROOT/'active-protocol.json')['rows'] if v['id']==14000)
    changed_trial=next(v for v in _read(central.ROOT/'active-training.json.gz')['rows'] if v['id']==14000)
    hit=next(v for v in changed_trial['branches'] if v['scout']['seed']==562374737)
    if hit['stock']['answered'] or r['placements']['delivered'][0]['near_landed']:
        raise ValueError('Expected two established low training misses')
    known=[dict(id=11846,source=archival['source'],source_sha256=archival['source_sha256'],
                index=archival['index'],card=c,seed=c['seed'],offer=r['placements']['delivered'][0],
                stock_events_sha256=digest_json(r['events']),stock_commands=r['actions']),
           dict(id=14000,source=t0['case']['source'],source_sha256=t0['case']['source_sha256'],index=t0['case']['index'],
                card=t0['card'],seed=hit['scout']['seed'],offer=hit['stock']['offer'],
                stock_events_sha256=digest_json(hit['stock']['events']),
                stock_commands=_read(Path(t0['case']['source']))['rows'][t0['case']['index']]['actions'][:hit['scout']['preparing_gate']]+hit['stock']['controls'])]
    reserved=set(_read(central.bank.ROOT/'central-sources.json')['reserved'])
    if any(v['id'] in reserved or v['offer']['leg']!=0 for v in known):
        raise ValueError('The bench test cannot use a qualification release or earlier learner hit')
    optical.save(PLAN,dict(angles=ANGLES,actual_balls=known,model_batch=96,model_rounds=2,
        role='Separate nonqualifying mechanical presentation test on two developmental openings. Physical handle and shaft seat, captured rigid-body mass properties, center, dimensions, stock joints and motors never change. Only blade geom quaternion is offset about its existing local Z axis, at the start of each benchmark copy. Same physically matched real incoming contacts/bounce/line, unchanged TT37 target and far paddle. True state used only by a finite mechanism coach, not a claimed camera policy.'))


def setup(a,c,old,r):
    a.launch(old['shot'],card=c);a.partner.reset(seed=c['seed']);a.card=dict(c,seed=r['seed'])


def shape_relation(a,stock,angle,ready):
    q0=np.array(ready[:5]); old=low.Faces(stock);new=low.Faces(a)
    sample=[]
    for roll in (q0[4], -2.30,-1.22,-.70,.10,1.28,2.50):
        q=q0.copy();q[4]=roll
        q2=q.copy();r=roll+np.deg2rad(angle)
        equivalent=[r+k*np.pi for k in (-2,-1,0,1,2) if old.low[4] <= r+k*np.pi <= old.high[4]]
        if not equivalent:raise ValueError('Failed to find the geometrically equivalent signed wrist roll')
        q2[4]=min(equivalent,key=lambda m:abs(m-roll))
        x,n=new.pose(q);y,n2=old.pose(q2)
        sample.append(dict(optical_face_match=abs(float(n@n2)),offset_center_m=float(np.linalg.norm(x-y)),
                           stock_wrist_roll=float(q2[4]),new_wrist_roll=float(q[4])))
    return sample


def read_geometry(a,coach,base):
    route=a.partner.flight.path(a.ball,a.velocity,a.spin,1,bounced=a._received_bounce,horizon=1.10)
    out=[]
    for b in base:
        row=route[np.argmin(abs(route[:,0]-b['time']))]
        center,n=coach.ik.pose(b['q']);n=b['face']*n
        offset=row[1:4]-center;along=float(n@offset)
        r=float(np.linalg.norm(offset-along*n))
        out.append(dict(face=b['face'],depth=b['depth'],radial_m=r,axial_m=along,
                 axis=n.tolist(),legal_plane_geometry=.010<along<.046 and r<.058,
                 q=b['q'].tolist()))
    return out


def compare_offer(a,r):
    p=a.point_placements()
    if not p['requested'] or p['requested'][0]['target']!=list(draw(a.card,0,'random',None)[0]):
        raise ValueError('Changed ordinary independent target in a grip diagnostic')
    current=p['delivered'][0] if p['delivered'] else None
    same=bool(current and all(same_trace(current[k],r['offer'][k]) for k in ('far_contact','near_bounce'))
              and (not current['receive_line'] or same_trace(current['receive_line'],r['offer']['receive_line'])))
    return p,current,same


def trace_result(a,r,notes):
    p,current,match=compare_offer(a,r)
    landed=bool(current and current['near_landed']); answered=landed and len(p['delivered'])>1 and bool(p['delivered'][1]['near_bounce'])
    hit=None if not current or not current['near_contact'] else current['near_contact']['time']
    obs=next((k for k in notes if k['side']==1 and hit is not None and abs(k['time_s']-hit)<1e-7),None)
    dep=next((x for x in a.events if hit is not None and x.get('collider')=='learner_racket' and hit<x['time']<hit+.1),None)
    return dict(same=match,landed=landed,answered=answered,contact=obs,depart=None if dep is None else dep['velocity'],
                events=a.events,placements=p)


def matched_motion(a,c,old,r,tape,clear):
    setup(a,c,old,r); guarded(a,clear);hits=[];commands=[]
    try:
        with impact.physical_contacts(a,hits):
            for u in tape:
                if max(abs(np.array(u)))>.8+1e-10:raise ValueError('Changed stock signed rate')
                commands.append(u)
                _,_,end,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                if end or cap or a.rival_returns>=2:break
        response=trace_result(a,r,hits); response.update(commands=commands,clearance=a.clearance_result())
        if response['clearance']['samples']!=round(a.data.time/a.model.opt.timestep):
            raise ValueError('The full rotation replay omitted a CAD state')
        return response
    finally:a.sample_clearance(False)


def grade(a,c,old,r,seed,angle,clear,conf):
    # On a newly successful rotated arm the original failed tape may end
    # before its ball lands. Hold the final reference long enough to measure
    # whether that same programmed command prefix actually earns a reply.
    open_control=matched_motion(a,c,old,r,r['stock_commands']+([[0.]*6]*110 if angle else []),clear)
    if angle==0 and digest_json(open_control['events'])!=r['stock_events_sha256']:
        raise ValueError('Original stock score or full trace failed replay')
    if not open_control['same']:
        raise ValueError('Installed blade changed the actual incoming ball itself')
    far=next(e for e in open_control['events'] if e['kind']=='separation' and e.get('collider')=='rival_racket')
    tick=int(np.ceil(far['time']/mpc.DT+1e-7))+1
    setup(a,c,old,r)
    for u in r['stock_commands'][:tick]:a.exchange(u,mpc.FAR_COMMAND)
    first=[e for e in open_control['events'] if e['time']<=a.data.time+1e-8]
    if digest_json(first)!=digest_json(a.events):raise ValueError('Repeated physical far contact diverged before test')
    coach=low.Probe(a);basis=coach.basis();geom=read_geometry(a,coach,basis) if basis else []
    coach.basis=lambda:basis
    try:
        a.clearance_margin=.001
        choices,receipt=coach.search(seed,conf['model_batch'],conf['model_rounds'])
    finally:a.clearance_margin=0.
    choices.sort(key=lambda o:(o['feedback'][8]>.061,o['cost']))
    tested=[]
    for g in choices[:2]:
        tape=r['stock_commands'][:tick]+g['suffix']+[[0.]*6]*150
        got=matched_motion(a,c,old,r,tape,clear)
        got['model_choice']={k:v for k,v in g.items() if k!='suffix'}
        tested.append(got)
        if got['answered'] and got['same'] and got['clearance']['penetration_samples']==0:
            # A second actual, independent solver reset with identical tape.
            verified=matched_motion(a,c,old,r,got['commands'],clear)
            if digest_json(verified['events'])!=digest_json(got['events']) or verified['clearance']['penetration_samples']:
                raise ValueError('Altered grip actual scored witness did not reproduce cleanly')
            got['fully_verified']=True
            break
    return dict(exact_control=open_control,fork_tick=tick,geometry=geom,native=receipt,physical=tested)


def run(output=OUT):
    plan=_read(PLAN);results=_read(output) if output.exists() else dict(protocol_sha256=_hash(PLAN),rows=[],complete=False)
    if results['protocol_sha256']!=_hash(PLAN):raise ValueError('Changed rotation bench experiment')
    for r,angle in [(r,angle) for r in plan['actual_balls'] for angle in plan['angles']][len(results['rows']):]:
        p=Path(r['source']);d=_read(p);c,old=r['card'],d['rows'][r['index']]
        if _hash(p)!=r['source_sha256']:raise ValueError('Changed true received source')
        with mpc.centered(d['ready']):
            a=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail='cad');stock=mpc.arena(width=64,height=48)
            try:
                key=rotate(a,angle);shadow=rotate(cad,angle)
                if key['mass_kg']!=shadow['mass_kg'] or key['unchanged_arrays']!=proof_model(stock):
                    raise ValueError('Even a nonqualifying blade test may not change stock inertial or control properties')
                norm=shape_relation(a,stock,angle,d['ready'])
                check=CourtClearance(cad.model,articulated=True)
                result=grade(a,c,old,r,[571144,c['seed'],r['seed']],angle,check,plan)
                results['rows'].append(dict(id=r['id'],rotation=key,cad_rotation=shadow,wrist_equivalence=norm,result=result))
                optical.save(output,results)
                print(r['id'],angle,'shape',len(result['geometry']),sum(z['legal_plane_geometry'] for z in result['geometry']),
                      'model',[(q['contacts'],q['legal']) for q in result['native']['rounds']],
                      'actual',[(q['answered'],q['same'],q['clearance']['penetration_samples']) for q in result['physical']],flush=True)
            finally:a.close();cad.close();stock.close()
    results['complete']=True;optical.save(output,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    make_plan() if a.phase=='plan' else run(a.out)

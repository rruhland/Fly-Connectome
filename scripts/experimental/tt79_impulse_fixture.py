"""Measure attainable low face-patch speed after isolated real grip/seating changes."""

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from . import tt63_fixtures as fixture
from . import tt72_retarget_trajectory as tr
from . import tt77_centered_impulse as imp
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_bodycoach_windows import Reach


ROOT=Path('artifacts/tt79_impulse_fixture')
PLAN=ROOT/'protocol.json'
MAP=ROOT/'kinematics.json'
OUT=ROOT/'physical.json.gz'
MP=tr.mpc
CONFIG={'stock':{},'in3':dict(seat=.003),'out8':dict(seat=-.008),'out16':dict(seat=-.016),
        'out24':dict(seat=-.024),'riser8':dict(h=.008),'riser18':dict(h=.018),'riser30':dict(h=.030)}
RISER='tt29_clamp_tt79_riser'
CLOCKS=(-4,-2,0,2,4)
RADIUS=.058
JMAX=4.16
ROUNDS=3
N=96


def freeze():
    src=(tr.DONORS,imp.TRAIN,fixture.PLAN,fixture.grip.PREPARED,fixture.RESERVED)
    ROOT.mkdir(parents=True,exist_ok=True)
    d=dict(schema='tt79-contact-patch-authority-1',options=CONFIG,train=[9254,8831,15456],clocks=CLOCKS,
           source={str(p):_hash(p) for p in src},ready=_read(fixture.PLAN)['ready'],budget=dict(candidates=N,rounds=ROUNDS,physical=3),
           hold=_read(fixture.PLAN)['held'],rule='Same paddle, vertical-axis insertion only with an original jaw-width fit, or same arm on a bolted flat plate with no orientation or horizontal shift. Screen ordinary sites and true donor flight together. Positive face patch limits use the same 4.16 rad/s signed reference and reject motion into a hard stop, but ignore inertia and changing moments. Test actual candidates with the native rate envelope and full CAD. Keep all regulation conditions and original random responding opponent. Exact measured geometry is training-only.')
    if PLAN.exists() and digest_json(_read(PLAN))!=digest_json(d):
        raise ValueError('The frozen physical practice roster changed')
    _write(PLAN,d)


def plan():
    d=_read(PLAN)
    if d['options']!=CONFIG or any(_hash(Path(f))!=h for f,h in d['source'].items()):
        raise ValueError('Changed physical insertion/riser roster')
    return d


@contextmanager
def mounted(name):
    values=CONFIG[name]
    if not values:
        yield
        return
    if 'seat' in values:
        with patch.dict(fixture.grip.POSES,{name:(values['seat'],0.,0.)}),fixture.grip.mounted(name):
            yield
        return
    height=values['h']; old=fixture.TT20Arena._model_document
    def document(a):
        root=old(a)
        if a.fixture!=fixture.TABLE_EDGE:
            raise ValueError('Flat riser requires the standard near end plate')
        world=root.find('worldbody'); b=world.find("body[@name='learner_base']")
        support=world.find("body[@name='tt20_table_end_clamp']")
        plate=support.find("geom[@name='tt29_clamp_top_plate']")
        center=np.fromstring(plate.get('pos'),sep=' '); size=np.fromstring(plate.get('size'),sep=' ')
        top=center[2]+size[2]; new=np.fromstring(b.get('pos'),sep=' '); new[2]+=height
        b.set('pos',fixture._vector(new))
        # The two original clamp tabs project 1 mm past the plate. Keep a
        # 2 mm relieved undercut and bear on the central and inboard pads.
        relief=.002
        fixture._box(support,RISER,(*center[:2],top+.5*(height+relief)),(.077,.049,.5*(height-relief)),(.22,.25,.29))
        fixture._box(support,RISER+'_center_pad',(0.,center[1]-.018,top+.5*relief),(.049,.029,.5*relief),(.22,.25,.29))
        fixture._box(support,RISER+'_inboard_pad',(0.,center[1]+.031,top+.5*relief),(.073,.018,.5*relief),(.22,.25,.29))
        return root
    with patch.object(fixture.TT20Arena,'_model_document',document):
        yield


def fixed_arrays(m):
    g=[i for i in range(m.ngeom) if not m.geom(i).name.startswith(RISER)]
    b=[i for i in range(m.nbody) if m.body(i).name!='tt20_table_end_clamp']
    e=[i for i in range(m.neq) if i!=m.equality('tt19_grasp').id]
    return ({n:np.copy(getattr(m,n)) for n in ('jnt_range','jnt_axis','actuator_ctrlrange','actuator_forcerange',
                     'actuator_gainprm','dof_damping','cam_pos','cam_quat')} |
            {n:np.copy(getattr(m,n)[b]) for n in ('body_mass','body_inertia','body_ipos','body_iquat')} |
            {n:np.copy(getattr(m,n)[g]) for n in ('geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity')} |
            dict(jnt=m.eq_data[e].copy()))


def support(a, name):
    if name=='stock':
        return fixture.support_check(a,'stock')
    if 'seat' in CONFIG[name]:
        hit=fixture.grip.hold_check(a,name)
        # Measured stock-equivalent baseline, no nominal-overlap concession.
        hit['safe']=bool(hit['bilateral_span_m']>=.005-1e-8 and hit['least_blade_to_wrist_m']>=.001)
        return hit
    m,d=a.model,a.data; base=m.body('learner_base').id; geoms=np.flatnonzero(m.geom_bodyid==base)
    deck=m.geom(RISER); z=d.geom_xpos[deck.id,2]+deck.size[2]
    gap=min(float(mujoco.mj_geomDistance(m,d,i,deck.id,.1,None)) for i in geoms)
    bearing=m.geom('tt29_clamp_top_plate').id
    pads=[m.geom(RISER+s).id for s in ('_center_pad','_inboard_pad')]
    contact=[float(mujoco.mj_geomDistance(m,d,i,bearing,.1,None)) for i in pads]
    seat=[float(mujoco.mj_geomDistance(m,d,deck.id,i,.1,None)) for i in pads]
    jaw=[m.geom(f'tt29_clamp_{j}_upper_jaw').id for j in range(2)]
    gap_jaw=min(float(mujoco.mj_geomDistance(m,d,i,k,.1,None)) for i in [deck.id,*pads] for k in jaw)
    table=min(float(mujoco.mj_geomDistance(m,d,i,m.geom(n).id,.1,None)) for i in [deck.id,*pads] for n in a.table_panels)
    # The stock base CAD lower face is supported at the very same position
    # on the copied plate footprint. Compare full mesh vertices on that face.
    h=m.mesh('base_so101_v2').id; i=next(i for i in geoms if m.geom_dataid[i]==h)
    verts=m.mesh_vert[m.mesh_vertadr[h]:m.mesh_vertadr[h]+m.mesh_vertnum[h]]
    points=verts@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
    foot=points[np.abs(points[:,2]-z)<.00015,:2]
    rim=float(np.min(deck.size[:2]-abs(foot-d.geom_xpos[deck.id,:2]))) if len(foot) else -1.
    return dict(safe=bool(abs(gap)<.00015 and max(map(abs,contact+seat))<.00015 and table>.001
                          and gap_jaw>.00075 and rim>.001),
                base_to_riser_m=gap,pads_to_plate_m=contact,riser_to_pads_m=seat,
                riser_to_table_m=table,riser_to_upper_jaw_m=gap_jaw,footprint_margin_m=rim)


def twist_patch(ik,q,point,n,want,rotation_rate):
    """Optimistic box-constrained patch motion, not an actuator torque proof."""
    center,frame,J,W=ik.frame(q,True)
    body_n=frame[:,2]
    if body_n@n<0:
        body_n=-body_n
    offset=point-center
    local=J+np.cross(W.T,offset).T
    spin=np.cross(W.T,body_n).T
    target=np.r_[(want*np.array([.6,1.,1.2])),.040*rotation_rate]
    A=np.vstack((local*np.array([.6,1.,1.2])[:,None],.040*spin))
    low=np.full(5,-JMAX); hi=-low
    low[q<=ik.low+.006]=0.; hi[q>=ik.high-.006]=0.
    dq=lsq_linear(A,target,bounds=(low,hi),tol=1e-6).x
    achieved=local@dq
    return (dq,achieved,float(np.linalg.norm((want-achieved)*[.6,1.,1.2])),
            float(np.linalg.norm(spin@dq-rotation_rate)),float(np.abs(local[2])@np.ones(5)*JMAX))


def hard(a,donor,ready,seed):
    ik=tr.FullPose(a); route=Reach(a,ready,ready); lesson=tr.Course(donor); z=donor['impact']
    nominal_R=np.array(z['R']); template_n=nominal_R[:,2]; point=np.asarray(z['location']); center=np.asarray(z['p']);
    ball=np.asarray(z['ball']); speed=np.asarray(z['twist']); contact=point-center
    want=speed[3:]+np.cross(speed[:3],contact)
    nd=np.cross(speed[:3],template_n)
    seeds=qmc.scale(qmc.Sobol(5,scramble=True,seed=seed).random_base2(3),ik.low,ik.high)
    seeds=np.vstack([ready[:5],z['q'][:5],seeds]); rows=[]; nearest=None
    for clock in CLOCKS:
        dt=clock*MP.DT; t=z['time']+dt; p,R,T=(v[0] for v in lesson.target(np.array([t]),dt,1.))
        delta=p-center; eta=t-donor['fork']*MP.DT
        if eta<.07:continue
        for phase in (-90.,0.,90.,180.):
            shifted=R@Rotation.from_euler('z',phase,degrees=True).as_matrix()
            for start in seeds:
                q,_=ik.fit_frame(p,shifted,start,tangent=.012,stay=.00035,n=29)
                b,f,_,_=ik.frame(q,True); n=f[:,2]
                dist=ball+delta-b; axial=float(n@dist); r=float(np.linalg.norm(dist-axial*n)); dot=float(n@template_n)
                grade=abs(abs(axial)-.026)+r+.12*(1-abs(dot))
                summary=dict(clock=clock,phase_deg=phase,face_error_deg=float(np.rad2deg(np.arccos(np.clip(abs(dot),0,1)))),
                             radius_m=r,disk_plane_m=abs(axial),contact_reference=q.tolist(),proxy=grade)
                if nearest is None or grade<nearest['proxy']:nearest=summary
                if not (.009<abs(axial)<.046 and r<=RADIUS and abs(dot)>=.952 and np.max(abs(ready[:5]-q))<=JMAX*eta+.087):
                    continue
                if any(np.linalg.norm(q-v['contact_reference'])<.10 and v['clock']==clock for v in rows):continue
                if not route.route(ready[:5],q,steps=8):continue
                dq,motion,err,drift,max_z=twist_patch(ik,q,point+delta,template_n,want,nd)
                rows.append(dict(summary,desired_yz=want[1:].tolist(),achieved=motion.tolist(),joint_velocity=dq.tolist(),
                                 patch_error_mps=err,normal_rate_error_rad_s=drift,optimistic_z_mps=max_z,
                                 eta=eta,depth=float(b[1]),face_sign=1 if dot>=0 else -1))
    return dict(id=donor['case']['id'],nearest=nearest,faces=sorted(rows,key=lambda b:(b['patch_error_mps'],b['radius_m']))[:18])


def geometry():
    p=plan(); ready=np.array(p['ready']); donors=_read(tr.DONORS)['rows']; old=_read(fixture.grip.PREPARED)['rows']; base=None
    rec=dict(protocol=_hash(PLAN),operator=_hash(Path(__file__)),rows=[],complete=False)
    for name in CONFIG:
        with mounted(name),MP.centered(ready):
            a=MP.arena(width=64,height=48,detail='cad')
            try:
                proof=fixed_arrays(a.model)
                if base is None:
                    base=proof; goal=np.r_[*fixture.Faces(a).pose(ready[:5])]
                if any(not np.array_equal(v,proof[k]) for k,v in base.items()):
                    raise ValueError('Unrelated moving link mass, inertia, actuator or court changed')
                supported=support(a,name)
                item=dict(name=name,support=supported,mass_kg=float(a.model.body('learner_paddle').mass[0]),
                          inertia=a.model.body('learner_paddle').inertia.tolist(),
                          ready=None,ordinary=None,hard=[])
                if supported['safe']:
                    commission=fixture.calibrate(a,ready,goal) if name!='stock' else None
                    if name=='stock' or commission is not None:
                        rest=np.asarray(commission['ready']) if commission else ready
                        q=fixture.grip.scan_arena(a,rest,old)
                        item.update(ready=rest.tolist(),ordinary=q.get('ordinary_covered'),commission=commission,
                                    hard=[hard(a,d,rest,9254+i) for i,d in enumerate(donors)])
                rec['rows'].append(item);_write(MAP,rec)
                print(name,supported['safe'],item['ordinary'],[(q['id'],len(q['faces']),
                      None if not q['faces'] else round(q['faces'][0]['patch_error_mps'],3)) for q in item['hard']],flush=True)
            finally:a.close()
    rec['complete']=True;_write(MAP,rec)


def choices(a,d,map_entry,stock_replays):
    kin=tr.FullPose(a); ready=a.joints[:5].copy(); axes=a._arm_ids[1][2]; jaw=a.data.ctrl[axes[5]]
    raw=next(r for r in map_entry['hard'] if r['id']==d['case']['id'])['faces']
    options=[]
    for f in raw[:12]:
        q=f['contact_reference']; v=f['joint_velocity']; eta=f['eta']; at=eta/MP.DT
        # Two preparation accelerations and an attained patch velocity. The
        # native rollout integrates joints/racket, not the optimistic bound.
        from .tt38_motorcoach import schedules,SIZES
        for g in (.82,1.12):
            goal=schedules(np.asarray([q]),g*np.asarray([v]),np.asarray([eta]),np.zeros((1,len(SIZES))),MP.DT,120)[0]
            goal[:,5]=jaw;options.append(dict(ref=goal,at=at))
    # Balance with the exact measured stock impulse-learning motions. Ask
    # whether simply using the better installation changes their response.
    if d['case']['id']==9254:
        for s in stock_replays:
            options.append(dict(ref=imp.refs(a,np.array(s['commands']),d['fork'],120),
                 at=s['contact']['time_s']/MP.DT-d['fork']))
    return options[:24]


def physical():
    p=plan(); disk=_read(MAP); donors=_read(tr.DONORS)['rows']; states={i['name']:i for i in disk['rows']}
    if not disk['complete'] or disk['protocol']!=_hash(PLAN) or disk['operator']!=_hash(Path(__file__)):
        raise ValueError('Complete frozen kinematics map before opening physical trials')
    stock=states['stock']; limit=stock['ordinary']*.85
    q=lambda m,i:next((min((f['patch_error_mps'] for f in z['faces']),default=2.) for z in m['hard'] if z['id']==i),2.)
    selected=['stock']
    for group in ('out','riser'):
        candidates=[v for v in disk['rows'] if v['name'].startswith(group) and v['ordinary'] is not None and v['ordinary']>=limit]
        if not candidates:continue
        pick=min(candidates,key=lambda r:(sum(q(r,i) for i in p['train']),-r['ordinary']))
        selected.append(pick['name'])
    old=[s for r in _read(imp.TRAIN)['rounds'] for s in r['physical'] if s['contact'] and s['depart'] and not s['clearance']['penetration_samples']]
    old.sort(key=lambda x:(x['contact']['radial_m']>.058,-x['depart'][1]-.20*x['depart'][2]))
    docket=[(d,key) for d in donors for key in selected]
    receipt=dict(protocol=_hash(PLAN),kinematics=_hash(MAP),variants=selected,rows=[],complete=False)
    if OUT.exists():
        before=_read(OUT)
        if any(before[k]!=v for k,v in receipt.items() if k not in ('rows','complete')):raise ValueError('Changed physical fixture development')
        receipt=before
    with fixture.school.enabled(cameras=False):
        for d,key in docket[len(receipt['rows']):]:
            spec=d['case']; c,row,anchor=tr.qual.get(_read(Path(spec['source'])),spec)
            ready=states[key]['ready']; prefix=[tr.ZERO]*d['fork']; wins=[]; notes=[]
            with mounted(key),MP.centered(ready):
                a=MP.arena(width=64,height=48); cad=MP.arena(width=64,height=48,detail='cad')
                try:
                    guard=tr.qual.GripClearance(cad.model,stock='seat' not in CONFIG[key]); nominal=Rollout(a,1)
                    tr.tape.setup(a,c,row,anchor)
                    for u in prefix:a.exchange(u,MP.FAR_COMMAND)
                    if digest_json(a.events)!=digest_json([e for e in row['events'] if e['time']<=a.data.time+1e-8]):
                        raise ValueError('Altered fixture changed pre-swing physical incoming')
                    bases=choices(a,d,states[key],old[:6]); tested=[]
                    if bases:
                        # Deterministic shared-budget corrections to acquired
                        # schedules, including near misses as signed feedback.
                        rng=np.random.default_rng([790312,c['seed']]); labels=np.arange(N)%len(bases)
                        pars=rng.uniform(-.65,.65,(N,len(imp.MAG)));pars[:len(bases)]=0.
                        lucky=[]
                        for k in range(ROUNDS):
                            goals=imp.schedules(bases,labels,pars);fb=np.zeros((N,9));
                            u,v=nominal.trajectories(goals,tr.qual.AIM,1.0,feedback=fb)
                            loss,contact,errors,legal,interior,earned=imp.feedback(v,fb)
                            mask=np.argsort(loss);notes.append(dict(legal=int(legal.sum()),touch=int(contact.sum()),interior=int(interior.sum()),soft=int(earned.sum())))
                            for j in mask[:9]:
                                if contact[j]:lucky.append((float(loss[j]),u[j,:int(np.ceil(bases[labels[j]]['at']))+22].tolist(),dict(legal=bool(legal[j]),radius=float(fb[j,8]),departure=fb[j,2:5].tolist(),round=k)))
                            if k+1<ROUNDS:
                                pars,labels=imp.proposal(pars,labels,mask,errors,contact,rng,k)
                        tried=set()
                        for _,u,model in sorted(lucky,key=lambda h:h[0]):
                            digest=digest_json(u)
                            if digest in tried:continue
                            tried.add(digest)
                            real=tr.tape.matched_motion(a,c,row,anchor,prefix+u+[tr.ZERO]*145,guard)
                            real['native_model']=model;real['soft_answer']=imp.actual_soft(real);tested.append(real)
                            if real['soft_answer']:
                                again=tr.tape.matched_motion(a,c,row,anchor,real['commands'],guard)
                                if not imp.actual_soft(again) or digest_json(real['events'])!=digest_json(again['events']):
                                    raise ValueError('Actual soft answer changed on native repeat')
                                real['repeat']=digest_json(again['events']);break
                            if len(tested)>=p['budget']['physical']:break
                finally:a.close();cad.close()
            receipt['rows'].append(dict(case=spec,name=key,ready=ready,bases=len(bases),native=notes,physical=tested))
            receipt['complete']=len(receipt['rows'])==len(docket);_write(OUT,receipt)
            print(spec['id'],key,notes,[(v['landed'],v['answered'],v['soft_answer'],v['clearance']['penetration_samples']) for v in tested],flush=True)


if __name__=='__main__':
    load=argparse.ArgumentParser(description=__doc__);load.add_argument('mode',choices=('plan','geometry','physical'))
    {'plan':freeze,'geometry':geometry,'physical':physical}[load.parse_args().mode]()

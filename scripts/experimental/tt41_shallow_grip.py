"""Isolated handle-seating falsification. The default benchmark is unchanged."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import tt19_so101 as so
from .tt4_transfer_trial import _hash, _read, _write
from .tt31_ready import centered
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, guarded, verify_clearance
from .tt38_joint_mpc import BANK, PLAN, MPC, DeepField, play, resources
from .tt41_shallow_radius import endpoint
from .tt41_shallow_model import Chart
from .tt40_low_feasibility import Faces
from .tt29_rally_physics import measured_launches


ROOT=Path('artifacts')
PREPARED=ROOT/'tt41/shallow-middle-radius.json'
OPTS=ROOT/'tt37/rebound.npz'
FEEDS=Path('/tmp/tt25-rallies.json')
LEGEND=('shift_along_handle_m','tilt_deg_about_jaw_pressure_axis','roll_deg_about_handle')
POSES={
    'stock':(0.,0.,0.),
    'out-8':(-.008,0.,0.),
    'out-16':(-.016,0.,0.),
    'in-8':(.008,0.,0.),
    'out-12-tilt-minus':(-.012,-7.,0.),
    'out-12-tilt-plus':(-.012,7.,0.),
    'out-12-roll-minus':(-.012,0.,-15.),
    'out-12-roll-plus':(-.012,0.,15.),
}
GRIP_DEPTH=-.0981274
HEIGHTS=(1.06,1.14)
XS=(-.26,-.13,0.,.13,.26)
NORMAL_DEPTHS=(-1.42,-1.50,-1.58)
JAW=('learner_gripper','learner_moving_jaw_so101_v1')
GEOMETRY=dict(radius=.0085,shaft=(-.043,.022),required_pinching_span_m=.006,
              max_jaw_gap_m=.002,minimum_blade_to_finger_m=.001,normal_centered_radius_m=.060)


class GripRejected(ValueError):
    """Configuration fails the geometry-only preflight."""


def weld_for(config):
    shift,tilt,roll=POSES[config]
    rotation=Rotation.from_rotvec([np.deg2rad(tilt),0.,0.])*Rotation.from_rotvec([0.,0.,np.deg2rad(roll)])
    relative=np.asarray(so._STRAIGHT_GRASP[:3])
    pivot=np.array([*relative[:2],GRIP_DEPTH])
    center=pivot+rotation.apply(relative-pivot+[0,0,shift])
    return tuple(np.r_[center,rotation.as_quat()[[3,0,1,2]]])


@contextmanager
def mounted(config):
    """Apply an opt-in weld pose at construction, incl. existing calibrated private twins."""
    if config=='stock':
        yield
        return
    weld=weld_for(config)
    document,reset=so.SO101Arena._model_document,so.SO101Arena.reset

    def model(self):
        if self.paddle_grip!='straight':
            raise GripRejected('Auxiliary grip may only replace the standard straight grip')
        self._grasp=weld
        return document(self)

    def placed(self,*args,**kwargs):
        render=kwargs.pop('render',True)
        reset(self,*args,render=False,**kwargs)
        g=self.model.body(so.GRIP_BODY).id;f=self.model.joint('learner_paddle_free').qposadr[0]
        mujoco.mju_mulQuat(self.data.qpos[f+3:f+7],self.data.xquat[g],np.asarray(weld[3:]))
        mujoco.mj_forward(self.model,self.data)
        return self.render_cameras() if render else None

    with patch.object(so.SO101Arena,'_model_document',model),patch.object(so.SO101Arena,'reset',placed):
        yield


def mesh_triangles(m,d,i,origin,basis):
    data=m.geom_dataid[i]
    if m.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH or data<0:
        raise GripRejected('Jaw mesh is not present in the full model')
    f=m.mesh_face[m.mesh_faceadr[data]:m.mesh_faceadr[data]+m.mesh_facenum[data]]
    v=m.mesh_vert[m.mesh_vertadr[data]:m.mesh_vertadr[data]+m.mesh_vertnum[data]]
    pose=d.geom_xmat[i].reshape(3,3);v=(v@pose.T+d.geom_xpos[i]-origin)@basis
    return v[f]


def cross_section_x(tri,y,z):
    """Exact triangle crossings for a line at fixed y,z in the grip frame."""
    p=tri[:,0];u=tri[:,1]-p;v=tri[:,2]-p
    den=u[:,1]*v[:,2]-u[:,2]*v[:,1]
    nz=np.abs(den)>1e-12
    p,u,v,den=p[nz],u[nz],v[nz],den[nz]
    dy=y-p[:,1];dz=z-p[:,2]
    a=(dy*v[:,2]-dz*v[:,1])/den;b=(u[:,1]*dz-u[:,2]*dy)/den
    hit=(a>=-1e-8)&(b>=-1e-8)&(a+b<=1+1e-8)
    return np.sort((p[:,0]+a*u[:,0]+b*v[:,0])[hit])


def hold_check(a,config):
    m,d=a.model,a.data
    blade=m.geom('learner_racket').id;handle=m.geom('learner_handle').id
    g=m.body(so.GRIP_BODY).id;origin=d.xpos[g];basis=d.xmat[g].reshape(3,3)
    weld=np.array(weld_for(config));axis=Rotation.from_quat(weld[[4,5,6,3]]).apply([0.,0.,1.])
    mates={m.mesh(m.geom_dataid[k]).name:k for k in range(m.ngeom)
           if m.body(m.geom_bodyid[k]).name in JAW and m.geom_type[k]==mujoco.mjtGeom.mjGEOM_MESH}
    ids=[mates[s] for s in ('wrist_roll_follower_so101_v1','moving_jaw_so101_v1')]
    fixed,moving=(mesh_triangles(m,d,i,origin,basis) for i in ids)
    intervals=[]
    for z in np.arange(-.111,-.069,.0005):
        t=(z-weld[2])/axis[2]
        if not GEOMETRY['shaft'][0]+.004<t<GEOMETRY['shaft'][1]-.004:continue
        x,y,_=weld[:3]+t*axis
        left=cross_section_x(fixed,y,z);right=cross_section_x(moving,y,z)
        left=left[left<x];right=right[right>x]
        if not len(right) or not len(left):continue
        gap=[float(x-GEOMETRY['radius']-left[-1]),float(right[0]-x-GEOMETRY['radius'])]
        intervals.append(dict(z=float(z),gap=gap))
    okay=[r['z'] for r in intervals if max(r['gap'])<GEOMETRY['max_jaw_gap_m'] and min(r['gap'])>-.002]
    pieces=np.split(okay,np.flatnonzero(np.diff(okay)>.00051)+1) if okay else []
    span=max((float(p[-1]-p[0]+.0005) for p in pieces if len(p)),default=0.)
    wrist=wrist_meshes(m)
    blade_room=min((mujoco.mj_geomDistance(m,d,i,blade,.08,None),i) for i in wrist)
    room,i=blade_room
    held=d.geom_xpos[handle].copy()
    with np.errstate(all='raise'):
        return dict(weld=weld.tolist(),bilateral_span_m=span,sections=intervals,
                    least_blade_to_wrist_m=float(room),nearest=m.geom(i).name or m.body(m.geom_bodyid[i]).name,
                    handle_center=held.tolist(),approved=(config=='stock' or (
                        span>=GEOMETRY['required_pinching_span_m'] and room>=GEOMETRY['minimum_blade_to_finger_m'])))


def plan(out):
    source=_read(PLAN);prior=_read(BANK.with_suffix('.json'))
    cards=source['held_openings'][12:18]
    if {c['id'] for c in cards}&{r['card']['id'] for r in prior['practice']}:
        raise GripRejected('Live cards collide with instruction archive')
    _write(out,dict(poses={k:dict(zip(LEGEND,v,strict=True)) for k,v in POSES.items()},welds={k:weld_for(k) for k in POSES},
          geometry=GEOMETRY,normal_atlas=dict(x=XS,depth=NORMAL_DEPTHS,z=HEIGHTS),
          source=dict(opponent='TT37 continuous random reach, stock card and reply seeds',
                      plan_sha256=_hash(PLAN),bank_sha256=_hash(BANK),optics_sha256=_hash(OPTS)),
          ready=prior['ready'],cards=cards,cap=24,
          selection='Never change the declared candidates. Only test a new grip in a live game if it has bilateral jaw and blade clearance, improves two low-ray best interior radii by ≥4mm, and retains ≥85% of stock regular-height coverage. Ties by geometry only.'))


def check_plan(p):
    d=_read(p)
    if d['welds']!={k:list(weld_for(k)) for k in POSES} or d['geometry']!=_readable(GEOMETRY):
        raise GripRejected('Frozen candidate geometry changed')
    return d


def _readable(x):
    if isinstance(x,dict):return {k:_readable(v) for k,v in x.items()}
    if isinstance(x,tuple):return list(x)
    return x


def holds(procedure,out):
    d=check_plan(procedure);rows=[]
    for name in POSES:
        with mounted(name),centered(d['ready']):
            a=arena(detail='cad')
            try:
                rows.append(dict(name=name,**hold_check(a,name)))
                v=rows[-1]
                print(name,v['approved'],round(1000*v['bilateral_span_m'],2),
                      round(1000*v['least_blade_to_wrist_m'],2),flush=True)
            finally:a.close()
    _write(out,dict(plan_sha256=_hash(procedure),rows=rows))


def wrist_meshes(model):
    return [i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name in (
        *JAW,'learner_wrist','learner_wrist_camera_mount','learner_wrist_camera') and model.geom_group[i]!=3]


class GripClearance(CourtClearance):
    def __init__(self,model,*,stock):
        super().__init__(model,articulated=True)
        self.wrist=() if stock else wrist_meshes(model)
        self.blade=model.geom('learner_racket').id

    def distances(self,data,*,horizon):
        yield from super().distances(data,horizon=horizon)
        for i in self.wrist:
            r=mujoco.mj_geomDistance(self.model,data,i,self.blade,horizon,None)
            if r<horizon:yield 'learner_racket',self.arm.labels[i],r


def morph(model):
    """Arrays that must be identical. The only changed transform is the weld."""
    names=('body_pos','body_quat','body_mass','body_inertia','body_ipos','jnt_range','jnt_pos','jnt_axis',
           'dof_damping','geom_pos','geom_quat','geom_size','geom_type','actuator_ctrlrange','actuator_forcerange',
           'cam_pos','cam_quat','geom_friction','geom_contype','geom_conaffinity')
    return {n:np.copy(getattr(model,n)) for n in names}


def scan_arena(a,ready,source):
    k=Faces(a);chart=Chart(a,ready,ready[:5]);guard=chart.guard
    if not guard.clear(ready[:5]):
        return dict(reason='Actual stock ready falls outside CAD margin')
    low=[]
    for row in source:
        q0=np.array(row['prepared'])[:5];stock=np.array(row['stock_contact']['encoder'])[:5]
        checks_=[]
        for ray in row['rays']:
            p,v=np.asarray(ray['p']),np.asarray(ray['v'])
            checks_.append(dict(depth=float(p[1]),**endpoint(k,guard,p,v,q0,stock,ray['available_s'],'timed_mechanical')))
        deep=[r['tangent_m'] for r in checks_ if 'tangent_m' in r]
        low.append(dict(id=row['id'],best_m=min(deep) if deep else None,windows=checks_))
    normal=[]
    for x in XS:
        for depth in NORMAL_DEPTHS:
            for z in HEIGHTS:
                p=np.array([x,depth,z]);v=np.array([-.15*x,-3.5,-.50])
                poses=[]
                for pace,pitch in ((4.25,.07),(3.7,.18)):
                    for q in chart.poses_at(p,v,pace,pitch,ready[:5],quick=True):
                        if (q['tangent']<=GEOMETRY['normal_centered_radius_m']
                                and .011<=q['distance']<=.037 and guard.route(ready[:5],q['q'],steps=7)):
                            poses.append((q['tangent'],q['travel']))
                normal.append(dict(pos=p.tolist(),best_radius_m=min([p[0] for p in poses],default=None),
                                   travel_rad=min([p[1] for p in poses],default=None)))
    count=sum(r['best_radius_m'] is not None for r in normal)
    return dict(low=low,ordinary=normal,ordinary_covered=count)


def compare(stock,result):
    changes=[((a['best_m'] if a['best_m'] is not None else .12)-
              (b['best_m'] if b['best_m'] is not None else .12))
             for a,b in zip(stock['low'],result['low'],strict=True)]
    return dict(
        low_improvements_m=changes,
        previously_covered_lost=[a['pos'] for a,b in zip(stock['ordinary'],result['ordinary'],strict=True)
                                 if a['best_radius_m'] is not None and b['best_radius_m'] is None],
        newly_covered=[a['pos'] for a,b in zip(stock['ordinary'],result['ordinary'],strict=True)
                       if a['best_radius_m'] is None and b['best_radius_m'] is not None],
        live_eligible=sum(k>=.004 for k in changes)>=2 and result['ordinary_covered']>=.85*stock['ordinary_covered'],
    )


def geometry(procedure,checks,out,*,stock_equivalent=False):
    plan_data=check_plan(procedure);h=_read(checks);ready=np.array(plan_data['ready'])
    if h['plan_sha256']!=_hash(procedure):raise GripRejected('Jaw evidence belongs to another plan')
    source=_read(PREPARED)['rows'];results=[]
    original=None
    stock_span=h['rows'][0]['bilateral_span_m']
    for c in h['rows']:
        name=c['name']
        matched=(stock_equivalent and stock_span>=.004 and c['bilateral_span_m']>=stock_span-1e-8
                 and c['least_blade_to_wrist_m']>=GEOMETRY['minimum_blade_to_finger_m'])
        if not c['approved'] and not matched:continue
        with mounted(name),centered(ready):
            a=arena(detail='cad')
            try:
                m=a.model
                if original is None:original=morph(m)
                if any(not np.array_equal(getattr(m,n),g) for n,g in original.items()):
                    raise GripRejected('The held tool altered body, motor, camera or court parameters')
                result=scan_arena(a,ready,source)
                print(name,result.get('ordinary_covered'),[(r['id'],None if r['best_m'] is None else round(r['best_m']*1000))
                      for r in result.get('low',())],flush=True)
                qualifier='legacy stock' if name=='stock' else 'absolute' if c['approved'] else 'stock-equivalent'
                results.append(dict(name=name,qualified_by=qualifier,**result))
            finally:a.close()
    stock=results[0];eligible=[]
    for result in results[1:]:
        if 'reason' in result:continue
        result.update(compare(stock,result))
        if result['live_eligible']:eligible.append(result)
    pick=(max(eligible,key=lambda row:(sum(row['low_improvements_m']),row['ordinary_covered']))['name'] if eligible else None)
    _write(out,dict(plan_sha256=_hash(procedure),jaw_sha256=_hash(checks),low_sha256=_hash(PREPARED),chosen=pick,
          stock_equivalent=stock_equivalent,stock_bilateral_span_m=stock_span,rows=results))


def live(procedure,census,out,mode,condition):
    d=check_plan(procedure);source=_read(census);picked=source['chosen']
    if source['plan_sha256']!=_hash(procedure) or (condition!='stock' and not picked):
        raise GripRejected('The auxiliary grip has no independent mechanical qualification')
    config='stock' if condition=='stock' else picked
    if d['source']!={'plan_sha256':_hash(PLAN),'bank_sha256':_hash(BANK),'optics_sha256':_hash(OPTS),
                    'opponent':'TT37 continuous random reach, stock card and reply seeds'}:
        raise GripRejected('Frozen live roster, procedure, memory or optical preprocessing changed')

    physical_live(d,out,mode,condition,mounted(config),audit_wrist=config!='stock',
                  metadata=dict(grip=config,plan_sha256=_hash(procedure),census_sha256=_hash(census),weld=weld_for(config)))


def physical_live(d,out,mode,condition,body_context,*,audit_wrist,metadata):
    """Execute unadapted stock sensory/motor memory under a declared physical fixture."""
    cards=d['cards'];ready=d['ready'];shots=measured_launches(FEEDS)
    with np.load(BANK) as b:raw=DeepField(b,'local')
    rows=[]
    with body_context,centered(ready):
        a=arena();cad=arena(detail='cad');model=None
        camera,rig,stride,lo,hi=resources(a)
        try:
            clearance=GripClearance(cad.model,stock=not audit_wrist)
            if mode=='frozen':model=MPC(BANK,raw,variant='elastic',budget=28)
            for c in cards:
                guarded(a,clearance)
                before=a.launches
                row=play(a,camera,rig,stride,lo,hi,ready,c,shots[c['id']],raw,model,OPTS,mode,d['cap'])
                verify_clearance(a)
                if a.launches!=before+1 or (model is not None and row['memory']['before']!=row['memory']['after']):
                    raise ValueError('Frozen run had a reset or memory change')
                row['clearance']=a.clearance_result()
                rows.append(row)
                _write(out,dict(mode=mode,ready=ready,cap=d['cap'],cards=cards,rows=rows,
                                bank_sha256=_hash(BANK),optic_sha256=_hash(OPTS),**metadata))
                print(condition,mode,c['id'],row['score']['learner'],row['score']['returns'],
                      row['clearance']['penetration_samples'],flush=True)
        finally:
            if model is not None:model.close()
            camera.close();cad.close();a.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','holds','geometry','raw','frozen'))
    p.add_argument('out',type=Path)
    p.add_argument('--plan',type=Path,default=ROOT/'tt41/shallow-grip-plan.json')
    p.add_argument('--holds',type=Path,default=ROOT/'tt41/shallow-grip-holds.json')
    p.add_argument('--geometry',type=Path,default=ROOT/'tt41/shallow-grip-geometry.json')
    p.add_argument('--stock-equivalent',action='store_true',
                   help='compare actual shaft pinch to measured stock span, preserving positive blade clearance')
    p.add_argument('--condition',choices=('stock','alternate'),default='alternate')
    a=p.parse_args()
    if a.mode=='plan':plan(a.out)
    elif a.mode=='holds':holds(a.plan,a.out)
    elif a.mode=='geometry':geometry(a.plan,a.holds,a.out,stock_equivalent=a.stock_equivalent)
    else:live(a.plan,a.geometry,a.out,a.mode,a.condition)

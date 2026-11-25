"""Build and fit an opt-in, physically opaque, open-bridge wrist sensor fixture."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import tt85_wrist_overlap as original
from . import tt87_reaim_wrist as old_look
from . import tt20_fixture as fixture
from .tt19_so101 import CAMERA_BODY, GRIP_BODY, PADDLE_BODY, TIP, WRIST_CAMERA
from .tt29_vision import matrix

mpc,school=original.mpc,original.school
_read,_write,_hash=original._read,original._write,original._hash
ROOT=Path("artifacts/tt93_open_wrist")
MODEL=ROOT/"fixture.json"
BOARD=fixture._board_camera
STOCK=np.array((.006,.054,-.035))
FOOT=np.array(((-.011,.024,-.041),(.003,.024,-.041)))
BORE_RADIUS=.005
DENSITY=1850.
V=fixture._vector
SPAN_RADIUS=.0034
GATE=(4*mpc.DT,.125)
MODULE_MASS=.015


def drawing(document, spec, fovy=124.):
    """Retain the old beveled shoe and connect its edges to the PCB rear corners."""
    if fovy!=124.:
        raise ValueError("Same 124 degree wrist optics required")
    BOARD(document,fovy,alternate=False)
    dock=document.find(".//body[@name='learner_wrist_camera_mount']")
    camera=dock.find(f"body[@name='{CAMERA_BODY}']")
    pos=np.asarray(spec["aperture"]); R=np.asarray(spec["axes"])
    camera.set("pos",V(pos));camera.set("xyaxes",V(np.r_[R[:,0],R[:,1]]))
    inertial=camera.find("inertial")
    size=np.asarray((.030,.028,.016))
    moment=MODULE_MASS*(np.sum(size**2)-size**2)/12
    inertial.attrib.update(pos="0 0 .011",mass=str(MODULE_MASS),diaginertia=V(moment))
    dock.remove(dock.find("inertial"))
    keep=dock.find("geom[@name='tt29_camera_foot']")
    keep.set("density",str(DENSITY))
    dock.remove(dock.find("geom[@name='tt29_camera_bracket']"))
    # This is an open rear carrier, not a transparent or masked occluder.
    # Endpoints tie into the old foot and physically meet the rear PCB.
    for i,side in enumerate((-1,1)):
        point=pos+R@np.array([side*.011,-.010,.016])
        mid=pos+R@np.array([side*.013,-.014,.024])
        for j,(start,end) in enumerate(((FOOT[i],mid),(mid,point))):
            ET.SubElement(dock,"geom",name=f"tt93_rear_strut_{i}_{j}",type="capsule",
               fromto=V(np.r_[start,end]),size=V([SPAN_RADIUS]),density=str(DENSITY),
               rgba=".83 .80 .73 1",contype="0",conaffinity="1")
    # Retain every opaque PCB, screw, barrel, lens, arm mesh and collision.
    return document


def builder(spec):
    def open_board(document,fovy=124.,*,alternate=False):
        if alternate:
            raise ValueError("TT93 is its own explicit fixture variant")
        return drawing(document,spec,fovy)
    return patch.object(fixture,"_board_camera",open_board)


def opportunities(a):
    rows=_read(original.OUT)["rows"]
    g=a.model.body(GRIP_BODY).id
    p,b,j,qs,groups=[],[],[],[],[]
    from .tt29_rally_physics import BladeIK
    blade=BladeIK(a)
    for i,s in enumerate(rows):
        for r in s["rows"]:
            if not r["inbound_after_bounce"] or not GATE[0]<=r["lead_s"]<=GATE[1]:
                continue
            q=np.array(r["q"]); a.data.qpos[a._arm_ids[1][0]]=q
            mujoco.mj_kinematics(a.model,a.data)
            at=a.data.xpos[g].copy(); axes=a.data.xmat[g].reshape(3,3).copy()
            p.append((np.array(r["p"])-at)@axes)
            b.append((blade.pose(q[:5])[0]-at)@axes)
            j.append((a.data.site(TIP).xpos-at)@axes)
            qs.append(q);groups.append(i)
    return np.asarray(p),np.asarray(b),np.asarray(j),np.asarray(qs),np.asarray(groups),rows


def targets(a):
    ball,blade,jaw,q,groups,rows=opportunities(a)
    mask=np.array([r["case"]["group"] in original.GROUPS[:2] for r in rows])
    R0=matrix(a.model.body(CAMERA_BODY).quat)
    aim=[(R0@old_look.D@Q@old_look.D,angles) for Q,angles in old_look.views()
         if 40<=angles[0]<=75 and angles[1]%20==0 and angles[2]%60==0]
    lattice=[np.array((x,y,z)) for x in (-.053,-.045,-.036,-.027,-.018,.022,.032,.042,.052)
             for y in (.038,.050,.062,.073) for z in (-.046,-.058,-.070)
             if np.linalg.norm([x,y,z])<=.103 and np.linalg.norm(np.array([x,y,z])-STOCK)<=.067]
    # A position cannot recover a ray blocked by the untouched opaque host.
    # Run this fit-only bound without the assembly being replaced but with
    # the old keyed foot still present. Native changed-fixture builds below
    # reinsert the actual camera and new opaque rails before approval.
    m=a.model; g=m.body(GRIP_BODY).id;c=m.body(CAMERA_BODY).id
    wedge=m.geom("tt29_camera_bracket").id
    geom_group=int(m.geom_group[wedge]);m.geom_group[wedge]=5
    upper=np.zeros((len(lattice),len(q)),dtype=bool)
    blocking=Counter()
    for i,jq in enumerate(q):
        a.data.qpos[a._arm_ids[1][0]]=jq
        mujoco.mj_kinematics(m,a.data);mujoco.mj_camlight(m,a.data)
        loc=a.data.xpos[g]; frame=a.data.xmat[g].reshape(3,3)
        hit=loc+frame@ball[i]
        for k,lp in enumerate(lattice):
            eye=loc+frame@lp
            upper[k,i],name=unobstructed_ray(a,eye,hit,int(c))
            if not upper[k,i]:blocking[name]+=1
    m.geom_group[wedge]=geom_group
    best=[]
    for k,pos in enumerate(lattice):
        if upper[k].sum()<7:
            continue
        separation=np.linalg.norm(pos-STOCK)
        # Axes in original board coordinates. Vary the new position and
        # physically point it at the moving-wrist ball family.
        for R,angles in aim:
            exposed=old_look.screen(ball-pos,R@old_look.D,margin=.87,radius=.020)
            exposed&=old_look.screen(blade-pos,R@old_look.D,margin=.92)
            exposed&=old_look.screen(jaw-pos,R@old_look.D,margin=.94)
            exposed&=upper[k]
            votes=np.bincount(groups,weights=exposed,minlength=len(rows))>=2
            score=[int(votes[mask].sum()),int(votes.sum()),int(exposed.sum()),-float(separation)]
            best.append((score,dict(aperture=pos.tolist(),axes=R.tolist(),tilt_deg=angles,
                              geometric_coverage=score[:3],shift_m=separation),exposed))
    best.sort(key=lambda w:w[0],reverse=True)
    distinct=Counter()
    def unique():
        for entry in best:
            _,s,_=entry; k=tuple(s["aperture"])
            if distinct[k]>=2:
                continue
            distinct[k]+=1
            yield entry
    return list(unique()),q,groups,rows,mask,ball,dict(positions=len(lattice),largest_clear=int(upper.sum(axis=1).max()),
       viable_seat_rays=int(np.sum(upper.sum(axis=1)>=7)),obstacles=dict(blocking))


class Clear:
    def __init__(self,a):
        self.a=a
        self.m=a.model; m=a.model
        cam=int(m.body(CAMERA_BODY).id); dock=int(m.body("learner_wrist_camera_mount").id)
        self.refs=np.array([i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in (cam,dock)],dtype=int)
        grip=int(m.body(GRIP_BODY).id)
        pv=grip
        bodies={grip,m.body("learner_moving_jaw_so101_v1").id,m.body("learner_wrist").id}
        for _ in range(2):
            pv=int(m.body_parentid[pv]);bodies.add(pv)
        self.others=np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] in bodies
                         and m.geom_group[i]!=3],dtype=int)
        self.f=np.repeat(self.refs,len(self.others));self.s=np.tile(self.others,len(self.refs))
        self.radii=m.geom_rbound[self.f]+m.geom_rbound[self.s]
        self.foot=m.geom("tt29_camera_foot").id
        self.board=m.geom("tt29_camera_board").id
        # Untouched visual CAD already beds the keyed shoe 4.000 mm into
        # the second, fixed gripper shell mesh. Keep that exact intended
        # interface, but do not excuse the jaw or any new rail penetration.
        self.original_foot_seat=int(m.body(GRIP_BODY).geomadr[0])+1
        self.foot_connection=tuple(m.geom(f"tt93_rear_strut_{i}_0").id for i in range(2))
        self.circuit_connection=tuple(m.geom(f"tt93_rear_strut_{i}_1").id for i in range(2))

    def dist(self,i,j):
        return float(mujoco.mj_geomDistance(self.m,self.a.data,int(i),int(j),.018,None))

    def at(self,q):
        a=self.a;m=self.m
        a.data.qpos[a._arm_ids[1][0]]=q
        mujoco.mj_kinematics(m,a.data);mujoco.mj_camlight(m,a.data)
        delta=a.data.geom_xpos[self.f]-a.data.geom_xpos[self.s]
        test=np.einsum("ij,ij->i",delta,delta)<(self.radii+.003)**2
        failures=[]
        for i,j in zip(self.f[test],self.s[test],strict=True):
            d=self.dist(i,j)
            # A retained foot may stay bedded at its stock shaft interface;
            # all new rails and the PCB must clear the existing meshes.
            allowed=-.00402 if i==self.foot and j==self.original_foot_seat else -.00035
            if d<allowed:
                failures.append((d,m.geom(i).name or f"geom#{i}",m.geom(j).name or f"geom#{j}"))
        return sorted(failures)

    def joints_retained(self):
        return [[self.dist(i,self.foot),self.dist(j,self.board)]
                for i,j in zip(self.foot_connection,self.circuit_connection,strict=True)]


def unobstructed_ray(a,eye,point,exclude=-1):
    v=old_look.unit(point-eye)
    opt=mujoco.MjvOption()
    i=np.empty(1,dtype=np.int32)
    d=float(mujoco.mj_ray(a.model,a.data,eye,v,opt.geomgroup,True,exclude,i))
    j=int(i[0])
    if j<0:return True,None
    name=a.model.geom(j).name or a.model.body(int(a.model.geom_bodyid[j])).name
    return d>np.linalg.norm(point-eye)-.016,name


def unobstructed(a, point):
    return unobstructed_ray(a,a.data.camera(WRIST_CAMERA).xpos.copy(),point)[0]


def fit():
    ROOT.mkdir(exist_ok=True,parents=True)
    initial=_read(original.OUT)
    if not initial["complete"]:
        raise ValueError("Separate completed optical geometry required")
    raw=mpc.arena(width=64,height=48)
    try:
        choices,qs,groups,rows,mask,balls,host=targets(raw)
    finally:
        raw.close()
    trials=[];viable=[]
    # Native first-pass construction, recorded moving encoder sets and
    # original opaque wrist/paddle geometry. Whole-game CAD is later.
    do=[]
    for _,spec,visible in choices:
        if len(do)>=32:
            break
        do.append((spec,visible))
    for spec,visible in do:
        with builder(spec),mpc.centered(rows[0]["case"]["home"]),school.enabled(cameras=False):
            a=mpc.arena(width=64,height=48,detail="cad")
            try:
                guard=Clear(a); attachment=guard.joints_retained()
                c=a.model.body(CAMERA_BODY); dock=a.model.body("learner_wrist_camera_mount")
                evidence=dict(spec=spec,attachment_m=attachment,camera_mass=float(a.model.body_mass[c.id]),
                    camera_inertia=a.model.body_inertia[c.id].tolist(),mount_mass=float(a.model.body_mass[dock.id]),
                    mount_inertia=a.model.body_inertia[dock.id].tolist(),clear_frames=0)
                if any(any(d>.001 for d in r) for r in attachment):
                    evidence["reject"]="a support end did not meet its intended seat"
                    trials.append(evidence);continue
                keep=[]; fail=[]; safe=0
                gr=a.model.body(GRIP_BODY).id
                for i,q in enumerate(qs):
                    unsafe=guard.at(q)
                    if unsafe:fail.extend(unsafe)
                    else:safe+=1
                    world=a.data.xpos[gr]+a.data.xmat[gr].reshape(3,3)@balls[i]
                    # This screen uses all visible arm solids, but the live
                    # ball and actual welded paddle pose are checked later.
                    keep.append(not unsafe and bool(visible[i]) and unobstructed(a,world))
                tally=np.bincount(groups,weights=keep,minlength=len(rows))>=2
                score=[int(tally[mask].sum()),int(tally.sum()),int(sum(keep)),-spec["shift_m"]]
                evidence["score"]=score;evidence["clear_frames"]=safe
                evidence["penetrations"]=sorted(fail)[:6]
                trials.append(evidence)
                if score[0]>=3 and not fail:
                    viable.append((score,evidence))
                print("fixture",spec["aperture"],spec["tilt_deg"],score,fail[:1],flush=True)
            finally:
                a.close()
    winner=max(viable,key=lambda v:v[0])[1] if viable else None
    _write(MODEL,dict(fixture="TT93 open bridge sensor fixture, not original benchmark",winner=winner,
        source=_hash(original.OUT),size_mm=(30,28,16),polymer_kg_m3=DENSITY, aperture_m=BORE_RADIUS,
        lens_fovy=124, q_snapshots=len(qs),trials=trials,unchanged_host_sweep=host,
        baseline_foot_fixed_shell_m=-.004000000861526915,
        note="Stock fixture unmodified by default. Existing beveled shoe remains with its measured original fixed-shell seat. Two short paired rear rails carry the opaque board with computed MuJoCo inertia. No old solid blocking wedge remains. Training ray/clearance screen only."))
    print("TT93 chosen",None if winner is None else winner["spec"],flush=True)


def bore():
    """Audit that the physical aperture is not looking through its own hardware."""
    m=_read(MODEL)["winner"]
    if m is None:
        return
    with builder(m["spec"]):
        a=mpc.arena(width=64,height=48,detail="cad")
        try:
            camera=a.data.camera(WRIST_CAMERA)
            R=camera.xmat.reshape(3,3);eye=camera.xpos.copy();v=-R[:,2].copy()
            geom=np.empty(1,dtype=np.int32);out=[]
            for x,y in [(0.,0.),*((.004*np.cos(k*np.pi/4),.004*np.sin(k*np.pi/4)) for k in range(8))]:
                start=eye+x*R[:,0]+y*R[:,1]
                d=mujoco.mj_ray(a.model,a.data,start,v,mujoco.MjvOption().geomgroup,True,-1,geom)
                target=int(geom[0])
                name=None if target<0 else a.model.geom(target).name or a.model.body(int(a.model.geom_bodyid[target])).name
                out.append(dict(aperture_xy_m=[x,y],first_hit_m=None if d<0 else float(d),name=name))
            _write(ROOT/"bore.json",dict(fixture_sha256=_hash(MODEL),
                nearest_hit_m=min(q["first_hit_m"] for q in out if q["first_hit_m"] is not None),
                native_wrist_geometry_only=True,rays=out))
        finally:a.close()


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode",choices=("fit","bore"))
    {"fit":fit,"bore":bore}[a.parse_args().mode]()

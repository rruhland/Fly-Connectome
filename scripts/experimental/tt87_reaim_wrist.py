"""Fit a feasible orientation for the unchanged stock wrist module at its existing aperture."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import tt85_wrist_overlap as wrist
from . import tt20_fixture as fixture
from .tt19_so101 import CAMERA_BODY,GRIP_BODY,TIP
from .tt29_rally_physics import BladeIK
from .tt29_vision import Rig,matrix
from .tt4_arena import BALL_RADIUS

mpc,school=wrist.mpc,wrist.school
_read,_write,_hash=wrist._read,wrist._write,wrist._hash
ROOT=Path("artifacts/tt87_reaim_wrist")
PLAN,MODEL,HELD=(ROOT/n for n in ("protocol.json","geometry.json","held.json.gz"))
OCCLUSION=ROOT/"occlusion.json.gz"
BOARD=fixture._board_camera
D=np.diag([1.,-1.,-1.])
MM=.001
NOMINAL_DEADLINE=(4*mpc.DT,.125)
MOUNT=(-.00446,.00050)
CONTACT_TOL=.0005


def begin():
    old=_read(wrist.PLAN)
    train=_read(wrist.OUT)
    excluded={q["id"] for q in old["assignments"]}
    # TT78's completed evidence is untouched. Its immutable stock archive
    # roots can be held photometric witnesses here, never training.
    for n in ("artifacts/tt80_continuous_games/protocol.json",
              "artifacts/tt83_population_search/protocol.json","artifacts/tt75/protocol.json"):
        if Path(n).exists():
            excluded.update(wrist.ids(_read(Path(n))))
    srcs=(*old["data"],*(n for n in (
        "artifacts/tt46_season/after128.json.gz",
        "artifacts/tt48_transfer/live-plain.json.gz",
        "artifacts/tt48_transfer/live-exploit.json.gz",
        "artifacts/tt49_elite/season-24.json.gz",
        "artifacts/tt49_body/season-24.json.gz",
        "artifacts/tt42/after12.json.gz",
        "artifacts/tt42/direct-execution-held.json.gz",
        "artifacts/tt52_matched_movements/recover-live-24.json.gz",
    ) if Path(n).exists()))
    rare={}
    cards=set()
    for src in srcs:
        d=_read(Path(src))
        if not all(k in d for k in ("rows","cards","ready")):
            continue
        for i,(c,r) in enumerate(zip(d["cards"],d["rows"])):
            if c["id"] in excluded or c["opening"]!="random" or c["profile"]!="reach":
                continue
            for o in r["placements"]["delivered"]:
                g=wrist.classify(o)
                if g is None:
                    continue
                p=o["receive_line"]; x,z,*_=wrist.central.deep(np.r_[p["xyz"],p["velocity"]])
                if o["far_contact"]["time"]>14:
                    continue
                rare.setdefault(g,[]).append(dict(id=c["id"],group=g,source=src,card=c,i=i,leg=o["leg"],
                                                 x=float(x),z=float(z),home=d["ready"]))
    rng=np.random.default_rng(870419)
    roster=[]
    for g in wrist.GROUPS:
        pool=rare.get(g,[])
        order=[pool[j] for j in rng.permutation(len(pool))]
        for sign in (-1,1,-1):
            opts=[r for r in order if r["id"] not in cards]
            if not opts:
                break
            p=min(opts,key=lambda w:(sign*w["x"]<0,abs(w["x"])>.25))
            cards.add(p["id"]);roster.append(p)
    if not train["complete"]:
        raise ValueError("Complete the original wrist evidence first")
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema="tt87-physical-reaim-1",tt85_protocol=_hash(wrist.PLAN),training=_hash(wrist.OUT),
       source={n:_hash(Path(n)) for n in srcs},held=roster,excluded=sorted(excluded),aperture_m=[.006,.054,-.035],fovy=124,
       tolerance_m=CONTACT_TOL,housing_mount_interface_tolerance_m=CONTACT_TOL, bracket_interval_m=MOUNT,
       scope="Changed sensor geometry only. Physically rotate lens and entire opaque PCB/sensor/barrel/screws about the existing gripper-fixed aperture, retaining original unchanged stock bracket, jaw and other links. Optimize exclusively on archived actual moving RGB-controlled opportunities. Freeze full assembly before release-disjoint physical RGB validation. Look for multiple verified unoccluded images before normal receiving deadline with the blade also visible. No changes to regular benchmark or motion labels."))
    print("TT87 held",Counter(c["group"] for c in roster),[c["id"] for c in roster],flush=True)


def design():
    p=_read(PLAN)
    if p["training"]!=_hash(wrist.OUT) or p["tt85_protocol"]!=_hash(wrist.PLAN):
        raise ValueError("Original fixture measurements changed")
    for s,h in p["source"].items():
        if _hash(Path(s))!=h:
            raise ValueError("Archived command tape changed")
    return p


def unit(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True)


def views():
    yield np.eye(3),[0.,0.,0.]
    for theta in range(5,101,5):
        for phi in range(0,360,10):
            P,T=np.deg2rad([phi,theta])
            w=np.array([np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T)])
            base=np.array([0.,1.,0.])
            if abs(w@base)>.96:
                base=np.array([1.,0.,0.])
            x=unit(np.cross(base,w)); y=np.cross(w,x)
            frame=np.column_stack([x,y,w])
            for roll in (0,30,60,90,120,150,180,210,240,270,300,330):
                Q=frame@Rotation.from_rotvec([0.,0.,np.deg2rad(roll)]).as_matrix()
                yield Q,[theta,phi,roll]


def screen(local, q, *, margin, radius=0.):
    u=local@q
    x,y,z=np.moveaxis(u,-1,0)
    d=np.maximum(.003,z)
    # Coordinates normalized to depth, with a conventional close-focus margin.
    half_y=np.tan(np.deg2rad(124/2))
    return (z>.025)&((abs(x)+radius)<d*half_y*4/3*margin)&((abs(y)+radius)<d*half_y*margin)


def sampled(obs,a):
    rig=Rig.from_arena(a)
    kin=BladeIK(a)
    pairs=[];allp=[]; blade=[]; grip=[]; tags=[]; j=[]
    for i,s in enumerate(obs["rows"]):
        for r in s["rows"]:
            if (not r["inbound_after_bounce"] or not NOMINAL_DEADLINE[0]<=r["lead_s"]<=NOMINAL_DEADLINE[1]):
                continue
            q=np.array(r["q"]); c=rig.wrist.at(q)
            b,_=kin.pose(q[:5])
            a.data.qpos[a._arm_ids[1][0]]=q
            mujoco.mj_kinematics(a.model,a.data)
            g=a.data.site(TIP).xpos.copy()
            allp.append(c.rotation@(np.asarray(r["p"])-c.eye))
            blade.append(c.rotation@(b-c.eye))
            grip.append(c.rotation@(g-c.eye))
            tags.append(i);j.append(q)
        pairs.append(s["case"]["group"])
    return *(np.stack(v) for v in (allp,blade,grip)),np.array(tags),pairs,np.asarray(j)


class Enclosure:
    """Full opaque module surrounds the same aperture; no ray may pass through it or the wrist."""
    def __init__(self,a):
        self.a=a;self.m=a.model;self.d=a.data
        self.motor_axes=a._arm_ids[1][0]
        self.body=self.m.body(CAMERA_BODY)
        self.nom=matrix(self.body.quat)
        self.anchor=self.body.pos.copy()
        self.q0=a.joints.copy()
        dock=int(self.body.parentid[0])
        grip=self.m.body(GRIP_BODY)
        include={dock,int(grip.id),int(grip.parentid[0])}
        # Include moving jaws and original neighboring wrist/forearm pieces.
        for i in range(self.m.nbody):
            if i==self.body.id:
                continue
            parent=int(self.m.body_parentid[i])
            if parent in include and self.m.body(i).name.startswith("learner_"):
                include.add(i)
        self.geoms=[i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in include]
        lo=int(self.body.geomadr[0]); hi=lo+int(self.body.geomnum[0])
        self.module=list(range(lo,hi))
        self.board=self.m.geom("tt29_camera_board").id
        self.mount=self.m.geom("tt29_camera_bracket").id
        self.pairs=[(i,j) for i in self.module for j in self.geoms]
        self.defined=[]
        self.install(np.eye(3),self.q0)
        for i,j in self.pairs:
            dist=self.distance(i,j)
            allowed=min(-CONTACT_TOL,dist-CONTACT_TOL) if j==self.mount else -CONTACT_TOL
            self.defined.append((i,j,allowed,dist))

    def distance(self,i,j):
        return float(mujoco.mj_geomDistance(self.m,self.d,i,j,.026,None))

    def install(self,Q,q):
        R=self.nom@D@Q@D
        mujoco.mju_mat2Quat(self.body.quat,np.ascontiguousarray(R).reshape(9))
        self.a.data.qpos[self.motor_axes]=q
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_camlight(self.m,self.d)

    def check(self,Q,q,verbose=False):
        self.install(Q,q)
        attach=self.distance(self.board,self.mount)
        worst=None
        breaks=False
        for i,j,lo,old in self.defined:
            dist=self.distance(i,j)
            if dist<lo:
                number=dist-lo
                if worst is None or number<worst[0]:
                    worst=(number,self.m.geom(i).name or f"geom#{i}",self.m.geom(j).name or f"geom#{j}",dist,lo)
                if not verbose:
                    breaks=True;break
        if not verbose:
            return bool(not breaks and MOUNT[0]<=attach<=MOUNT[1])
        return dict(admissible=bool(worst is None and MOUNT[0]<=attach<=MOUNT[1]),mount_gap_m=attach,worst=worst)

    def angle(self,Q):
        a=self.nom@D@Q@D
        return a


def find():
    p=design()
    d=_read(wrist.OUT)
    weights=np.asarray([s["case"]["group"] in wrist.GROUPS[:2] for s in d["rows"]])
    with mpc.centered(d["rows"][0]["case"]["home"]),school.enabled(cameras=False):
        a=mpc.arena(width=64,height=48,detail="cad")
        try:
            balls,blade,jaw,groups,labels,joints=sampled(d,a)
            fit=Enclosure(a)
            feasibility=Counter(); options=[]; informative=0; best_view=None
            for Q,rotation in views():
                vis=screen(balls,Q,margin=.90,radius=BALL_RADIUS)
                vis&=screen(blade,Q,margin=.90)
                vis&=screen(jaw,Q,margin=.94)
                hits=np.bincount(groups,weights=vis,minlength=len(labels))>=2
                difficult=int(hits[weights].sum()); all_groups=int(hits.sum()); visible=int(vis.sum())
                base=[difficult,all_groups,visible,-rotation[0]]
                if best_view is None or base>best_view["score"]:
                    best_view=dict(score=base,rotation=rotation)
                if difficult<2:
                    continue
                informative+=1
                if not fit.check(Q,fit.q0):
                    continue
                feasibility["admissible_reference_geometry"]+=1
                quiet=[fit.check(Q,q) for q in joints]
                if not all(quiet):
                    feasibility["blocked_somewhere_in_moving_wrist"]+=1
                    continue
                feasibility["admissible_all_moving_wrist"]+=1
                options.append((base,Q,rotation))
            options.sort(key=lambda x:x[0],reverse=True)
            winner=None
            if options:
                s,Q,angle=options[0]
                witnesses=[fit.check(Q,q,verbose=True) for q in joints]
                winner=dict(score=s,rotation=angle,relative=Q.tolist(),body_rotation=fit.angle(Q).tolist(),
                            minimum_mount_gap_m=min(w["mount_gap_m"] for w in witnesses),
                            maximum_mount_gap_m=max(w["mount_gap_m"] for w in witnesses),witnesses=witnesses)
            _write(MODEL,dict(plan_sha256=_hash(PLAN),frames=len(balls),geometric_best_unconstrained=best_view,
                            geometry_viable_orientations=informative,physical=feasibility,
                            winner=winner,aperture=fit.anchor.tolist(),helmet_preserved=True))
            print("TT87 geometry",[len(balls),best_view,informative,dict(feasibility)],
                  None if not winner else [winner["rotation"],winner["score"]],flush=True)
        finally:
            a.close()


def swap(rotation):
    def reaim(document,fovy=124.,*,alternate=False):
        if alternate:
            raise ValueError("TT87 compares the actual original bracket only.")
        BOARD(document,fovy,alternate=False)
        body=document.find(f".//body[@name='{CAMERA_BODY}']")
        body.set("xyaxes",fixture._vector(np.r_[rotation[:,0],rotation[:,1]]))
    return patch.object(fixture,"_board_camera",reaim)


def capture():
    p=design(); g=_read(MODEL)
    if g["plan_sha256"]!=_hash(PLAN):
        raise ValueError("Optic was not frozen before seeing holds")
    if not g["winner"]:
        _write(HELD,dict(rows=[],complete=True,geometry_sha256=_hash(MODEL),
                        status="No simultaneous geometry passes physical enclosure. Held captures unopened."))
        return
    result=_read(HELD) if HELD.exists() else dict(rows=[],complete=False,geometry_sha256=_hash(MODEL))
    if result["geometry_sha256"]!=_hash(MODEL):
        raise ValueError("Physical trials cannot change orientation")
    for spec in p["held"][len(result["rows"]):]:
        doc=_read(Path(spec["source"])); row=doc["rows"][spec["i"]]
        notes={}
        for mode in ("stock","reaim"):
            rotation=np.asarray(g["winner"]["body_rotation"])
            with mpc.centered(spec["home"]),school.enabled(cameras=False):
                # To preserve exact achieved arm and ball motions, only the
                # camera assembly's view pose is changed at exposure. The
                # retained opaque module and bracket are ray-traced in RGB.
                a=mpc.arena(width=384,height=288)
                try:
                    if mode=="reaim":
                        from . import tt85_wrist_overlap as prior
                        original=prior.frame_data
                        module=a.model.body(CAMERA_BODY)
                        quat0=module.quat.copy()
                        quat=np.empty(4)
                        mujoco.mju_mat2Quat(quat,np.ascontiguousarray(rotation).reshape(9))
                        other=Rig.from_arena(a)
                        def frame(one,rig,lens,tick):
                            module.quat[:]=quat
                            try:
                                rig.wrist=Rig.from_arena(a).wrist
                                return original(one,rig,lens,tick)
                            finally:
                                module.quat[:]=quat0
                                rig.wrist=other.wrist
                                mujoco.mj_kinematics(one.model,one.data)
                                mujoco.mj_camlight(one.model,one.data)
                        with patch.object(prior,"frame_data",frame):
                            out=wrist.inspect(a,other,spec,row)
                    else:
                        out=wrist.inspect(a,Rig.from_arena(a),spec,row)
                    notes[mode]=out
                finally:
                    a.close()
        result["rows"].append(dict(case=spec,views=notes));_write(HELD,result)
        print("TT87 held",spec["id"],[(k,wrist.per_shot(v)[1]) for k,v in notes.items()],flush=True)
    result["complete"]=True;_write(HELD,result)


def lines(a,exclude):
    """Evaluator-only rays to nine known surface directions, with the host still opaque."""
    camera=a.data.camera("tt19_wrist")
    center=a.ball.copy(); eye=camera.xpos.copy()
    v=unit(center-eye)
    v0=np.array([0.,0.,1.]) if abs(v[2])<.96 else np.array([1.,0.,0.])
    dx=unit(np.cross(v,v0));dy=np.cross(v,dx)
    target=np.vstack((center,np.array([center+.78*BALL_RADIUS*(np.cos(t)*dx+np.sin(t)*dy)
         for t in np.arange(8)*(2*np.pi/8)])))
    geoms=np.empty(1,dtype=np.int32);obstacles=[]
    show=mujoco.MjvOption().geomgroup
    for point in target:
        direction=unit(point-eye)
        hit=float(mujoco.mj_ray(a.model,a.data,eye,direction,show,True,exclude,geoms))
        g=int(geoms[0])
        obstacles.append(dict(geom=None if g<0 else a.model.geom(g).name or f"geom#{g}",
                              owner=None if g<0 else a.model.body(int(a.model.geom_bodyid[g])).name,
                              distance_m=hit, ball=bool(g==a._geom_ids["ball"])))
    return obstacles


def occlusion():
    """Can any camera aim possibly see these rays if the mounting aperture stays fixed?"""
    plan=design();model=_read(MODEL);held=_read(HELD)
    if not model["winner"] or not held["complete"]:
        return
    rotation=np.array(model["winner"]["body_rotation"])
    qq=np.empty(4);mujoco.mju_mat2Quat(qq,np.ascontiguousarray(rotation).reshape(9))
    docs={};tapes=[]
    for stage,rows in (("training",_read(wrist.OUT)["rows"]),("held",held["rows"])):
        for r in rows:
            s=r["case"]
            frames=(r["rows"] if stage=="training" else r["views"]["stock"]["rows"])
            take={f["tick"]:f for f in frames if f["inbound_after_bounce"] and
                  NOMINAL_DEADLINE[0]<=f["lead_s"]<=NOMINAL_DEADLINE[1]}
            if s["source"] not in docs:
                docs[s["source"]]=_read(Path(s["source"]))
            doc=docs[s["source"]]
            row=doc["rows"][s["i"]]
            rec=[]
            if not take:
                tapes.append(dict(stage=stage,id=s["id"],group=s["group"],rows=rec))
                continue
            with mpc.centered(s["home"]),school.enabled(cameras=False):
                a=mpc.arena(width=384,height=288)
                try:
                    m=a.model; c=m.body(CAMERA_BODY); q0=c.quat.copy()
                    a.launch(row["shot"],card=s["card"]);a.partner.reset(seed=s["card"]["seed"])
                    for t,u in enumerate(row["actions"][:max(take)+1]):
                        if t in take:
                            if not np.allclose(a.ball,take[t]["p"],rtol=0.,atol=1.e-10):
                                raise ValueError("Same motion no longer has same optical ray")
                            c.quat[:]=qq
                            mujoco.mj_kinematics(m,a.data);mujoco.mj_camlight(m,a.data)
                            with_host=lines(a,-1)
                            no_module=lines(a,int(c.id))
                            if stage=="held" and len(rec)==1 and s["id"] in (2741,9230):
                                from PIL import Image
                                k=ROOT/"frames";k.mkdir(exist_ok=True)
                                Image.fromarray(a.render_wrist()).save(k/f"reaim-{s['id']}-tick{t}.png")
                            c.quat[:]=q0
                            mujoco.mj_kinematics(m,a.data);mujoco.mj_camlight(m,a.data)
                            rec.append(dict(tick=t,lead_s=take[t]["lead_s"],full=with_host,without_rotated_module=no_module))
                        a.exchange(u,mpc.FAR_COMMAND)
                finally:
                    a.close()
            tapes.append(dict(stage=stage,id=s["id"],group=s["group"],rows=rec))
    _write(OCCLUSION,dict(plan_sha256=_hash(PLAN),optics_sha256=_hash(MODEL),held_sha256=_hash(HELD),
             note="Ground truth evaluator only. Nine sightline rays traverse the full visible scene. Compare the candidate module and same fixed aperture with just the rotating PCB assembly excluded from ray evaluation. The original jaw, paddle, bracket, and aperture position are unchanged.",
             rows=tapes,complete=True))


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("mode",choices=("plan","fit","capture","occlusion"))
    {"plan":begin,"fit":find,"capture":capture,"occlusion":occlusion}[a.parse_args().mode]()

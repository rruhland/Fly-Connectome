"""Audit the stock wrist RGB cue before normal moving-arm contact."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt56_central_training as central
from .tt19_so101 import WRIST_CAMERA
from .tt29_fast_camera import native_ball_pixels
from .tt29_vision import Lens, Rig
from .tt4_arena import BALL_RADIUS
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

mpc=school.mpc
ROOT=Path("artifacts/tt85_wrist_overlap")
PLAN=ROOT/"protocol.json"
OUT=ROOT/"visibility.json.gz"
SUMMARY=ROOT/"summary.json"
SOURCES=(school.season.ROOT/"learning.json.gz",Path("artifacts/tt48_transfer/continued-64.json.gz"),
         Path("artifacts/tt53_responsive_school/ordinary-live.json.gz"),
         Path("artifacts/tt42/after48.json.gz"),Path("artifacts/tt42/diverse-evaluation.json.gz"),
         Path("artifacts/tt44_combined/combined.json.gz"),
         Path("artifacts/tt52_practice_dose/live-from-12.json.gz"),
         Path("artifacts/tt52_practice_dose/live-from-48-96.json.gz"))
TIME_BANDS=((.050,.08333),(.08333,.11667),(.11667,.168),(.168,.235))
MIN_CONTROL_TICKS=4
LIMIT=4
DT=mpc.DT
RNG=850593
GROUPS=("central-low-no-touch","central-or-wing-low-unscored-touch","low-scored","normal-scored")


def classify(o):
    f=o.get("receive_line")
    if not f or not o.get("near_bounce") or not o.get("far_contact"):
        return None
    x,z,*_=central.deep(np.r_[f["xyz"],f["velocity"]])
    if abs(x)>.30 or not .94<z<1.18:
        return None
    if not o["near_landed"] and abs(x)<.10 and z<1.012 and not o["near_contact"]:
        return GROUPS[0]
    if not o["near_landed"] and z<1.042 and o["near_contact"]:
        return GROUPS[1]
    if o["near_landed"] and z<1.035:
        return GROUPS[2]
    if o["near_landed"] and z>1.057:
        return GROUPS[3]


def ids(obj):
    if isinstance(obj,dict):
        yield from (v for k,v in obj.items() if k=="id" and isinstance(v,int))
        for v in obj.values():
            yield from ids(v)
    if isinstance(obj,list):
        for v in obj:
            yield from ids(v)


def freeze():
    other=[Path(p) for p in ("artifacts/tt80_continuous_games/protocol.json",
       "artifacts/tt83_population_search/protocol.json","artifacts/tt78_empirical_switch/protocol.json",
       "artifacts/tt75/protocol.json")]
    forbidden=set()
    for p in other:
        if p.exists():
            obj=_read(p)
            forbidden.update(x for x in ids(obj) if x>=0)
    raw={}
    for source in SOURCES:
        d=_read(source)
        for i,(c,row) in enumerate(zip(d["cards"],d["rows"])):
            if c["id"] in forbidden or c["profile"]!="reach" or c["opening"]!="random":
                continue
            for o in row["placements"]["delivered"]:
                name=classify(o)
                if name is None:
                    continue
                t=o["far_contact"]["time"]
                if t>16 or o["near_bounce"]["time"]-t>.91:
                    continue
                lane=float(central.deep(np.r_[o["receive_line"]["xyz"],o["receive_line"]["velocity"]])[0])
                raw.setdefault(name,[]).append(dict(group=name,id=c["id"],leg=o["leg"],x=lane,card=c,
                                                    i=i,source=str(source),home=d["ready"],far_time=t))
    taken=set()
    rng=np.random.default_rng(RNG)
    selected=[]
    for name in GROUPS:
        opts=raw.get(name,[])
        ranked=[opts[i] for i in rng.permutation(len(opts))]
        doors=[-1,1,-1,1]
        for sign in doors:
            ok=[q for q in ranked if q["id"] not in taken]
            if not ok:
                raise ValueError(f"No balls left in {name}")
            q=min(ok,key=lambda x:(sign*x["x"]<0,x["far_time"]//4))
            taken.add(q["id"]);selected.append(q)
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema="tt85-stock-moving-wrist-1",assignments=selected,
         data={str(p):_hash(p) for p in SOURCES}, forbidden=sorted(forbidden),
         bounds=dict(rate_hz=60,resolution=(384,288),min_control_ticks=MIN_CONTROL_TICKS,
                     bands_s=TIME_BANDS, retained_visible_min_count=2, latest_predeadline_s=.125),
         goal="Measure whether the ordinary, opaque, fixed board currently sees the incoming ball before the motor deadline on actual moving strike tapes. Ground truth may only identify matched native observations and quantify detector error. No actuator, camera calibration, mount, fixture, jaw, glare, markers, object ids, or ball appearance changes. Distinct source IDs, no opened held outcome or returned ball alters the view."))
    print(Counter(q["group"] for q in selected),[q["id"] for q in selected],flush=True)


def native_fit(camera,lens):
    return lens.at(camera.xpos.copy(),camera.xmat.reshape(3,3).copy())


def angular_error(camera,pixel,position):
    ray=camera.ray(pixel)
    d=position-camera.eye
    s=float(ray@d/(ray@ray))
    return float(np.linalg.norm(d-s*ray))


def silhouette(frame,prediction,radius):
    """Independent ball pixels then score time-coincident match."""
    found=native_ball_pixels(frame)
    if found is None:
        return None,False
    matched=bool(prediction is not None and
            np.linalg.norm(found.center-prediction)<max(3.,min(12.,.52*radius)))
    return found,matched


def frame_data(a,rig,lens,tick):
    frame=a.render_wrist()
    # Everything below which uses the real position is per-frame evaluation
    # only, never the detector or the encoder calibration.
    camera=native_fit(a.data.camera(WRIST_CAMERA),lens)
    q=a.joints.copy(); b=a.ball.copy(); v=a.velocity.copy()
    xyz=camera.rotation@(b-camera.eye)
    proj=camera.project(b)[:2] if xyz[2]>0.003 else None
    radius=lens.focal*BALL_RADIUS/max(.003,xyz[2])
    if proj is not None:
        full=bool(np.min(np.r_[proj,[lens.width-1,lens.height-1]-proj])>radius)
        overlap=bool(np.min(np.r_[proj,[lens.width-1,lens.height-1]-proj])>-radius)
    else:
        full=overlap=False
    s,match=silhouette(frame,proj,radius)
    # Simple extraction of the ordinary red blade, not a role/geom id image.
    rgb=frame.astype(np.int16);r,g,b0=np.moveaxis(rgb,-1,0)
    blade=(r>80)&(r-g>47)&(r-b0>36)&(g-b0<16)&(g<.56*r)
    yy,xx=np.nonzero(blade)
    seen=a._received_bounce
    if match:
        fitted=rig.wrist.at(q)
        lens_error=angular_error(camera,s.center,b)
        q_error=angular_error(fitted,s.center,b)
    else:
        lens_error=q_error=None
    qd=dict(tick=tick,time=float(a.data.time),p=b.tolist(),v=v.tolist(),q=q.tolist(),
            inbound_after_bounce=bool(seen and v[1]<-.65),
            expected=None if proj is None else proj.tolist(),depth=float(xyz[2]),expected_radius=float(radius),
            camera_ray_screen=overlap,full_silhouette_in_frame=full,match=match,
            sight=None if s is None else dict(center=s.center.tolist(), radius=s.radius, area=s.area,clipped=s.clipped),
            blade_pixels=len(xx),blade_center=None if not len(xx) else [float(np.mean(xx)),float(np.mean(yy))],
            bearing_m=lens_error,encoder_bearing_m=q_error)
    return qd,frame


def inspect(a,rig,spec,row,save=False):
    c=spec["card"]; o=row["placements"]["delivered"][spec["leg"]]
    hit=o["near_contact"]; ref=o["receive_line"]; bounce=o["near_bounce"]
    predicted=(bounce["time"]+(bounce["ball"][1]+1.53)/max(1.,-.98*ref["velocity"][1]))
    event=float(hit["time"] if hit else predicted)
    start=max(0,int((event-.30)/DT)); stop=int(np.ceil((event+(.065 if hit else .12))/DT))
    a.launch(row["shot"],card=c);a.partner.reset(seed=c["seed"])
    data=[]; picture=None
    l=rig.wrist.lens
    crossing=None
    for tick,u in enumerate(row["actions"]):
        if tick>stop:
            break
        if tick>=start:
            t=a.data.time
            ball=a.ball.copy()
            if ball[1]<-1.53 and a.velocity[1]<-.65 and crossing is None:
                crossing=t
            log,frame=frame_data(a,rig,l,tick)
            data.append(log)
            if save and (picture is None or abs(t-event+.075)<picture[0]):
                picture=(abs(t-event+.075),frame.copy(),tick)
        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
        if done or cap:
            break
    seen=a.point_placements()["delivered"][spec["leg"]]
    if digest_json(seen["far_contact"])!=digest_json(o["far_contact"]) or digest_json(seen["near_bounce"])!=digest_json(o["near_bounce"]):
        raise ValueError(f"Physical incoming tape diverged on {spec['id']}")
    if hit is not None and digest_json(seen["near_contact"])!=digest_json(hit):
        raise ValueError(f"Measured racket timing changed on {spec['id']}")
    end=(float(hit["time"]) if hit else
         (float(crossing-DT*.5) if crossing is not None else min(float(a.data.time),event)))
    if not any(r["inbound_after_bounce"] and r["time"]<end for r in data):
        raise ValueError(f"No measured precontact wrist interval {spec['id']}")
    for r in data:
        r["lead_s"]=float(end-r["time"])
    if picture is not None:
        from PIL import Image
        folder=ROOT/"frames"
        folder.mkdir(exist_ok=True)
        Image.fromarray(picture[1]).save(folder/f"{spec['group']}-{spec['id']}-tick{picture[2]}.png")
    return dict(clock=end, clock_kind="actual racket impact" if hit else
                ("native virtual y=-1.53 crossing" if crossing is not None else "terminal-or-y projection"),
                hit=hit,actual_received_bounce=o["near_bounce"],rows=data)


def audit(limit=None):
    p=_read(PLAN)
    if any(_hash(Path(n))!=h for n,h in p["data"].items()):
        raise ValueError("The archived tape changed")
    result=_read(OUT) if OUT.exists() else dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    if result["plan_sha256"]!=_hash(PLAN):
        raise ValueError("A different split is already underway")
    sources={}
    for spec in p["assignments"][len(result["rows"]):limit]:
        n=spec["source"]
        if n not in sources:
            sources[n]=_read(Path(n))
        src=sources[n]["rows"][spec["i"]]
        with mpc.centered(spec["home"]),school.enabled(cameras=False):
            a=mpc.arena(width=384,height=288)
            try:
                rig=Rig.from_arena(a)
                if a.wrist_optics!="board" or a.wrist_fovy!=124 or a.fixture!="table-edge":
                    raise ValueError("This check requires the original unmodified stock board and clamp")
                seen=inspect(a,rig,spec,src,not any(v["case"]["group"]==spec["group"] for v in result["rows"]))
            finally:
                a.close()
        result["rows"].append(dict(case=spec,**seen))
        result["complete"]=len(result["rows"])==len(p["assignments"]);_write(OUT,result)
        valid,summary=per_shot(seen)
        print(spec["group"],spec["id"],len(valid),summary,flush=True)


def per_shot(d):
    candidate=[r for r in d["rows"] if r["inbound_after_bounce"] and r["lead_s"]>0]
    ready=[r for r in candidate if MIN_CONTROL_TICKS*DT<=r["lead_s"]<=.125]
    matches=[r for r in ready if r["match"] and not r["sight"]["clipped"]]
    n_known=all(r["blade_pixels"]>=30 for r in matches)
    return candidate,dict(early_frame_count=len(ready),useful=len(matches)>=2,matched_early=len(matches),
                          ready_with_blade=n_known and len(matches)>=2,
                          frame_geometry_count=sum(r["camera_ray_screen"] for r in ready),
                          last_lead_s=min([r["lead_s"] for r in candidate if r["match"]],default=None),
                          encoder_bearing_mm=(float(1000*np.median([r["encoder_bearing_m"] for r in matches])) if matches else None))


def summarize():
    d=_read(OUT)
    if not d["complete"]:
        raise ValueError("Visibility sample not complete")
    total=Counter();details=[];table={g:Counter() for g in GROUPS}
    band_counts=np.zeros((len(TIME_BANDS),5),dtype=int)
    late=[]; axes=Counter()
    for s in d["rows"]:
        rows,k=per_shot(s);key=s["case"]["group"]
        t=Counter(shots=1,useful=k["useful"],with_blade=k["ready_with_blade"],
                  geometry_frames=k["frame_geometry_count"],early_frames=k["early_frame_count"],early_visible=k["matched_early"])
        total.update(t);table[key].update(t)
        for r in rows:
            if MIN_CONTROL_TICKS*DT<=r["lead_s"]<=.125:
                axes.update(behind_lens=int(r["depth"]<=.003),
                            in_front_outside_frustum=int(r["depth"]>.003 and not r["camera_ray_screen"]))
            if r["match"] and r["lead_s"]<.050:
                late.append(dict(id=s["case"]["id"],tick=r["tick"],lead_s=r["lead_s"],
                                 clipped=r["sight"]["clipped"],bearing_mm=1000*r["encoder_bearing_m"]))
        for i,(lo,hi) in enumerate(TIME_BANDS):
            rr=[r for r in rows if lo<=r["lead_s"]<hi]
            band_counts[i]+=[len(rr),sum(r["camera_ray_screen"] for r in rr),
                             sum(r["match"] for r in rr),sum(r["match"] and not r["sight"]["clipped"] for r in rr),
                             sum(r["match"] and r["blade_pixels"]>=30 for r in rr)]
        details.append(dict(id=s["case"]["id"],group=key,**k))
    keep=bool(sum(table[g]["useful"] for g in GROUPS[:2])>=3)
    _write(SUMMARY,dict(total=dict(total),groups={k:dict(v) for k,v in table.items()},
                        physical_capture_sha256=_hash(OUT),bands_s=TIME_BANDS, band_columns=("frames","frustum_overlap",
                        "RGB_verified","unclipped","with_blade"),bands=band_counts.tolist(),details=details,
                        deadline_geometry=dict(axes),late_visible=late,open_fusion=keep))
    print("all",dict(total),"by group",{k:dict(v) for k,v in table.items()},"open_fusion",keep,flush=True)


def late_frame():
    """Reproduce one confirmed late raw image as a visual receipt."""
    from PIL import Image
    s=next(r for r in _read(OUT)["rows"] if r["case"]["id"]==3355)
    obs=next(r for r in s["rows"] if r["match"] and r["lead_s"]>0)
    case=s["case"]; old=_read(Path(case["source"]))["rows"][case["i"]]
    with mpc.centered(case["home"]),school.enabled(cameras=False):
        a=mpc.arena(width=384,height=288)
        try:
            card=case["card"]; a.launch(old["shot"],card=card);a.partner.reset(seed=card["seed"])
            for u in old["actions"][:obs["tick"]]:
                a.exchange(u,mpc.FAR_COMMAND)
            rig=Rig.from_arena(a)
            row,frame=frame_data(a,rig,rig.wrist.lens,obs["tick"])
            if not row["match"] or not np.array_equal(row["p"],obs["p"]):
                raise ValueError("Late raw image failed to reproduce")
            Image.fromarray(frame).save(ROOT/"frames"/f"late-{case['id']}-tick{obs['tick']}.png")
        finally:
            a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("plan","capture","summary","image"))
    p.add_argument("--n",type=int);a=p.parse_args()
    if a.mode=="capture":audit(a.n)
    else:{"plan":freeze,"summary":summarize,"image":late_frame}[a.mode]()

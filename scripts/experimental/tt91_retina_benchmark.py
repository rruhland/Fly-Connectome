"""Benchmark unchanged high-rate RGB clocks and paired optics on actual held deliveries."""
import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np

from . import tt75_incoming_state as parent
from .tt29_fast_camera import native_ball_pixels,_sighting_equal
from .tt75_dense_fit import DenseFlight
from .tt75_retina import Retina
from .tt91_packed_retina import PackedRetina,CulledRetina,CulledMonoRetina
from .tt4_transfer_trial import _hash,_read,_write

school,mpc=parent.school,parent.school.mpc
ROOT=Path("artifacts/tt91_retina_budget")
PLAN=ROOT/"protocol.json"
OUT=ROOT/"rgb.json.gz"
FACTORIES=dict(original=Retina,packed=PackedRetina,culled=CulledRetina,mono=CulledMonoRetina)
MODES=dict(original=Retina,mono=CulledMonoRetina)


def freeze():
    p=_read(parent.PLAN)
    held=_read(parent.ROOT/"held.json.gz")
    if not held["complete"] or held["plan_sha256"]!=_hash(parent.PLAN):
        raise ValueError("Seal optical holdout before the speed check")
    ROOT.mkdir(exist_ok=True,parents=True)
    sources={**p["sources"],**{str(q):_hash(q) for q in (parent.PLAN,parent.ROOT/"held.json.gz",parent.ROOT/"sensor-calibration.json")}}
    _write(PLAN,dict(rows=p["rows"],settings=p["settings"],sources=sources,modes=list(MODES),
                     role="Paired actual 120 Hz images on the same physical held control tape predeclared by TT75; standard partner and actual commands unchanged. One pass of native plant. Each renderer maintains separate causal ROI history and DenseFlight. Rotate execution order each exposure. Timings exclude separate parity/precision analysis. Geometry culling occurs only inside the ordinary RGB renderer and applies to every bounded scene object, not to a privileged observation."))


def same(a,b):
    if a is None or b is None:return a is b
    return np.array_equal(a,b)


def packet(value):
    if value is None:return None
    (x,w),fit=value
    return dict(pv=x.tolist(),spin=w.tolist(),bounce=fit.bounce,residual=float(fit.residual),
                posterior=fit.posterior.tolist())


def run(n):
    p=_read(PLAN)
    if any(_hash(Path(f))!=h for f,h in p["sources"].items()):
        raise ValueError("A frozen optical test source changed")
    result=dict(schema="tt91-optical-cost-and-parity-1",plan=_hash(PLAN),rows=[],complete=False)
    if OUT.exists():
        result=_read(OUT)
        if result["plan"]!=_hash(PLAN):raise ValueError("Changed optical benchmark")
    docs={}
    with school.enabled():
        for case in p["rows"][len(result["rows"]):n]:
            began=process_time()
            src=case["source"]
            if src not in docs:docs[src]=_read(Path(src))
            d=docs[src]; card, tape=d["cards"][case["i"]],d["rows"][case["i"]]
            with mpc.centered(d["ready"]):
                a=mpc.arena();wide,rig,*_=mpc.resources(a)
                retina,fit={},{}
                try:
                    for k,Factory in MODES.items():
                        retina[k]=Factory(a,rig,wide)
                        fit[k]=DenseFlight(retina[k].period,p["settings"])
                    order=list(retina)
                    a.launch(tape["shot"],card=card)
                    a.partner.reset(seed=card["seed"])
                    fork=case["tick"]
                    start=max(0,fork-round(1.5/a.control_dt))
                    next_image=start*a.control_dt
                    stamps=[]
                    fit_cpu=Counter()
                    diffs={k:Counter() for k in order[1:]}
                    color={k:dict(squared=0,changed=0,total=0,max_abs=0) for k in order[1:]}
                    p_errors={k:[] for k in order}
                    measures=[]
                    original_contacts=a._contacts

                    def capture(now):
                        nonlocal next_image
                        offers={}
                        for m in np.roll(order,len(stamps)%len(order)):
                            m=str(m)
                            q,_,detail=retina[m].capture(now)
                            fit[m].ingest(now,q)
                            offers[m]=(q,detail)
                        stamps.append(float(now))
                        next_image+=1/120
                        # Evaluator-only truth and parity checks. Never used by
                        # a renderer's causal target or by a fit.
                        for k,(q,_) in offers.items():
                            if q is not None:
                                p_errors[k].append((q-a.ball).tolist())
                        ref,control=offers["original"]
                        for k,diff in diffs.items():
                            q,info=offers[k]
                            diff["centers_exact"]+=int(control is None and info is None or
                                  control is not None and info is not None and control["centers"]==info["centers"])
                            diff["optical_point_exact"]+=int(same(q,ref))
                            diff["optical_point_availability_same"]+=int((q is None)==(ref is None))
                            if control is None or info is None:
                                continue
                            for eye in ("left","right"):
                                x,y=control["frames"][eye],info["frames"][eye]
                                z=np.abs(x.astype(np.int16)-y.astype(np.int16))
                                color[k]["changed"]+=int(np.count_nonzero(z));color[k]["total"]+=z.size
                                color[k]["squared"]+=int(np.sum(z.astype(np.int32)**2))
                                color[k]["max_abs"]=max(color[k]["max_abs"],int(np.max(z)))
                                diff["crops"]+=1
                                old,new=native_ball_pixels(x),native_ball_pixels(y)
                                diff["detector_exact"]+=int(_sighting_equal(old,new))
                                diff["detector_availability_same"]+=int((old is None)==(new is None))
                                if old is not None and new is not None:
                                    diff["detector_matched"]+=1

                    def contacts(incoming):
                        original_contacts(incoming)
                        now=float(a.data.time)
                        if now>=next_image-1.e-8 and now>=start*a.control_dt and not (a.terminated or a.truncated):
                            capture(now)
                    a._contacts=contacts
                    until=min(len(tape["actions"]),fork+school.H+8)
                    for tick,u in enumerate(tape["actions"][:until]):
                        now=float(a.data.time)
                        if tick==start and not stamps:capture(now)
                        if tick>=start:
                            outcome={}
                            for m in np.roll(order,tick%len(order)):
                                m=str(m)
                                tm=process_time()
                                f=fit[m].fit(now)
                                fit_cpu[m]+=process_time()-tm
                                outcome[m]=packet(f)
                            if tick>=max(start,fork-30):
                                truth=dict(pv=np.r_[a.ball,a.velocity].tolist(),spin=a.spin.tolist())
                                measures.append(dict(tick=tick,time=now,truth=truth,estimates=outcome))
                        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
                        if done or cap:break
                    expected=[e for e in tape["events"] if e["time"]<=float(a.data.time)+1.e-8]
                    if school.digest_json(a.events)!=school.digest_json(expected):
                        raise ValueError("Packed raster changed the original physical prefix")
                    summary={}
                    for k,r in retina.items():
                        c=r.crop
                        summary[k]=dict(exposures=r.calls,wide=r.wide_calls,valid=r.valid,sensor_cpu_s=r.cpu,
                                        fit_cpu_s=float(fit_cpu[k]),crop_prepare_s=getattr(c,"prepare_cpu",None),
                                        crop_render_s=getattr(c,"render_cpu",None),
                                        submitted=getattr(c,"submitted",None),culled=getattr(c,"removed",None),
                                        point_errors=p_errors[k])
                    result["rows"].append(dict(case=case,clock=stamps,native_prefix_exact=True,
                          native_control_ticks=a.steps,native_seconds=float(a.data.time),
                          captured_seconds=len(stamps)/120,
                          sensors=summary,parity=diffs,frame_channels=color,measures=measures,
                          measured_all_cpu_s=process_time()-began))
                    result["complete"]=len(result["rows"])==len(p["rows"])
                    _write(OUT,result)
                    print(card["id"],len(stamps),[(m,x["valid"],round(1000*x["sensor_cpu_s"]/len(stamps),2))
                                               for m,x in summary.items()],diffs,flush=True)
                finally:
                    for r in retina.values():
                        r.crop.renderer._gl_context.make_current()
                        r.close()
                    wide.renderer._gl_context.make_current()
                    wide.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--n",type=int,default=6)
    p.add_argument("--variant",choices=("mono","packing"),default="mono")
    q=p.parse_args()
    if q.variant=="packing":
        PLAN,OUT=ROOT/"packing.protocol.json",ROOT/"packing.json.gz"
        MODES={k:FACTORIES[k] for k in ("original","packed","culled")}
    freeze() if q.mode=="plan" else run(q.n)

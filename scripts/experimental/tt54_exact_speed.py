"""Profile and benchmark parity on matched ordinary single-thread live training games."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt29_partner as opponent
from . import tt42_learning_course as course
from . import tt44_rally_course as fresh
from . import tt46_continuous_season as season
from .tt48_rim_recovery import STOCK
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt54_packed_stereo import PackedStereo
from .tt54_opponent_approx import approximate, compiled

ROOT=Path("artifacts/tt54_exact_speed")


class Telemetry:
    def __init__(self,whole):
        self.cpu=Counter()
        self.calls=Counter()
        self.rgb=sha256()
        self.whole=whole

    def timed(self,name,f, *, pixels=False):
        def call(*a,**kw):
            t=process_time()
            answer=f(*a,**kw)
            self.cpu[name]+=process_time()-t
            self.calls[name]+=1
            if pixels:
                t=process_time()
                for k in ("left","right"):
                    self.rgb.update(answer[k].tobytes())
                self.cpu["parity_hash"]+=process_time()-t
            return answer
        return call

    def bind(self,a,camera,actor,field):
        stack=ExitStack()
        funcs=[(a,"exchange","exchange"),(camera,"capture","stereo"),(mpc,"observed","optics"),
               (actor.engine,"propagate","body"),(actor.engine.predict,"trial","near_native"),
               (actor.priors,"priors","retrieve"),(field,"get","raw_field"),(mpc,"audit","audit"),
               (a.partner,"advance","opponent")]
        for obj,key,name in funcs:
            stack.enter_context(patch.object(obj,key,self.timed(name,getattr(obj,key),pixels=(name=="stereo" and self.whole))))
        return stack


def play(parts,a,c,cam,info,actor,raw,base, *, profiled, cap, fast_opponent=False):
    rig,stride,lo,hi=info
    clock=Telemetry(True)
    n=a.launches
    start=process_time()
    wall=perf_counter()
    with clock.bind(a,cam,actor,raw),ExitStack() as stack:
        law=approximate if fast_opponent else opponent._flight_approx
        stack.enter_context(patch.object(opponent,"_flight_approx",clock.timed("opponent_approx",law)))
        if profiled:
            for name in ("mjr_render","mjr_readPixels","mjv_updateScene","mj_step"):
                stack.enter_context(patch.object(mujoco,name,clock.timed(name,getattr(mujoco,name))))
            for name in ("_layer","_prepare"):
                if hasattr(cam,name):
                    stack.enter_context(patch.object(cam,name,clock.timed(name,getattr(cam,name))))
        row=mpc.play(a,cam,rig,stride,lo,hi,base["ready"],c,base["shot"],raw,actor,course.OPTIC,"live",cap)
    full_cpu=process_time()-start
    full_wall=perf_counter()-wall
    if a.launches!=n+1:
        raise ValueError("Expected one real launch")
    for leg, req in enumerate(row["placements"]["requested"]):
        if req["target"]!=list(draw(c,leg,"random",None)[0]):
            raise ValueError("Different wide opponent draw")
    return dict(parts=parts,cpu=dict(clock.cpu),calls=dict(clock.calls),cpu_whole_s=full_cpu,
                cpu_without_pixel_audit_s=full_cpu-clock.cpu["parity_hash"],
                wall_s=full_wall,simulated_s=float(a.data.time),row=row,rgb_sha256=clock.rgb.hexdigest())


def profile(out,order):
    archive=_read(STOCK)
    state=_read(season.CARRY)
    cards=[archive["cards"][i] for i in order]
    ready=archive["ready"]
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    result=dict(profile_clock="process CPU, one native worker",cards=cards,
                archive_sha256=_hash(STOCK),carry_sha256=_hash(season.CARRY),rows=[])
    out.parent.mkdir(parents=True,exist_ok=True)
    with mpc.centered(ready):
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        with patch.object(mpc,"Fragments",lambda b,e: Carried(b,e,practice=[course.PRACTICE],held={c["id"] for c in cards})):
            actor=season.Player(mpc.BANK,raw,variant=archive["variant"],budget=archive["budget"],
                                coverage=fresh.shot.Coverage(_read(fresh.RISK)))
        try:
            for i,c in zip(order,cards,strict=True):
                course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),state)
                row=play("normal",a,c,cam,(rig,stride,lo,hi),actor,raw,dict(ready=ready,shot=archive["rows"][i]["shot"]),
                         profiled=True,cap=archive["cap"])
                scored=row.pop("row")
                row.update(id=c["id"],learner=scored["score"]["learner"],tick=len(scored["actions"]),
                           action_sha256=digest_json(scored["actions"]),events_sha256=digest_json(scored["events"]))
                result["rows"].append(row)
                _write(out,result)
                print(c["id"],row,flush=True)
        finally:
            actor.close();cam.close();a.close()


def benchmark(out,order,fast_opponent):
    archive=_read(STOCK)
    state=_read(season.CARRY)
    cards=[archive["cards"][i] for i in order]
    ready=archive["ready"]
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    doc=dict(carry_sha256=_hash(season.CARRY),control_protocol_sha256=_hash(STOCK),
             source_sha256={p.name:_hash(p) for p in (Path(__file__),*(Path(__file__).with_name(name) for name in
                  ("tt54_packed_stereo.py","tt54_packed_stereo.cpp","tt54_opponent_approx.py","tt54_opponent_approx.cpp")))},
             fast_opponent=fast_opponent,
             clock="One physical CPU worker. Includes regular instrumentation but excludes measured extra RGB hashing time.",pairs=[],complete=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with mpc.centered(ready):
        a=mpc.arena()
        stock,rig,stride,lo,hi=mpc.resources(a)
        fast=PackedStereo(a)
        if fast.hardware is None or (fast_opponent and compiled() is None):
            fast.close();stock.close();a.close()
            raise ValueError(f"Optimized renderer unavailable: {fast.fallback}")
        doc["graphics"]=fast.hardware
        with patch.object(mpc,"Fragments",lambda b,e: Carried(b,e,practice=[course.PRACTICE],held={c["id"] for c in cards})):
            actor=season.Player(mpc.BANK,raw,variant=archive["variant"],budget=archive["budget"],
                                coverage=fresh.shot.Coverage(_read(fresh.RISK)))
        try:
            for j,(i,c) in enumerate(zip(order,cards,strict=True)):
                rows={}
                numeric={}
                moved={}
                tags=("stock","packed") if j%2==0 else ("packed","stock")
                for tag in tags:
                    course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),state)
                    camera=stock if tag=="stock" else fast
                    run=play(tag,a,c,camera,(rig,stride,lo,hi),actor,raw,dict(ready=ready,shot=archive["rows"][i]["shot"]),
                             profiled=False,cap=archive["cap"],fast_opponent=(tag=="packed" and fast_opponent))
                    checkpoint=out.with_name(f"{c['id']}-{tag}.npz")
                    saved=course.save(actor,checkpoint)
                    final=list(actor.priors.online)
                    numeric[tag]=[(u.copy(),v.copy(),t) for u,v,t in final]
                    moved[tag]=run.pop("row")
                    run.update(checkpoint_sha256=saved,checkpoint=checkpoint.name,
                               action_sha256=digest_json(moved[tag]["actions"]),events_sha256=digest_json(moved[tag]["events"]),
                               motor_digest=moved[tag]["memory"],scored=moved[tag]["score"]["learner"],
                               credited=moved[tag]["planner"].get("credited_fragments",0))
                    rows[tag]=run
                    print(c["id"], tag, run["scored"], round(run["cpu_without_pixel_audit_s"],3),
                          round(run["cpu"]["stereo"],3), run["rgb_sha256"], flush=True)
                for k in ("actions","events","score","placements","motor_audit","searches","memory"):
                    if digest_json(moved["stock"][k])!=digest_json(moved["packed"][k]):
                        raise ValueError(f"Different full training game {c['id']} {k}")
                if (rows["stock"]["rgb_sha256"]!=rows["packed"]["rgb_sha256"] or
                        rows["stock"]["calls"]["stereo"]!=rows["packed"]["calls"]["stereo"]):
                    raise ValueError(f"Actual ordinary RGB changed on {c['id']}")
                same=lambda a,b:np.array_equal(a.view(np.uint64),b.view(np.uint64))
                if any(t!=k or not same(a,x) or not same(b,y) for (a,b,t),(x,y,k) in zip(numeric["stock"],numeric["packed"],strict=True)):
                    raise ValueError(f"The actual retained motor bank changed on {c['id']}")
                doc["pairs"].append(dict(id=c["id"],order=tags,arms=rows,identical_rgb=True,identical_commands=True,
                                         identical_native_referee=True,identical_retained_bank=True,
                                         controls=moved["stock"]["actions"],events=moved["stock"]["events"]))
                doc["complete"]=len(doc["pairs"])==len(cards)
                _write(out,doc)
        finally:
            actor.close();fast.close();stock.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,nargs="+",default=[0])
    p.add_argument("--out",type=Path,default=ROOT/"profile.json")
    p.add_argument("--benchmark",action="store_true")
    p.add_argument("--fast-opponent",action="store_true")
    a=p.parse_args()
    benchmark(a.out,a.index,a.fast_opponent) if a.benchmark else profile(a.out,a.index)

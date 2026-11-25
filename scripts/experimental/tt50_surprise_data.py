"""Reconstruct emitted prefixes on real offers that lacked a legal native proposal."""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

ROOT = Path("artifacts/tt50_surprise")
MANIFEST = Path("artifacts/tt49_sustain/fit.json")
BASELINE = season.ROOT / "before.json.gz"
PLAN = ROOT / "plan.json"
PACKET = ROOT / "captured.json.gz"
STEPS = 5
PER_CLASS = {True: 64, False: 82}
RANGE = (-1.50, -.51)
OPENINGS = (7398, 15541, 14429, 10762)


def eligible(n):
    return (n.get("legal_trials") == 0 and len(n.get("p", ())) == 3
            and RANGE[0] < n["p"][1] < RANGE[1])


def occurrences(files, exclude):
    groups = defaultdict(dict)
    valid_sources = {}
    otherwise = defaultdict(set)
    for name, fingerprint in files.items():
        p = Path(name)
        if not p.exists() or _hash(p)!=fingerprint:
            raise ValueError("A compatible physical archive changed: "+name)
        d = _read(p)
        if not d.get("ready") or len(d["ready"])!=6:
            continue
        for i,(card,row) in enumerate(zip(d["cards"],d["rows"],strict=True)):
            if card["id"] in exclude or card.get("profile")!="reach":
                continue
            ps, req = row["placements"]["delivered"],row["placements"]["requested"]
            if any(x["draw"]!="random" for x in req):
                continue
            end = row["events"][-1]
            for j,o in enumerate(ps):
                good = bool(o["near_landed"])
                hit = o.get("near_contact")
                if not o["far_contact"] or not o["near_bounce"] or not good and end["kind"]!="point":
                    continue
                close = (hit or end)["step"]
                start = o["far_contact"]["step"]
                options = [x for x in row["searches"] if eligible(x) and start<x["tick"]<close
                           and x["tick"]+STEPS<=len(row["actions"])]
                if not options:
                    continue
                if good:
                    quick = [x for x in options if 1<=close-x["tick"]<=STEPS]
                    if quick:
                        otherwise[card["id"]].add((j,quick[-1]["tick"],
                           digest_json((o["near_bounce"],row["actions"][quick[-1]["tick"]:quick[-1]["tick"]+STEPS]))))
                    options = [x for x in options if 5<=close-x["tick"]<=21]
                    if not options:
                        continue
                n=min(options,key=lambda x:abs(x["p"][1]+1.04))
                step=n["tick"]
                if good:
                    if step >= close:
                        continue
                    reward = next(e for e in row["events"] if e["kind"]=="legal_return"
                                  and e["side"]==1 and e["time"]>hit["time"])
                else:
                    reward=end
                end_tick=int(reward["step"])
                # Identical physical input and issued prefix across mirrored archives is one example.
                key=digest_json((o["far_contact"],o["near_bounce"],row["actions"][:step+STEPS]))
                saved = groups[good]
                if key not in saved:
                    saved[key]=dict(id=card["id"], title=name, row=i, leg=j,
                                    decision=step,native_note=n,card=card,
                                    success=good, end=end_tick, impact_close=close,
                                    later_nominal_legal=any(step<x["tick"]<close and x.get("legal_trials",0)>0
                                                           for x in row["searches"]))
                    valid_sources[name]=fingerprint
    return groups, valid_sources, otherwise


def sample(items, n, seed):
    by_id=defaultdict(list)
    for r in sorted(items,key=lambda v:(v["decision"],v["title"],v["row"],v["leg"])):
        by_id[r["id"]].append(r)
    ids=np.random.default_rng(seed).permutation(sorted(by_id))
    # Maximize release diversity and use shorter exact physical prefixes first within a release.
    return [by_id[i][k] for k in range(2) for i in ids if k<len(by_id[i])][:n]


def plan():
    spec=_read(season.PLAN)
    excluded={c["id"] for g in ("practice","assessment") for c in spec[g]}
    src=_read(MANIFEST)["input_files"]
    pools, valid, tiny=occurrences(src,excluded)
    routes=[r for b in (True,False) for r in sample(pools[b].values(),PER_CLASS[b],500201+int(b))]
    held=_read(BASELINE)
    challenges=[]
    used=set()
    for i,(c,r) in enumerate(zip(held["cards"],held["rows"],strict=True)):
        ident=c["id"]
        if ident in used or (ident not in OPENINGS and
             (not r["score"]["learner"] or sum(a["role"]=="ordinary control" for a in challenges)>=4)):
            continue
        role="ordinary control" if ident not in OPENINGS else "weak opening"
        challenges.append(dict(i=i,id=ident,card=c,role=role))
        used.add(ident)
    counts={str(int(b)):dict(offers=len(pools[b]), releases=len({r["id"] for r in pools[b].values()}))
            for b in (True,False)}
    d=dict(source_manifest=_hash(MANIFEST),source_files=valid,training=routes,
           held_excluded=sorted(excluded),held_source_sha256=_hash(BASELINE),held=challenges,
           census=counts,late_literal=dict(cases=sum(map(len,tiny.values())),releases=len(tiny)),
           causal_rules="First five ordinary issued rates at a selected no-legal note before the contact. The current and last four sensory rows are available at decision. Selection labels are actual score or terminal fault. Later valid searches may recover after this preparatory prefix.")
    _write(PLAN,d)
    print(counts,"selected", [(b,sum(x["success"]==b for x in routes)) for b in [True,False]],
        "literal",d["late_literal"],"held",[(r["id"],r["role"]) for r in challenges],flush=True)


def key(history):
    """A shared causal descriptor: current optics, local flight/limb change, encoders and reference error."""
    if len(history)<5:
        return None
    last, first = history[-1],history[-5]
    if last[0]-first[0]!=4 or any(x is None or x[4]>=-.9 for _,x,*_ in list(history)[-5:]):
        return None
    _,x,q,ref=last
    _,o,pre,_=first
    return np.r_[x,(x[:3]-o[:3]),x[5]-o[5],q[:5],ref[:5]-q[:5],q[:5]-pre[:5]]


def capture(output):
    p = _read(PLAN)
    d=_read(output) if output.exists() else dict(plan_sha256=_hash(PLAN),rows=[], complete=False)
    if d["plan_sha256"]!=_hash(PLAN):
        raise ValueError("Capture does not match the original schedule")
    done={r["index"] for r in d["rows"]}
    caches, ready= {}, defaultdict(lambda:defaultdict(list))
    for k,r in enumerate(p["training"]):
        if k in done:
            continue
        path=r["title"]
        if path not in caches:
            caches[path]=_read(Path(path))
            if _hash(Path(path))!=p["source_files"][path]:
                raise ValueError("A training receipt changed")
        ready[tuple(caches[path]["ready"])][(path,r["row"])].append((k,r))
    for q0,jobs in ready.items():
        with mpc.centered(q0):
            a=mpc.arena()
            res=mpc.resources(a)
            try:
                cam,rig,stride,lo,hi=res
                for (src,j),targets in jobs.items():
                    row=caches[src]["rows"][j]
                    card=targets[0][1]["card"]
                    a.launch(row["shot"],card=card)
                    a.partner.reset(seed=card["seed"])
                    eye=mpc.Vision(rig,a.control_dt,raw=True)
                    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
                    recent=deque(maxlen=7); ref=np.asarray(q0).copy()
                    select={r["decision"]:(k,r) for k,r in targets}
                    limit=max(max(r["end"],r["decision"]+STEPS) for k,r in targets)
                    records=[]
                    for t,u in enumerate(row["actions"][:limit]):
                        allow=any(h-18<=t<=h for h in select)
                        if allow:
                            q=a.joints
                            x=mpc.observed(eye,t,cam.capture() if t%2==0 else None,q)
                            recent.append((t,x,q,ref.copy()))
                        if t in select:
                            k,r=select[t]
                            z=key(recent)
                            if z is None or np.linalg.norm(z[:3]-np.asarray(r["native_note"]["p"]))>.047:
                                records.append(dict(index=k,missing="camera gate or archived optical disagreement"))
                            else:
                                motion=np.asarray(row["actions"][t:t+STEPS])
                                if motion.shape!=(STEPS,6) or np.max(abs(motion))>.8+1e-9:
                                    records.append(dict(index=k,missing="different command governor"))
                                else:
                                    records.append(dict(index=k,z=z.tolist(),actions=motion.tolist()))
                        ref=np.clip(ref+stride*np.asarray(u),lo,hi)
                        a.exchange(u,mpc.FAR_COMMAND)
                    stable=[e for e in row["events"] if e["time"]<=a.data.time+1e-8]
                    if digest_json(a.events)!=digest_json(stable):
                        records=[dict(index=k,missing="native event replay diverged") for k,r in targets]
                    else:
                        for v in records:
                            v["physical_prefix_sha256"]=digest_json(a.events)
                    d["rows"].extend(records)
                    d["complete"]=len(d["rows"])==len(p["training"])
                    _write(output,d)
                    if len(d["rows"])==len(records) or len(d["rows"])%10<=len(records):
                        print(len(d["rows"]),len(p["training"]),sum("actions" in v for v in d["rows"]),flush=True)
            finally:
                res[0].close()
                a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan","capture"))
    p.add_argument("--output",type=Path,default=PACKET)
    a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    plan() if a.phase=="plan" else capture(a.output)

"""Native common-random-draw counterfactuals for two achieved strokes from one state."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt37_opponent as live
from . import tt46_continuous_season as season
from . import tt48_practice_transfer as official
from . import tt51_afterstate as afterstate
from . import tt51_afterstate_native as native
from . import tt51_transfer_after_point as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import BallFlight
from .tt44_paired_credit import State
from .tt54_training_speedups import enabled


ROOT = Path("artifacts/tt55_paired_afterstate")
PLAN = ROOT / "protocol.json"
OUT = {k: ROOT / f"{k}.json.gz" for k in ("held", "train")}
PAIRED = transfer.ROOT / "transfer.json"
SOURCES = [transfer.ROOT / f"{k}-held.json.gz" for k in ("before", "after")]
EXTRA_EXCLUDES = {14843, 4322, 8302, 11846, 4214, 4309, 4636}
N_DRAWS = 6
N_TRAIN = 32
course, mpc = season.course, season.course.mpc


def stroke(row, leg, decision):
    requests = row["placements"]["requested"]
    if leg + 1 >= len(requests):
        return None
    o = row["placements"]["delivered"][leg]
    if not o["near_landed"] or o["near_contact"] is None or o["near_contact"]["time"] + 1e-8 < decision * mpc.DT:
        return None
    try:
        depart, _, _, _ = afterstate.episode(row, leg + 1)
    except (StopIteration, ValueError):
        return None
    hit = o["near_contact"]["time"]
    land = next((e for e in row["events"] if e["kind"] == "legal_return" and e["side"] == 1 and e["time"] > hit), None)
    if land is None:
        return None
    query = next((s for s in row["searches"] if s["tick"] == decision and s.get("chosen")
                  and len(s.get("target_errors", [])) == 7 and np.isfinite(s["target_errors"]).all()), None)
    anchor = int(np.floor(requests[leg + 1]["time_s"] / mpc.DT + 1e-7))
    return dict(anchor=anchor, depart=depart, landing=land, decision=query,
                leg=leg + 1, original_seed=row["placements"]["card"]["reply_seed"])


def pair(a, b, card, refs, *, expected=None):
    if a["shot"] != b["shot"]:
        return None
    k = next((t for t, (x, y) in enumerate(zip(a["actions"], b["actions"])) if x != y), None)
    if k is None or (expected is not None and k != expected["first_command"]):
        return None
    pre = [[e for e in r["events"] if e["step"] <= k] for r in (a, b)]
    if digest_json(pre[0]) != digest_json(pre[1]):
        return None
    options = [r["placements"]["delivered"] for r in (a,b)]
    eligible = [i for i, (x, y) in enumerate(zip(*options)) if max(x["far_contact"]["time"], y["far_contact"]["time"]) <= k*mpc.DT+1e-8]
    if not eligible:
        return None
    leg = eligible[-1]
    if expected is not None and leg != expected["first_offer"]:
        return None
    o, p = (v[leg] for v in options)
    if any(o[z] != p[z] for z in ("far_contact", "near_bounce", "receive_line", "target")):
        return None
    info = [stroke(r, leg, k) for r in (a,b)]
    if not all(info) or info[0]["depart"] == info[1]["depart"]:
        return None
    arms = [dict(v, path=str(f), row=i) for v, (f,i) in zip(info, refs, strict=True)]
    return dict(id=card["id"], card=card, fork=k, incoming=o["near_bounce"], leg=leg,
                full_prefix_sha256=digest_json(pre[0]), arms=arms)


def random_game(card, row):
    return card.get("profile") == "reach" and card.get("opening") == "random" and row.get("searches") and all(
        r["draw"] == "random" and r["target"] == list(live.draw(card, i, "random", None)[0])
        for i,r in enumerate(row["placements"]["requested"]))


def freeze():
    labels = _read(PAIRED)
    docs = [_read(p) for p in SOURCES]
    if labels["sources"] != [_hash(p) for p in SOURCES] or docs[0]["cards"] != docs[1]["cards"]:
        raise ValueError("The predeclared TT52 archive changed")
    hold = []
    for i, q in enumerate(labels["points"]):
        if q.get("first_legal") != [True, True] or not q.get("same_ball"):
            continue
        card = docs[0]["cards"][i]
        u,v = [d["rows"][i] for d in docs]
        p = pair(u, v, card, [(p,i) for p in SOURCES], expected=q)
        if p is None or not all(random_game(card,r) for r in (u,v)):
            raise ValueError(f"Anchored two-landed pair no longer matches {card['id']}")
        hold.append(dict(p, ready=docs[0]["ready"]))
    if len(hold) != 17:
        raise ValueError("Expected all 17 exact two-landed initial TT52 forks")
    excluded = set(EXTRA_EXCLUDES) | {c["id"] for d in (_read(season.PLAN), _read(official.PLAN))
                                  for k in ("practice", "assessment") for c in d.get(k, [])}
    excluded |= {r["id"] for r in hold}
    expected = _read(afterstate.INPUTS)["input_files"]
    groups, used, versions = defaultdict(list), {}, {}
    for name, sha in sorted(expected.items()):
        path = Path(name)
        if not path.exists() or _hash(path) != sha:
            raise ValueError(f"Changed previously mined ordinary source {path}")
        old = _read(path)
        if not old.get("ready"):
            continue
        versions[path] = sha
        for i,(card,r) in enumerate(zip(old["cards"], old["rows"], strict=True)):
            if card["id"] in excluded or not random_game(card, r):
                continue
            key = digest_json([card, r["shot"], old["ready"]])
            fp = digest_json(r["actions"])
            if fp not in used.setdefault(key, set()):
                used[key].add(fp)
                groups[key].append((path,i,card,r,old["ready"]))
    by_release = defaultdict(dict)
    for variants in groups.values():
        for j,(f,i,card,r,ready) in enumerate(variants):
            for g,k,_,s,_ in variants[j+1:]:
                p = pair(r,s,card,[(f,i),(g,k)])
                if p is None or p["leg"] > 3:
                    continue
                key = tuple(sorted(digest_json(a["depart"]) for a in p["arms"]))
                by_release[p["id"]][key] = dict(p,ready=ready)
    rng = np.random.default_rng(550684)
    train = []
    for ident in rng.permutation(sorted(by_release))[:N_TRAIN]:
        candidates = list(by_release[int(ident)].values())
        # Prioritize two decision-time candidate predictions, without using any reply outcome.
        opts = [p for p in candidates if all(v["decision"] is not None for v in p["arms"])] or candidates
        train.append(opts[int(rng.integers(len(opts)))])
    if len(train) <= len(hold):
        raise ValueError(f"Only {len(train)} training releases outside all reserved IDs")
    seen = {p["original_seed"] for r in train+hold for p in r["arms"]}
    for i,r in enumerate(hold+train):
        seeds = []
        while len(seeds) < N_DRAWS:
            s = int(rng.integers(1, 2**30))
            if s not in seen:
                seeds.append(s)
                seen.add(s)
        r.update(pair_index=i, goals=seeds)
    files = {p:_hash(Path(p)) for p in sorted({v["path"] for r in hold+train for v in r["arms"]})}
    assets = [PAIRED, afterstate.INPUTS, season.CARRY, season.CARRY.with_suffix("").with_suffix(".npz"),
              season.PLAN, official.PLAN, course.OPTIC, mpc.BANK, course.PRACTICE,
              season.fresh.RISK, season.fresh.servo.ROOT / "prior.npz"]
    code = [__file__, mpc.__file__, native.__file__, afterstate.__file__, season.__file__, live.__file__,
            Path(__file__).with_name("tt54_packed_stereo.py")]
    result = dict(held=hold, train=train, files=files, excluded_training=sorted(excluded),
                  assets={str(p):_hash(p) for p in assets}, source={Path(p).name:_hash(Path(p)) for p in code},
                  candidate_pairs=len(by_release), draws=N_DRAWS,
                  rule="All 17 source-anchored physical two-landed pairs and one predeclared seeded pair per "
                       "independent training release. Same physical prefix and incoming event; each achieved stroke "
                       "is replayed with six identical new native requests. The fixed receiver gets only RGB/encoders.")
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if _read(PLAN) != result:
            raise ValueError("Do not refreeze the existing assignment")
    else:
        _write(PLAN,result)
    print("pairs",len(hold),len(train),"available",len(by_release),
          "causal features",[sum(all(a["decision"] for a in r["arms"]) for r in cohort) for cohort in (hold,train)],flush=True)


def base_goal(raw, home, inbound, x, q):
    if x is not None and (x[4] < -.9 or (inbound and x[1] < .50)):
        g, alpha = raw.get(x,q)
        if g is not None:
            return home + alpha*(g-home)
    return home


class OpticalState:
    def __init__(self, eye, engine):
        self.eye = deepcopy(eye)
        self.body = mujoco.MjData(engine.body_model)
        mujoco.mj_copyData(self.body, engine.body_model, engine.body_state)
        self.force = engine.body_model.actuator_forcerange.copy()
        self.fields = {k:deepcopy(getattr(engine,k)) for k in ("previous","rate","latest","tick","reference","bounced")}

    def install(self, actor):
        actor.begin()
        engine = actor.engine
        eye = deepcopy(self.eye)
        mujoco.mj_copyData(engine.body_state,engine.body_model,self.body)
        engine.body_model.actuator_forcerange[:] = self.force
        for k,v in self.fields.items():
            setattr(engine,k,deepcopy(v))
        return eye


def prepare(a, resources, actor, card, row, arm, ready):
    image,rig,stride,lo,hi = resources
    a.launch(row["shot"],card=card)
    a.partner.reset(seed=card["seed"])
    eye = mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye,course.OPTIC)
    actor.begin()
    ref = np.array(ready)
    inbound = False
    for t,u in enumerate(row["actions"][:arm["anchor"]]):
        q = a.joints
        x = mpc.observed(eye,t,image.capture() if t%2==0 else None,q)
        inbound |= x is not None and x[4]<-.9 and x[1]<-.9
        actor.engine.observe(t,eye.eye,q,ref.copy())
        a.exchange(u,mpc.FAR_COMMAND)
        ref = np.clip(ref+stride*u,lo,hi)
    expected = [e for e in row["events"] if e["time"] <= a.data.time+1e-8]
    if digest_json(expected)!=digest_json(a.events) or len(a.partner.requests)!=arm["leg"]:
        raise ValueError("Pre-request control tape does not reproduce native state")
    return State(a), OpticalState(eye,actor.engine), ref, inbound, len(expected), digest_json(expected)


def line_info(a, flight, offer):
    if offer is None:
        return None,None,"unanswered"
    line=offer["receive_line"]
    if line is not None:
        return line,afterstate.projected(np.r_[line["xyz"],line["velocity"]]).tolist(),"measured"
    # An earlier real contact can prevent measuring y=-1.10. Only then extrapolate
    # from the actual preceding table separation in an isolated evaluator solver.
    if offer["near_contact"] is None:
        return None,None,"unobserved"
    bounce,hit=offer["near_bounce"]["time"],offer["near_contact"]["time"]
    e=next((e for e in a.events if e.get("collider")=="table" and bounce<e["time"]<hit and e["velocity"][1]<-.9),None)
    if e is None:
        return None,None,"unobserved"
    path=flight.path(e["ball"],e["velocity"],e["spin"],1,True,horizon=.42)
    for left,right in zip(path,path[1:]):
        if left[2]>-1.10>=right[2]:
            w=(-1.10-left[2])/(right[2]-left[2])
            x=left[1:7]+w*(right[1:7]-left[1:7])
            q=dict(xyz=x[:3].tolist(),velocity=x[3:6].tolist())
            return q,afterstate.projected(x).tolist(),"extrapolated"
    return None,None,"unobserved"


def replay(a, resources, actor, flight, saved, pair, row, arm, seed, ready):
    state,sensors,old_ref,inbound,events_n,prefix = saved
    state.restore(a)
    eye = sensors.install(actor)
    ref,home = old_ref.copy(),np.array(ready)
    image,_,stride,lo,hi=resources
    k,begin=arm["leg"],arm["anchor"]
    until=arm["depart"]["step"]+2
    law=live.draw
    controls=[]
    def response(c,j,mode,rng):
        if j==k and seed is not None:
            c=dict(c,reply_seed=seed)
        return law(c,j,mode,rng)
    with patch.object(live,"draw",response):
        for t in range(begin,begin+245):
            q=a.joints
            x=mpc.observed(eye,t,image.capture() if t%2==0 else None,q)
            inbound |= x is not None and x[4]<-.9 and x[1]<-.9
            if t<until:
                u=np.array(row["actions"][t])
                actor.engine.observe(t,eye.eye,q,ref.copy())
            else:
                goal=actor.step(t,eye.eye,x,q,ref,base_goal(actor.raw,home,inbound,x,q))
                u=np.clip((np.clip(goal,lo,hi)-ref)/stride,-.8,.8)
            controls.append(u.tolist())
            ref=np.clip(ref+u*stride,lo,hi)
            _,_,done,capped,info=a.exchange(u,mpc.FAR_COMMAND)
            if done or capped or (t>until and a.legal_returns>=k+2):
                break
    p=a.point_placements()
    if len(p["requested"])<=k or any(v["target"]!=list(response(pair["card"],j,"random",None)[0])
                                      for j,v in enumerate(p["requested"])):
        raise ValueError("Native far paddle did not respond to the common randomized requested goals")
    match=lambda event:any(digest_json(event)==digest_json(e) for e in a.events)
    if not match(arm["depart"]) or not match(arm["landing"]):
        # Mirrors the prior native verification tolerance after MjData restore.
        if not all(any(native.agrees(event,e) for e in a.events) for event in (arm["depart"],arm["landing"])):
            raise ValueError("A redraw or receiver changed the physically achieved prior stroke")
    offer=p["delivered"][k] if len(p["delivered"])>k else None
    line, projected, line_kind=line_info(a,flight,offer)
    score=audit(a.events,opening_side=1)
    near=bool(offer and offer["near_landed"])
    if near != (score["learner"]>k):
        raise ValueError("Native referee and next-receive placement disagree")
    exhausted=not (done or capped or near)
    if exhausted:
        raise ValueError("The reply exceeded the fixed physical observation window")
    return dict(seed=seed,goal=p["requested"][k]["target"],answered=offer is not None,
                landed=near,touched=bool(offer and offer["near_contact"]),offer=offer,
                line=line,geometry=projected,line_kind=line_kind,
                central_low=None if projected is None else abs(projected[1])<.10 and projected[0]<1.00,
                prior_landing_verified=True,pre_request_sha256=prefix,
                native_legal=score["learner"],last_event=a.events[-1],
                actions=controls,events=a.events[events_n:],receiver_searches=actor.notes)


def run(cohort):
    plan=_read(PLAN)
    for name,h in {**plan["files"],**plan["assets"]}.items():
        if _hash(Path(name))!=h:
            raise ValueError(f"Changed source data {name}")
    if plan["source"][Path(__file__).name]!=_hash(Path(__file__)):
        raise ValueError("Changed paired assay implementation")
    path=OUT[cohort]
    receipt=_read(path) if path.exists() else dict(plan_sha256=_hash(PLAN),cohort=cohort,rows=[],complete=False)
    if receipt["plan_sha256"]!=_hash(PLAN):
        raise ValueError("Changed paired assignment")
    stock=_read(season.CARRY)
    cache={}; arena=None; previous=None
    with np.load(mpc.BANK) as z:
        raw=mpc.DeepField(z,"local")
    rank=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    try:
        with enabled(cameras=True,opponent=False):
            for pair in plan[cohort][len(receipt["rows"]):]:
                ready=pair["ready"]
                if arena is None or ready!=previous:
                    if arena is not None:
                        resources[0].close(); actor.close(); arena.close()
                    with mpc.centered(ready):
                        arena=mpc.arena(); resources=mpc.resources(arena)
                        with patch.object(mpc,"Fragments",lambda f,e:course.Carried(f,e,practice=[course.PRACTICE],held=set())):
                            actor=season.Player(mpc.BANK,raw,variant="elastic",budget=28,coverage=rank)
                        course.resume(actor,season.CARRY.with_suffix("").with_suffix(".npz"),stock)
                        flight=BallFlight(arena)
                    previous=ready
                results=[]
                with mpc.centered(ready):
                    for arm in pair["arms"]:
                        f=arm["path"]
                        if f not in cache:
                            cache[f]=_read(Path(f))
                        old=cache[f]["rows"][arm["row"]]
                        state=prepare(arena,resources,actor,pair["card"],old,arm,ready)
                        draws=[replay(arena,resources,actor,flight,state,pair,old,arm,s,ready) for s in pair["goals"]]
                        results.append(draws)
                if any(l["goal"]!=r["goal"] for l,r in zip(*results,strict=True)):
                    raise ValueError("A paired physical request was not the same across the two actions")
                receipt["rows"].append(dict(pair_index=pair["pair_index"],id=pair["id"],branches=results))
                receipt["complete"]=len(receipt["rows"])==len(plan[cohort])
                _write(path,receipt)
                print(cohort,len(receipt["rows"]),pair["id"],
                      "answered",[sum(v["answered"] for v in rs) for rs in results],
                      "near",[sum(v["landed"] for v in rs) for rs in results],
                      "central_low",[sum(v["central_low"] is True for v in rs) for rs in results],flush=True)
    finally:
        if arena is not None:
            resources[0].close();actor.close();arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","held","train"))
    a=p.parse_args()
    freeze() if a.phase=="plan" else run(a.phase)

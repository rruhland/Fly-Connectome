"""Prospective, paired continuous live credit on the exact TT93 wrist fixture."""
import argparse
from pathlib import Path
from time import process_time

import numpy as np

from . import tt93_wrist_games as game
from . import tt93_open_wrist_fixture as mount
from . import tt93_wrist_course as optical
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_curriculum import card,opening_candidates
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance
from .tt40_long_low import same_trace
from .tt4_transfer_trial import _hash,_read,_write

ROOT=mount.ROOT/"live"
PLAN=ROOT/"protocol.json"
SUMMARY=ROOT/"comparison.json"
PARITY=ROOT/"parity.json"
FEEDS=optical.FEEDS
SEED=93062617
N=24
MODES=("stereo","fused")
mpc,school=optical.mpc,optical.school
goal_count=512
source_dir=Path(__file__).parent


def filename(tag):
    return ROOT/(tag+".json.gz")


def scope_inputs():
    carry=optical.season.CARRY
    return (FEEDS,mount.MODEL,mount.ROOT/"bore.json",optical.PLAN,optical.SCOUT,optical.HOLD,optical.REPORT,
            game.OUT,game.SUMMARY,carry,carry.with_suffix("").with_suffix(".npz"),mpc.BANK,
            mpc.BANK.with_suffix(".json"),optical.course.OPTIC,optical.course.PRACTICE,
            optical.course.PRACTICE.with_suffix(".json"),optical.season.fresh.RISK,
            source_dir/"tt38_joint_mpc.py",source_dir/"tt93_wrist_games.py",source_dir/"tt93_wrist_course.py",
            source_dir/"tt93_open_wrist_fixture.py",source_dir/"tt54_training_speedups.py")


def blocked_ids():
    prior=optical.mount.old_look.wrist
    original_stages=(prior.PLAN,optical.mount.old_look.PLAN,optical.PLAN,optical.SCOUT,optical.HOLD,game.OUT,
        mount.ROOT/"cone-pilot-partial-scouts.json.gz",optical.season.CARRY,
        mpc.BANK.with_suffix(".json"),optical.course.PRACTICE.with_suffix(".json"))
    busy=[Path("artifacts")/x for x in (
        "tt80_continuous_games/protocol.json",
        "tt84_installation_system/season.protocol.json",
        "tt84_installation_system/games.protocol.json",
        "tt90_installation_learning/protocol.json",
        "tt90_installation_learning/hard.protocol.json",
        "tt97_return_to_center/protocol.json",
        "tt99_protected_learning/protocol.json",
        "tt83_population_search/protocol.json",
        "tt75/protocol.json")]
    # A prior protocol alone need not imply sensory exposure. Distinguish
    # the unused 110-card TT93 source reservoir so it remains eligible if
    # its native scout, camera and game branches were never opened.
    chosen=set()
    busy_docs={}
    for f in (*original_stages,*busy):
        if not f.exists():
            continue
        d=_read(f)
        if f==optical.PLAN:
            continue
        chosen.update(x for x in prior.ids(d))
        if f in busy:
            busy_docs[str(f)]=_hash(f)
    return chosen,busy_docs


def targets(c,n=goal_count):
    return [list(draw(c,i,"random",None)[0]) for i in range(n)]


def freeze():
    if PLAN.exists():
        design()
        return
    _=optical.guaranteed()
    witness=_read(game.OUT)
    if not witness["complete"] or not _read(optical.REPORT)["admit_prospective_fusion"]:
        raise ValueError("Missing physical and optical build custody")
    held,reservations=blocked_ids()
    shots=mpc.measured_launches(FEEDS)
    admissible=sorted({int(r["id"]) for r in opening_candidates(shots)}-held)
    if len(admissible)<N:
        raise ValueError(f"Only {len(admissible)} unused measured launches available")
    rng=np.random.default_rng(SEED)
    # No practice ceiling, neutral run, delivery stratification, early
    # reject, grip trial, or result-dependent card selection.
    order=rng.permutation(admissible)[:N].tolist()
    rows=[]
    for i,num in enumerate(order):
        c=dict(card(int(num),"full",i,int(rng.integers(1,2**30))),
               profile="reach",opening="random",reply_seed=int(rng.integers(1,2**30)))
        rows.append(dict(card=c,shot=shots[num],goals=digest_json(targets(c)),
                         arm_order=list(MODES if i%2==0 else MODES[::-1])))
    ROOT.mkdir(exist_ok=True,parents=True)
    d=dict(schema="tt93-live-matched-open-wrist-1",source={str(k):_hash(k) for k in scope_inputs()},
           reserved_schedules=reservations,fixture_sha256=_hash(mount.MODEL),home=_read(optical.PLAN)["home"],
           withheld=sorted(held),admissible_count=len(admissible),cards=rows,goal_prefix=goal_count,
           qualification=100,arms=MODES,policy="Ordinary external RGB controller vs same full physical rig and live direct-joint policy with the frozen 62% calibrated TT93 wrist bearing correction. Both persist only earned RGB/encoder/rated-action fragments from their own actual returns. Same initial carry, complete unfiltered measured launches in random order, exact same independent TT37 target sequence for each matched pair. Whole bounds and behavior after any 100 milestone maintained. No timing/renderer regime change, dense optical optimization, canted geometry, or adaptive sensor model.")
    _write(PLAN,d)
    print("TT93 live roster",len(admissible),order,flush=True)


def design():
    d=_read(PLAN)
    if any(_hash(Path(f))!=sig for f,sig in d["source"].items()):
        raise ValueError("Frozen learner, optics, physical fixture or qualification data changed")
    ids={r["card"]["id"] for r in d["cards"]}
    if ids & set(d["withheld"]) or len(ids)!=len(d["cards"]) or len(ids)!=N:
        raise ValueError("Chromatic or motor practice release leaked into full-game course")
    return d


def border(ctl,tag):
    r=dict(plan_sha256=_hash(PLAN),tag=tag,rows=[],complete=False)
    output=filename(tag)
    if not output.exists():
        return r
    p=_read(output)
    if p["plan_sha256"]!=r["plan_sha256"] or p["tag"]!=tag:
        raise ValueError("Cannot combine memories from different schedules")
    expert_serial=ctl.serial
    if p["rows"]:
        document=dict(p,rows=[None]*expert_serial+p["rows"])
        optical.course.resume(ctl,ROOT/p["checkpoint"],document)
    return p


def append(ctl,doc,row):
    old=doc.get("checkpoint")
    count=len(doc["rows"])+1
    tag=doc["tag"]
    path=ROOT/f"{tag}-step-{count:02}.npz"
    h=optical.course.save(ctl,path)
    doc["rows"].append(row)
    doc.update(checkpoint=path.name,checkpoint_sha256=h,complete=count==N)
    _write(filename(tag),doc)
    # Preserve a recoverable old bank until the receipt update became atomic.
    if old and int(Path(old).stem.rsplit("-",1)[-1])%6:
        (ROOT/old).unlink(missing_ok=True)


def parity():
    """Two real short archive points prove defaults preserve the qualified TT93 behavior."""
    d=design()
    _,mounting=optical.guaranteed()
    src=next(r for r in _read(game.OUT)["rows"] if r["card"]["id"]==7715)
    home,raw,rank,cp=optical.act_setup()
    rows={}
    with mount.builder(mounting),mpc.centered(home),school.enabled():
        a=mpc.arena();res=mpc.resources(a)
        try:
            for mode in MODES:
                ctl=optical.actor_for(raw,rank,cp,{src["card"]["id"]})
                try:
                    old=src["arms"]["same-installation-wrist-fusion" if mode=="fused" else "same-installation-stereo"]
                    got=game.play(a,res,ctl,home,src,mode=="fused")
                    exact=all(digest_json(got[n])==digest_json(old[n]) for n in ("actions","events","placements","corrections"))
                    if not exact:
                        raise ValueError(f"Game mechanics/frozen optics parity failed in {mode}")
                    rows[mode]=dict(action_sha256=digest_json(got["actions"]),score=got["score"]["learner"],
                                    frames=len(got["actions"]),exact=exact)
                finally:ctl.close()
        finally:res[0].close();a.close()
    _write(PARITY,dict(frozen_source=_hash(game.OUT),plan=_hash(PLAN),release=src["card"]["id"],arms=rows))
    print("TT93 default parity",[(k,r["score"],r["frames"]) for k,r in rows.items()],flush=True)


def proof_summary(proof,guard,entry,row,prefix):
    actual=game.verify(proof,guard,entry,row)
    legal=audit(row["events"],opening_side=1)
    requests=row["placements"]["requested"]
    if (legal!=row["score"] or any(req["target"]!=goal for req,goal in zip(requests,targets(entry["card"],len(requests)),strict=True))
            or digest_json(targets(entry["card"]))!=entry["goals"]):
        raise ValueError("Native tally or full opponent stream differs")
    if prefix and legal["learner"]<100:
        raise ValueError("An ordinary replay did not reach one hundred legal arm returns")
    return actual


def hundred_receipt(tag,item,on):
    def keep(tree):
        path=ROOT/f"hundred-{tag}-{item['card']['id']}-prefix.json.gz"
        _write(path,dict(tag=tag,entry=item,body_sha256=_hash(mount.MODEL),live_trace=tree,
                         independent_replay_pending=True,preassigned_index=on))
    return keep


def run():
    p=design()
    if not PARITY.exists() or _read(PARITY)["plan"]!=_hash(PLAN):
        raise ValueError("Reproduce actual qualified optics and command tapes before continuing live")
    _,mounting=optical.guaranteed()
    home,raw,rank,cp=optical.act_setup()
    chosen={r["card"]["id"] for r in p["cards"]}
    with mount.builder(mounting),mpc.centered(home),school.enabled():
        a=mpc.arena();proof=mpc.arena(width=64,height=48);cad=mpc.arena(width=64,height=48,detail="cad")
        res=mpc.resources(a);guard=CourtClearance(cad.model,articulated=True)
        actor={tag:optical.actor_for(raw,rank,cp,chosen) for tag in MODES}
        doc={tag:border(actor[tag],tag) for tag in MODES}
        try:
            for j,item in enumerate(p["cards"]):
                for tag in item["arm_order"]:
                    ctx=doc[tag]; ctl=actor[tag]
                    if len(ctx["rows"])>j:continue
                    if len(ctx["rows"])!=j:
                        raise ValueError("Memory did not carry through every assigned earlier point")
                    print("TT93 live start",j+1,tag,item["card"]["id"],ctl.priors.digest(),flush=True)
                    r=game.play(a,res,ctl,home,item,tag=="fused",live=True,progress=12,
                                on_hundred=hundred_receipt(tag,item,j))
                    started=process_time()
                    r["cad"]=proof_summary(proof,guard,item,r,False)
                    r["cad_cpu_s"]=process_time()-started
                    r["id"]=item["card"]["id"];r["assignment"]=j;append(ctl,ctx,r)
                    print("TT93 live done",j+1,tag,r["id"],r["score"]["learner"],
                          r["complete_point"],r["planner"].get("updates",0),len(r["corrections"]),
                          r["cad"]["global_cad"]["penetration_samples"],flush=True)
        finally:
            for c in actor.values():c.close()
            res[0].close();proof.close();cad.close();a.close()


def inspect_hundred():
    """Re-run every recorded 100 prefix with renderer-free native TT37 and full CAD."""
    p=design();_,mounting=optical.guaranteed()
    todo=list(ROOT.glob("hundred-*-prefix.json.gz"))
    if not todo:return
    with mount.builder(mounting),mpc.centered(p["home"]),school.enabled(cameras=False):
        a=mpc.arena(width=64,height=48);c=mpc.arena(width=64,height=48,detail="cad")
        guard=CourtClearance(c.model,articulated=True)
        try:
            for f in sorted(todo):
                w=_read(f); row=w["live_trace"]; item=w["entry"]
                if p["cards"][w["preassigned_index"]]!=item or w["body_sha256"]!=_hash(mount.MODEL):
                    raise ValueError("Not an ordinary preassigned current-fixture witness")
                audited=proof_summary(a,guard,item,row,True)
                positive=audited["global_cad"]["penetration_samples"]==0 and not audited["local_module_penetrations"]
                out=f.with_name(f.name.replace("-prefix.json.gz","-witness.json"))
                _write(out,dict(assigned_id=item["card"]["id"],tag=w["tag"],score=row["score"],
                     prefix_sha256=_hash(f),fixture_sha256=_hash(mount.MODEL),exact_event_sha256=digest_json(row["events"]),
                     safe=positive,cad=audited,receipt="Renderer-free independent native physics replay. Counting only the ordinary referee's legal physical near returns within this continuous unmodified opponent point."))
                if not positive:raise ValueError(f"100 prefix did not independently clear CAD {out}")
        finally:a.close();c.close()


def summarize():
    p=design()
    existing={f:_read(filename(f)) if filename(f).exists() else dict(rows=[]) for f in MODES}
    count=min(len(r["rows"]) for r in existing.values()); rows=[]
    for i,r in enumerate(p["cards"][:count]):
        before,after=(existing[k]["rows"][i] for k in MODES)
        summary=dict(id=r["card"]["id"],paired=index_pair(before,after),
                     terminal=[before["complete_point"],after["complete_point"]],
                     updates=[int(s["planner"].get("updates",0)) for s in (before,after)],
                     wrist=len(after["corrections"]),
                     cad=[s["cad"]["global_cad"]["samples"] for s in (before,after)],
                     global_penetrations=[s["cad"]["global_cad"]["penetration_samples"] for s in (before,after)],
                     local_penetrations=[len(s["cad"]["local_module_penetrations"]) for s in (before,after)])
        rows.append(summary)
    paired=np.array([r["paired"]["scores"] for r in rows],dtype=int).reshape(-1,2)
    update=np.array([r["updates"] for r in rows],dtype=int).reshape(-1,2)
    data=dict(plan_sha256=_hash(PLAN),complete=count==N,closed_pairs=count,rows=rows,aggregate={})
    if count:
        data["aggregate"]=dict(total=paired.sum(axis=0).tolist(),max=paired.max(axis=0).tolist(),
            median=np.median(paired,axis=0).tolist(),learning_updates=update.sum(axis=0).tolist(),
            fused_better=int(np.sum(paired[:,1]>paired[:,0])),fused_worse=int(np.sum(paired[:,1]<paired[:,0])),
            at_least={str(j):np.sum(paired>=j,axis=0).tolist() for j in (1,5,10,25,60,100)})
    _write(SUMMARY,data)
    print("TT93 paired live",count,data["aggregate"],flush=True)


def index_pair(old,new):
    t=[]
    for i,(a,b) in enumerate(zip(old["placements"]["delivered"],new["placements"]["delivered"])):
        if all(same_trace(a[n],b[n]) for n in ("far_contact","near_bounce")):
            t.append(dict(leg=i,stock=a["near_landed"],fused=b["near_landed"]))
    return dict(scores=[x["score"]["learner"] for x in (old,new)],matched_offers=t)


if __name__=="__main__":
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument("mode",choices=("plan","parity","run","summary","hundred"))
    {"plan":freeze,"parity":parity,"run":run,"summary":summarize,"hundred":inspect_hundred}[q.parse_args().mode]()

"""Learn and deploy a bank of centered SO-101 action sensitivities from paid ordinary strokes."""
import argparse
from collections import Counter
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt67_action_sensitivity as local
from . import tt59_fewshot_joint_games as game
from . import tt59_online_corrections as live
from .tt29_rally_physics import Rollout
from .tt39_carry_joints import Carried
from .tt4_transfer_trial import _hash, _read, _write
from .tt44_paired_credit import prefix

school, mpc, course, season = game.school, game.mpc, game.course, game.season
ROOT = Path("artifacts/tt74_sensitivity_bank")
PLAN, MODEL, OUT = (ROOT / x for x in ("protocol.json", "bank.json.gz", "games.json.gz"))
SOURCE_NAMES = (
    "tt42/after12", "tt42/after36", "tt42/after48", "tt42/direct-execution-held",
    "tt42/diverse-learning", "tt42/diverse-evaluation", "tt44_combined/combined",
    "tt48_transfer/continued-64", "tt48_transfer/live-plain",
)
SOURCES = tuple(Path("artifacts") / (n + ".json.gz") for n in SOURCE_NAMES)
SEED = 740623
WORLDS = 5
TRAIN = 34
ASSESS = 10
GATE_Y = -.82
CONTACT_Y = -1.48
MODES = ("stock", "conditional")
AUTHORITY = .032
H = school.H
KEY_SCALE = np.array([.14, .105, .16, .66, .78, .54, .52, .50, .38, .46, .48, .19, .19, .19, .21, .23])


def gate(x):
    return x is not None and x[4] < -1.2 and x[5] > .42 and -1.030 < x[1] <= GATE_Y


def key(x, q, ref):
    when = local.projected(x, CONTACT_Y)
    return np.r_[when, x[3:6], q[:5], (ref-q)[:5]] / KEY_SCALE


def family(x):
    z = local.projected(x, CONTACT_Y)[1]
    bx = local.projected(x, CONTACT_Y)[0]
    return int(np.searchsorted([-.10,.10],bx)), "low" if z<1.028 else "standard"


def protocol(feeds):
    rng = np.random.default_rng(SEED)
    schools = _read(school.PLAN)
    excluded = {r["id"] for s in ("held",) for r in schools[s]}
    # Fixed recorded bodies were all standard centered SO-101. Older
    # corrected branches and modified mount fixtures never enter this bank.
    examples = []
    supported = set()
    all_cards = {}
    ready0 = _read(mpc.BANK.with_suffix(".json"))["ready"]
    used_files = []
    for path in SOURCES:
        d = _read(path)
        if d["ready"] != ready0:
            continue
        used_files.append(path)
        for i,(c,r) in enumerate(zip(d["cards"],d["rows"])):
            if c["id"] in excluded or c["opening"] != "random" or c["profile"] != "reach":
                continue
            all_cards[c["id"]] = c
            if r["score"]["learner"]:
                supported.add(c["id"])
            requests, deliveries = (r["placements"][k] for k in ("requested","delivered"))
            if any(o.get("draw") != "random" for o in requests):
                continue
            for o in deliveries:
                # Sparse legitimate return credit; no future collision
                # position, contact clock, or radial coordinate is used.
                leg = o["leg"]
                if not o["near_landed"] or not (leg+1 < len(deliveries) and leg < len(requests)):
                    continue
                cut = int(np.floor(requests[leg]["time_s"]/mpc.DT+1e-7))
                if cut>720:
                    continue
                observed = [n for n in r["searches"] if n.get("p") and
                            cut+8<n["tick"]<cut+90 and -.75<n["p"][1]<-.53]
                if not observed:
                    continue
                n=observed[0]
                lane=int(np.searchsorted([-.095,.095], n["p"][0]))
                examples.append(dict(source=str(path), i=i, id=c["id"], leg=leg, card=c, cut=cut,
                    start=max(cut,n["tick"]-19), anchor_tick=n["tick"], p=n["p"], lane=lane,
                    public_robust=int(n.get("robust",0)), public_separation=n.get("best_separation_m",.2)))
    # Holdout releases come from the same measured-opening pool only to
    # preserve fixture/reach compatibility, with freshly drawn opponents.
    # Assignment is based on untreated archive opening viability, never
    # on a learned model or a treatment score.
    carry = _read(season.CARRY)
    training_readouts = {r["card"]["id"] for path in (mpc.BANK,course.PRACTICE)
                         for r in _read(path.with_suffix(".json"))["practice"]}
    training_readouts |= {c["id"] for c in carry["cards"][:len(carry["rows"])]}
    tt67_ids = {r["id"] for r in _read(local.PLAN)["cases"]}
    assessment = [k for k in rng.permutation(sorted(set(all_cards)-training_readouts-excluded-tt67_ids))
                  if k in supported][:ASSESS]
    if len(assessment) != ASSESS:
        raise ValueError("Too few untouched prospective release assignments")
    shots=mpc.measured_launches(feeds)
    prospective=[dict(card=dict(all_cards[k], seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))),
                      shot=shots[k], exploratory_seed=int(rng.integers(1,2**30))) for k in assessment]
    candidates = [examples[j] for j in rng.permutation(len(examples)) if examples[j]["id"] not in assessment]
    selected=[]
    seen=set()
    while len(selected)<min(len({c["id"] for c in candidates}),TRAIN+30):
        room = [r for r in candidates if r["id"] not in seen]
        if not room:
            break
        r = min(room,key=lambda v: (sum(p["lane"]==v["lane"] for p in selected),
                  v["cut"]//260, v["public_robust"]<=0, v["public_separation"]))
        selected.append(r)
        seen.add(r["id"])
    if len(selected)<TRAIN+4:
        raise ValueError(f"Only {len(selected)} independent source reserves")
    ROOT.mkdir(exist_ok=True,parents=True)
    fixed=(*used_files,local.PLAN,school.PLAN,course.OPTIC,mpc.BANK,course.PRACTICE,
           season.CARRY,season.CARRY.with_suffix("").with_suffix(".npz"),season.fresh.RISK)
    _write(PLAN,dict(schema="tt74-centered-sensitivity-bank-1", sources={str(f):_hash(f) for f in fixed},
         ready=ready0, examples=selected, releases=prospective, stock_memory=_hash(season.CARRY),
         teacher_worlds=WORLDS, desired_independent_train=TRAIN, authority=AUTHORITY,
         scope="Centered original SO-101 on new ordinary one-launch broad-random games. Training is independent by release. Candidate archive windows use physically paid returned-and-answered labels and earlier public search features, not exact future contact radius, location or clock. Source traces use normal RGB/encoder gate and freeze native response matrices before any assessment. Assessment has no source ID, leg, private physical state or future command. Only an under-supported ordinary search admits one verified correction. Real scored sent strokes can enter a bounded online memory without any new probing or extra practice."))
    print("TT74 fitted source reserves",len(selected),"prospective",[x["card"]["id"] for x in prospective],flush=True)


def design():
    p=_read(PLAN)
    if any(_hash(Path(f)) != h for f,h in p["sources"].items()):
        raise ValueError("Frozen sources changed")
    return p


def calibrate(count):
    p=design()
    d=dict(plan=_hash(PLAN), observations=[], complete=False)
    if MODEL.exists():
        pre=_read(MODEL)
        if pre["plan"] != d["plan"]:
            raise ValueError("Cannot resume altered bank")
        d=pre
    docs={}
    with mpc.centered(p["ready"]), school.enabled():
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        plant=Rollout(a,1)
        try:
            for spec in p["examples"][len(d["observations"]):]:
                if sum("entry" in r for r in d["observations"])>=count:
                    break
                start=process_time()
                src=spec["source"]
                if src not in docs:
                    docs[src]=_read(Path(src))
                original=docs[src]["rows"][spec["i"]]
                card=spec["card"]
                tick=spec["start"]
                prefix(a,card,original,tick,None,a._arm_ids[1][2],window=0)
                observed=local.acquire(a,cam,rig,original,tick,spec["anchor_tick"]+45,
                                       lambda t,x,eye:gate(x))
                item=dict(id=spec["id"], source=src, abs_public_tick=None)
                if observed is not None:
                    x,q,ref=(observed[k] for k in ("x","q","ref"))
                    t=observed["tick"]; item["abs_public_tick"]=t
                    if t+7 < len(original["actions"]):
                        r=mpc.tail(np.asarray(original["actions"][t:t+H]), H)
                        goals=school.atlas.reference_path(ref,r,stride,lo,hi)[1:]
                        stop=int(np.clip(round(local.projected(x,CONTACT_Y)[2]/mpc.DT)-2, 4, 13))
                        # Small source-only exact-physics ensemble. Native
                        # contacts at new deliveries never calibrate a J.
                        rng=np.random.default_rng([SEED,spec["id"],t])
                        try:
                            J,B,n,pose=estimate_short(a,plant,goals,local.smooth(stop),stop,rng)
                        except ArithmeticError:
                            item["rejected"]="insufficient finite sensitivity"
                        else:
                            # Estimate the strike's interior approach entirely
                            # from public optical path and source probe pose.
                            obs=local.projected(x,CONTACT_Y)
                            interior=float(np.linalg.norm(obs[:2]-pose[[0,2]]))
                            if interior < .133:
                                weak=float(np.linalg.svd(J[[0,2]],compute_uv=False)[-1])
                                example=dict(obs={k:vv.tolist() if isinstance(vv,np.ndarray) else vv
                                             for k,vv in observed.items()}, contact_line=CONTACT_Y, probe=stop,
                                             shape=(goals-ref).tolist(), J=J.tolist(), body_jac=B.tolist(),
                                             worlds=n, approach_residual_m=interior,
                                             authority=weak, key=key(x,q,ref).tolist(), family=family(x), source="offline")
                                item["entry"]=example
                            else:
                                item["rejected"]="observed interior support absent"
                d["observations"].append(item)
                accepted=sum("entry" in r for r in d["observations"])
                d["complete"]=accepted>=count
                _write(MODEL,d)
                print("fit",spec["id"],accepted,item.get("rejected"),item.get("entry",{}).get("authority"),
                      round(process_time()-start,2),flush=True)
        finally:
            cam.close(); a.close()
    if not d["complete"]:
        raise ValueError("Ran out of independent observed, paid interior anchors")


def estimate_short(a, plant, goals, shape, when, rng):
    # Same finite-difference identification as TT67, with a smaller ensemble
    # to spread the native probes over many independent scored bodies.
    snapshot=school.State(a)
    axes,dofs=a._arm_ids[1][2],a._near_dofs
    pos=a.model.jnt_qposadr[a.model.actuator_trnid[axes,0]]
    kin=local.BladeIK(a)
    directions=np.r_[np.eye(5)*local.RATE,-np.eye(5)*local.RATE,np.zeros((1,5))]
    # Need only the pre-contact clock; the rest of the motion is not a
    # candidate held score. A fixed extra frame preserves the observed row.
    moves=np.asarray([goals[:when+1]+shape[:when+1]*np.r_[d,0.] for d in directions])
    values=[]
    try:
        for k in range(WORLDS):
            snapshot.restore(a)
            delta=np.zeros(10) if k==0 else rng.normal(size=10)*np.r_[[.012,.014,.017,.017,.020],[.16]*5]
            a.data.qpos[pos[:5]]+=delta[:5]
            a.data.qvel[dofs[:5]]+=delta[5:]
            tr=np.full((*moves.shape[:2],17),np.nan)
            plant.trajectories(moves,school.AIM,.39,traces=tr)
            q=tr[:,when,6:11]
            if not np.isfinite(q).all():
                if k==0:
                    raise ArithmeticError("No nominal pre-impact probe")
                continue
            blade=np.asarray([np.r_[z[0],z[1][[0,2]]] for z in map(kin.pose,q)])
            values.append((delta,blade[-1],((blade[:5]-blade[5:10])/(2*local.RATE)).T))
    finally:
        snapshot.restore(a)
    if len(values)<3:
        raise ArithmeticError("Too few supported perturbations")
    B=np.asarray([x[0] for x in values]); y=np.asarray([x[1] for x in values])-values[0][1]
    response=np.linalg.solve(B.T@B+.012*np.eye(10),B.T@y).T
    return np.median([x[2] for x in values],axis=0),response,len(values),values[0][1]


class Bank:
    def __init__(self, data, acquired=()):
        self.source=[r["entry"] for r in data["observations"] if "entry" in r]
        self.acquired=list(acquired)
        if len(self.source)<30 or not data["complete"]:
            raise ValueError("Assessment requires at least 30 independent sensitivity roots")

    def suggest(self, x, q, ref):
        target=key(x,q,ref)
        obs=dict(x=x,q=q,ref=ref)
        merged=self.source+self.acquired[-12:]
        distance=np.linalg.norm(np.asarray([r["key"] for r in merged])-target,axis=1)
        out, notes=[],[]
        for j in np.argsort(distance,kind="stable"):
            m=merged[j]
            d=float(distance[j])
            if d>2.75:
                break
            if m["authority"]<AUTHORITY or m["family"][0]!=family(x)[0]:
                continue
            goals, note=local.policy(m,obs,"forward")
            change=np.asarray(note["observed_displacement"][:2])
            if not .012<=np.linalg.norm(change)<=.103:
                continue
            # Prefer a materially supported correction, not an arbitrary
            # stroke lookup that happens to have a close state.
            gain=np.clip(np.asarray(note["gain"]),-.145,.145)
            predicted=np.asarray(m["J"])[[0,2]]@gain
            improvement=np.linalg.norm(change)-np.linalg.norm(change-predicted)
            if improvement<.007:
                continue
            out.append(goals)
            notes.append(dict(distance=d,transported_m=float(improvement),library_index=int(j),
                              experience=m["source"],**note))
            if len(out)==8:
                break
        return out,notes


class Conditional(live.Online):
    def search(self, tick, x, q, ref, base):
        season.Player.search(self,tick,x,q,ref,base)
        if self.examined or not gate(x) or not self.notes or self.notes[-1]["tick"]!=tick:
            return
        self.examined=True
        lane, depth=family(x)
        public=dict(tick=tick, flight=x.tolist(), lane=lane, depth=depth)
        ordinary=self.notes[-1]
        if ordinary.get("robust",0)>0 or ordinary.get("legal_trials",0)>0:
            self.repairs.append(dict(public,issued=False,reason="stock-supported"))
            return
        paths, n = self.readout.suggest(x,q,ref)
        if not paths or not self.engine.sync():
            self.repairs.append(dict(public,issued=False,reason="no-supported-local-transport",candidates=len(paths)))
            return
        options,diag=live.base.rank_paths(self.engine,np.asarray(paths),ref,n)
        robust=[(u,d) for u,d in options if d["legal"] and d["robust"]>0]
        if not robust:
            self.repairs.append(dict(public,issued=False,reason="projection-not-robust",**diag))
            return
        rates,pick=min(robust,key=lambda item:item[1]["score"])
        self.pending=school.atlas.reference_path(ref,rates,*self.limits)[1:]
        self.pending_tick=tick
        self.best,self.tick,self.lease=self.pending,tick,H
        seeds=self.readout.source+self.readout.acquired[-12:]
        self.credit_context=(x.copy(),q.copy(),ref.copy(),seeds[pick["library_index"]],[])
        self.repairs.append(dict(public,issued=True,selection=pick,**diag))

    def sensory_credit(self,tick,sent_reference,legal_near):
        if self.credit_context is None:
            return
        x,q,ref,teacher,sent=self.credit_context
        age=tick-self.pending_tick
        if age>105:
            self.credit_context=None
            return
        if age<H:
            sent.append(sent_reference.copy()-ref)
        if legal_near:
            if len(sent)>=6:
                # A real credited act supplies its local shape and sensory
                # root. It inherits the frozen source derivative with
                # reduced authority, not a fabricated new Jacobian.
                entry=dict(teacher,obs=dict(x=x.tolist(),q=q.tolist(),ref=ref.tolist()),
                           shape=mpc.tail(np.asarray(sent),H).tolist(),key=key(x,q,ref).tolist(),
                           authority=teacher["authority"]*.80,family=family(x),source="paid-online")
                self.readout.acquired.append(entry)
                self.learned_here+=1
            self.credit_context=None


def games(n):
    p=design()
    initial=_read(MODEL)
    rec=dict(plan=_hash(PLAN),bank=_hash(MODEL),rows=[],new_entries=[],complete=False)
    if OUT.exists():
        rec=_read(OUT)
        if rec["plan"]!=_hash(PLAN) or rec["bank"]!=_hash(MODEL):
            raise ValueError("Different frozen continuous bank or assignments")
    book=Bank(initial,rec["new_entries"])
    ids={c["card"]["id"] for c in p["releases"]}
    if ids & {r["id"] for r in initial["observations"]}:
        raise ValueError("Calibration release entered prospective docket")
    carry=_read(season.CARRY)
    ready=np.asarray(p["ready"])
    ranking=season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,"local")
    with mpc.centered(ready),school.enabled():
        a=mpc.arena()
        cam,rig,stride,lo,hi=mpc.resources(a)
        limits=(stride,lo,hi)
        actors={}
        try:
            for mode in MODES:
                opts=dict(coverage=ranking,variant="elastic",budget=28)
                with patch.object(mpc,"Fragments",lambda b,e:Carried(b,e,practice=[course.PRACTICE],held=ids)):
                    actors[mode]=(season.Player(mpc.BANK,raw,**opts) if mode=="stock" else
                        Conditional(mpc.BANK,raw,readout=book,mode=mode,limits=limits,**opts))
                course.resume(actors[mode],season.CARRY.with_suffix("").with_suffix(".npz"),carry)
            for ex in p["releases"][len(rec["rows"]):n]:
                results={}
                c=ex["card"]
                for mode in MODES:
                    started=process_time()
                    row=live.play(a,cam,rig,limits,ready,c,ex["shot"],raw,actors[mode],ex["exploratory_seed"])
                    row["cpu_s"]=process_time()-started
                    if mode!="stock":
                        row["same_offer_changes"]=game.compare(results["stock"],row)
                    results[mode]=row
                    print("point",c["id"],mode,row["score"]["learner"],len(row["placements"]["delivered"]),
                          Counter(x.get("reason","issued") for x in row["repairs"]),row["motor_lessons"],
                          round(row["cpu_s"],2),flush=True)
                rec["rows"].append(dict(ex,arms=results))
                rec["new_entries"]=book.acquired
                rec["complete"]=len(rec["rows"])==len(p["releases"])
                _write(OUT,rec)
        finally:
            for actor in actors.values():
                actor.close()
            cam.close()
            a.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","fit","games"))
    parser.add_argument("--n",type=int)
    parser.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    a=parser.parse_args()
    if a.mode=="plan":
        protocol(a.feeds)
    elif a.mode=="fit":
        calibrate(a.n or TRAIN)
    else:
        games(a.n or ASSESS)

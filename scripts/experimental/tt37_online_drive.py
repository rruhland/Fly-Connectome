"""Receive on the geometric clinic scaffold, learn timing and drive from real RGB outcomes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import BladeIK, measured_launches
from .tt31_ready import centered
from .tt34_results import paired
from .tt36_impact_refine import Frames
from .tt37_motor_clinic import SCALE, geometry, recall, schedules
from .tt37_online import OUT, PUPIL, ROOT
from .tt37_opponent import arena
from .tt37_posture import Surface, sample

DEPTH = -1.56
TILT = .35
CUT = .185
GRIP = -.1
SHAPE = np.array([.055,.65])
BANK = OUT/"clinic-deep.json.gz"
OBSERVATION = ROOT/"tt36"/"impact-model.json"
ANCHORS = (9412,12166,6208,11139,6011,5864)
GRID = np.mgrid[-1.0:1.01:.25,-1.0:1.01:.25].reshape(2,-1).T
CAP = 32


def plan():
    original = _read(OUT/"opponent-plan.json")
    cases = {c["id"]:c for c in original["practice"]}
    groups = {}
    for name, blocks, seed in (("practice",4,378201),("continuations",2,379911)):
        cards = []
        for i in range(blocks):
            for sid in ANCHORS:
                n = len(cards)
                cards.append(dict(cases[sid],opening="anchor",reply_seed=seed+11*n,key=f"drive-{name}-{n}-{sid}"))
        groups[name] = cards
    groups["new"] = original["held_openings"]
    groups["feed_sha256"] = original["feed_sha256"]
    _write(OUT/"online-drive-plan.json",groups)


def grade(legal, observation, landing):
    if legal:
        return 1.0
    if observation is None:
        return None
    forward = float(np.clip((observation["outgoing"][1]+4.0)/8.0,0.,1.))
    lift = float(np.clip((observation["outgoing"][2]+1.5)/3.3,0.,1.))
    side = 0. if landing is None else float(np.clip((landing+.90)/1.20,0.,1.))
    return .40*forward+.10*lift+.13*side


def bounce(frames, hit):
    if hit is None:
        return None
    samples = [(t,p) for t,p,_ in frames.stereo if hit["time"]+.035<t<hit["time"]+1.6]
    for (a,p),(b,q),(c,r) in zip(samples,samples[1:],samples[2:]):
        if c-a<.103 and .740<q[2]<.823 and q[2]<min(p[2],r[2])-.003:
            return float(q[1])
    return None


class LocalDrive:
    def __init__(self, samples=()):
        self.samples = list(samples)

    def choose(self, x, rng, exploring):
        if not self.samples:
            if exploring:
                return rng.uniform(-.9,.9,2), True
            return np.zeros(2), False
        xs = np.array([r["context"] for r in self.samples])
        ps = np.array([r["parameter"] for r in self.samples])
        rewards = np.array([r["value"] for r in self.samples])
        w = np.exp(-.50*np.sum(((xs-x)/SCALE)**2,axis=1))
        c = np.vstack((np.zeros((1,2)),ps,GRID))
        psim = np.exp(-2.2*np.sum((c[:,None]-ps[None,:])**2,axis=-1))*w[None,:]
        mass = psim.sum(axis=1)
        prior = .46-.035*np.sum(c**2,axis=1)
        objective = (1.0*prior+psim@rewards)/(1.0+mass)
        objective[1:][mass[1:]<.32] = -1.
        control = c[np.argmax(objective)]
        if exploring and rng.random()<.67:
            return np.clip(control+rng.normal(0.,.49,2),-1.,1.), True
        return control, False

    def observe(self, rows):
        self.samples.extend(r for r in rows if r["value"] is not None)
        self.samples = self.samples[-160:]


def point(a, camera, rig, model, surf, stride, lo, hi, card, shot, bank, memory, mode, observer, rng):
    a.launch(shot,card=card)
    a.partner.reset(seed=card["seed"])
    actor=joint.JointPolicy(model|{"ready":surf.ready.tolist()},stride,lo,hi)
    eye=joint.Vision(rig,a.control_dt)
    view=Frames(rig)
    records,actions=[],[]
    tape=None
    deadline=lastprep=-100.
    prep=surf.ready
    start=process_time()
    for tick in range(a.max_steps):
        now=tick*a.control_dt
        q=a.joints
        # Extra frames are used for the signed observer, never injected into the
        # scaffold's regular 30 Hz visual estimate to keep controls comparable.
        dense=bool(records and now<records[-1]["selected_s"]+records[-1]["eta_s"]+.215)
        frame=camera.capture() if tick%2==0 or dense else None
        if frame is not None:
            view.observe(frame,now,q)
        obs=eye.update(tick,frame if tick%2==0 else None,q)
        if tape is not None and tick-tape_start<len(tape):
            desired=tape[tick-tape_start]
        else:
            tape=None
            desired=surf.ready
            if obs is not None and .03<obs[0]<.80 and now>deadline:
                if obs[0]<=CUT:
                    eta=obs[0]
                    fit=geometry(surf,obs)
                    x=np.r_[obs[1:6],q[:5]]
                    remembered=recall(bank,x)
                    parameter, probe = memory.choose(x,rng,mode=="learn")
                    timing,gain=SHAPE*parameter
                    modified=(fit[0],fit[1]*(1.+gain))
                    tape=schedules(*modified,eta+timing,np.array([remembered]),a.control_dt,surf.ready,
                                   max(1,round((eta+.075)/a.control_dt)))[0]
                    desired=tape[0]
                    tape_start=tick
                    deadline=now+eta+.26
                    records.append(dict(tick=tick,selected_s=now,eta_s=float(eta),context=x.tolist(),
                                        parameter=parameter.tolist(),probed=probe,legal=False))
                else:
                    if tick-lastprep>=4:
                        p,v=geometry(surf,obs)
                        prep=np.r_[p-.075*v,GRIP]
                        lastprep=tick
                    desired=prep
        action=actor.drive(desired,q,.8)
        _,_,end,capped,events=a.exchange(action,joint.FAR_COMMAND)
        actions.append(action.tolist())
        if records and any(e["kind"]=="legal_return" and e["side"]==1 for e in events["events"]):
            records[-1]["legal"]=True
        if end or capped or a.legal_returns+a.rival_returns-1>=CAP:
            break
    hits=view.estimates(observer)
    for s in records:
        time=s["selected_s"]+s["eta_s"]
        choices=[h for h in hits if abs(h["time"]-time)<.155]
        hit=min(choices,key=lambda h:abs(h["time"]-time)) if choices else None
        landing=bounce(view,hit)
        s.update(impact=hit, signed_landing_y_m=landing,value=grade(s["legal"],hit,landing))
    return dict(actions=actions,records=records,events=a.events.copy(),placements=a.point_placements(),
                score=audit(a.events,opening_side=1),shot=shot,opponent_seed=card["seed"],
                runtime_cpu_s=process_time()-start)


def run(args):
    source=OUT/"online-drive-plan.json"
    roster=_read(source)
    if _hash(args.feeds)!=roster["feed_sha256"]:
        raise ValueError("Wrong measured releases")
    cards=[c for c in roster[args.phase] if not args.ids or c["id"] in args.ids][:args.limit]
    memory_file=OUT/"online-drive-checkpoint.json"
    source_bank=_read(BANK)["bank"] if args.mode!="nominal" else []
    state=_read(memory_file) if args.mode in ("carry","binary") else None
    if state and set(state["reply_seeds"])&{c["reply_seed"] for c in cards}:
        raise ValueError("A held rollout uses an acquisition seed")
    initial=OUT/"online-drive-practice-known.json.gz" if args.mode=="learn" else None
    if initial is not None:
        initial_tape=_read(initial)
        if initial_tape["cards"]!=cards[:6] or len(initial_tape["rows"])!=6:
            raise ValueError("Finish the matched frozen memory control on the first six practice cards")
        starting=[r for row in initial_tape["rows"] for r in row["records"] if r["value"] is not None]
    else:
        starting=state["samples"] if state else []
    if args.mode=="binary":
        # Same physical trials and observational censoring as the continuously
        # graded selector. Only the stored labels change.
        starting=[dict(s,value=float(s["legal"])) for s in starting]
    memory=LocalDrive(starting)
    observer=_read(OBSERVATION)["models"]["60"]
    model=_read(PUPIL)
    shot=measured_launches(args.feeds)
    nominal=arena()
    ik=BladeIK(nominal)
    surf=Surface(ik,sample(ik),DEPTH,TILT)
    nominal.close()
    joint.RECEIVE_Y=DEPTH
    data=dict(phase=args.phase,mode=args.mode,cards=cards,ready=surf.ready.tolist(),depth=DEPTH,
              tilt=TILT,cut=CUT,opening_side=1,cap=CAP,protocol="random",rows=[],
              bank_sha256=_hash(BANK) if source_bank else None,observer_sha256=_hash(OBSERVATION),
              plan_sha256=_hash(source),memory_sha256=_hash(memory_file) if state else None,
              fresh_control_prior_sha256=_hash(initial) if initial else None, initial_samples=len(starting))
    output=OUT/f"online-drive-{args.phase}-{args.mode}.json.gz"
    with centered(surf.ready):
        physical=arena()
        camera,rig,stride,lo,hi=joint.resources(physical)
        try:
            for i,c in enumerate(cards):
                row=point(physical,camera,rig,model,surf,stride,lo,hi,c,shot[c["id"]],source_bank,memory,
                          args.mode,observer,np.random.default_rng(379181+i))
                row["memory_before"]=len(memory.samples)
                if args.mode=="learn":
                    memory.observe(row["records"])
                row["memory_after"]=len(memory.samples)
                data["rows"].append(row)
                _write(output,data)
                if args.mode=="learn":
                    _write(memory_file,dict(samples=memory.samples,reply_seeds=[c["reply_seed"] for c in cards[:i+1]],
                                            acquisition_sha256=_hash(output)))
                print(args.phase,args.mode,i+1,c["id"],row["score"]["learner"],row["score"]["returns"],
                      [(r["parameter"],r["legal"],r["impact"] and round(r["impact"]["outgoing"][1],3))
                       for r in row["records"]],flush=True)
        finally:
            camera.close()
            physical.close()


def comparisons():
    report={}
    read=lambda k,m:_read(OUT/f"online-drive-{k}-{m}.json.gz")
    for phase,modes in (("practice",("nominal","known","learn")),
                        ("continuations",("known","carry")),("new",("known","carry"))):
        d=report[phase]={}
        for mode in modes:
            source=read(phase,mode)
            rows=source["rows"]
            if len(rows)!=len(source["cards"]):
                raise ValueError("The match tape is incomplete")
            if any(sum(r["legal"] for r in row["records"])!=row["score"]["learner"] for row in rows):
                raise ValueError("Actual legal-return events did not uniquely attach to actions")
            if mode!="learn" and any(row["memory_after"]!=row["memory_before"] for row in rows):
                raise ValueError("Frozen checkpoint changed")
            plays=[r["score"]["learner"] for r in rows]
            impacts=[s for row in rows for s in row["records"] if s["impact"] is not None]
            first=dict()
            for c,row in zip(source["cards"],rows,strict=True):
                stroke=next(iter(row["records"]),{})
                event=stroke.get("impact")
                first.setdefault(c["id"],[]).append(dict(
                    legal=stroke.get("legal",False), rgb_forward_mps=None if event is None else event["outgoing"][1],
                    signed_table_height_y_m=stroke.get("signed_landing_y_m"), control=stroke.get("parameter")))
            d[mode]=dict(points=len(rows),legal=sum(plays),first=sum(v>0 for v in plays),
                         later=sum(max(0,v-1) for v in plays),longest=max(plays,default=0),
                         alternating=sum(r["score"]["returns"] for r in rows),
                         per_point=plays,rgb_signed_impacts=len(impacts),
                         changed_controls=sum(bool(np.any(s["parameter"])) for row in rows for s in row["records"]),
                         cpu_s=sum(r["runtime_cpu_s"] for r in rows),opening_by_release=first)
        matched=[read(phase,m) for m in modes]
        reference=matched[0]
        l=len(reference["cards"])
        for doc, m in zip(matched[1:],modes[1:],strict=True):
            if reference["cards"]!=doc["cards"][:l]:
                raise ValueError("The paired physical trials differ")
            subset=[i for i,row in enumerate(reference["rows"]) if row["placements"]["delivered"]]
            active=[i for i,row in enumerate(doc["rows"][:l]) if row["placements"]["delivered"]]
            if active!=subset:
                raise ValueError("The actual first legal offers changed")
            revised=[dict(s,cards=[s["cards"][i] for i in subset],rows=[s["rows"][i] for i in subset])
                     for s in (reference,doc)]
            d[m]["paired_against_first_condition"]=paired(*revised)
            d[m]["opponent_initial_faults"]=l-len(subset)
    binary=read("continuations","binary")
    combined=read("continuations","carry")
    selected=[combined["cards"].index(c) for c in binary["cards"]]
    subset=dict(combined,cards=[combined["cards"][i] for i in selected],rows=[combined["rows"][i] for i in selected])
    if binary["memory_sha256"]!=combined["memory_sha256"]:
        raise ValueError("The scoring-only control used different acquired experience")
    if any(row["score"]["learner"]!=sum(s["legal"] for s in row["records"]) for row in binary["rows"]):
        raise ValueError("Legal rewards do not match the control referee")
    report["binary_6208_comparison"]=paired(binary,subset)
    _write(OUT/"online-drive-results.json",report)


def verify():
    receipts={}
    with TemporaryDirectory(prefix="tt37-drive-replay-") as temporary:
        tape,proof=(Path(temporary)/f for f in ("subset.json.gz","audit.json"))
        for phase in ("continuations","new"):
            source=OUT/f"online-drive-{phase}-carry.json.gz"
            app=_read(source)
            compare=_read(OUT/f"online-drive-{phase}-known.json.gz")
            scores=[d["score"]["learner"]-b["score"]["learner"]
                    for d,b in zip(app["rows"],compare["rows"],strict=True)]
            picks=[i for i,s in enumerate(scores) if s]
            if not picks:
                picks=[max(range(len(scores)),key=lambda i:app["rows"][i]["score"]["learner"])]
            sub=dict(app,cards=[app["cards"][i] for i in picks],rows=[app["rows"][i] for i in picks])
            _write(tape,sub)
            with patch.object(joint,"replay_arena",lambda cards,**kw:arena(cards,protocol="random",**kw)):
                joint.verify(tape,proof,cad=True)
            check=_read(proof)["points"]
            receipts[phase]=dict(scoring_differences=[scores[i] for i in picks],positions=picks,replay=check,
                                 source_sha256=_hash(source))
    _write(OUT/"online-drive-audit.json",receipts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","practice","continuations","new","summary","audit"))
    p.add_argument("mode",choices=("nominal","known","learn","carry","binary"),nargs="?")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--limit",type=int)
    p.add_argument("--ids",type=int,nargs="+")
    args=p.parse_args()
    operation={"plan":plan,"summary":comparisons,"audit":verify}.get(args.phase)
    operation() if operation else run(args)

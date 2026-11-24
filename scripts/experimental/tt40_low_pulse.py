"""Practice a signed low-ball lift around the incumbent's actually reachable motion."""
import argparse
from pathlib import Path

import numpy as np

from . import tt40_low_lift as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

LOW = (.860, 1.003)
# Wrist flex opens/lowers the face, elbow and shoulder then accelerate up.
MOVES = [
    (0.,0.,0.,0.), (-.14,.0,0.,.0), (-.25,0.,0.,0.), (-.37,0.,0.,0.),
    (-.14,.11,0.,.0), (-.25,.10,.04,.0), (-.37,.12,.06,.0),
    (-.24,.10,.04,-.12), (-.24,.10,.04,.12),
    (-.38,.18,.08,-.13), (-.38,.18,.08,.13),
    (-.43,.20,.10,.0), (-.56,.24,.13,.0),
    (-.25,-.04,0.,.0), (-.36,-.04,0.,.0), (-.38,.08,.0,.0),
    (-.08,.11,.06,.0), (-.16,.07,.02,.12),
]


def source_cases(d):
    for c,r,_ in core.train_cases(d):
        stroke=r["placements"]["delivered"][0]
        e=stroke["near_contact"]["time"] if stroke["near_contact"] is not None else stroke["near_bounce"]["time"]+.5
        rr=[p for p in r["records"] if stroke["far_contact"]["time"]<p["tick"]/60<e+.035]
        if rr and LOW[0]<rr[-1]["context"][1]<LOW[1]:
            yield c,r,rr[-1]


def pulse(center, motion, selected, eta, dt, home, move, *, n=22):
    stock=core.clinic.schedules(center,motion,eta,selected[None],dt,home,n)[0]
    wf, sl, el, roll = move
    sample=(np.arange(n)+1)*dt
    gate=np.minimum(1.,sample/max(.036,eta*.49))
    gate=gate**2*(3.-2.*gate)
    # Maintain a low, pitched face before impact, then drive upward on approach.
    drive=np.clip((sample-eta+.075)/.067,0.,1.)
    stock[:,1]+=sl*gate*drive
    stock[:,2]+=el*gate*drive
    stock[:,3]+=wf*gate
    stock[:,4]+=roll*gate
    return stock


def try_moves(args):
    d=_read(core.SCHOOL)
    cases=list(source_cases(d))[:args.limit]
    output=dict(archive=str(core.SCHOOL),source_sha256=_hash(core.SCHOOL),code_sha256=_hash(Path(__file__)),
                mode="act, observe and score repeated receiving opportunities in physical practice",
                candidates=MOVES, rows=[])
    core.joint.RECEIVE_Y=d["depth"]
    surf=core.setup(d["depth"])
    model=_read(core.PUPIL)
    with core.centered(d["ready"]):
        a=core.arena()
        engine=core.OpticalClinic(body=True)
        stereo,rig,stride,lo,hi=core.joint.resources(a)
        try:
            for c,r,decision in cases:
                tick=decision["tick"]
                eye=core.joint.Vision(rig,a.control_dt)
                eye.eye=core.LearnedOptics(eye.eye,d["visual"])
                observations=core.ArrivalHistory()
                def start(rebuild):
                    a.launch(r["shot"],card=c)
                    a.partner.reset(seed=c["seed"])
                    actor=core.joint.JointPolicy(model|{"ready":d["ready"]},stride,lo,hi)
                    for k in range(tick+1):
                        q=a.joints.copy()
                        obs=None
                        if rebuild:
                            obs=eye.update(k,stereo.capture() if k%2==0 else None,q)
                            obs=observations.update(k*a.control_dt,obs)
                        if k==tick:
                            break
                        u=np.asarray(r["actions"][k])
                        a.exchange(u,core.clinic.FAR)
                        actor.reference=np.clip(actor.reference+u*stride,lo,hi)
                    if digest_json(a.events)!=digest_json([e for e in r["events"] if e["step"]<=tick]):
                        raise ValueError("The randomized original physical prefix did not reproduce")
                    return actor,obs,q
                _,obs,q=start(True)
                if obs is None or not np.allclose(np.r_[obs[1:6],q[:5]],decision["context"],atol=1e-9):
                    raise ValueError("The original RGB and encoder prefix did not reproduce")
                center, motion = core.clinic.geometry(surf,obs)
                selected=np.asarray(decision["action"])
                n=max(1,round((obs[0]+.075)/a.control_dt))
                trials=[]
                accepted=[]
                for i,mod in enumerate(MOVES):
                    actor,_,_=start(False)
                    before=len(a.events)
                    tape=pulse(center,motion,selected,obs[0],a.control_dt,surf.ready,mod,n=n)
                    for k in range(100):
                        goal=tape[k] if k<len(tape) else surf.ready
                        u=actor.drive(goal,a.joints.copy(),.8)
                        _,_,done,cap,info=a.exchange(u,core.clinic.FAR)
                        scored=any(e["kind"]=="legal_return" and e["side"]==1 for e in info["events"])
                        if scored or done or cap:
                            break
                    result=core.audit(a.events,opening_side=1)
                    ep=a.events[before:]
                    end = next((e for e in reversed(ep) if e["kind"]=="point"),{})
                    strike = next((e for e in ep if e["kind"]=="separation" and e.get("collider")=="learner_racket"),None)
                    landed = bool(result["learner"])
                    if landed:
                        accepted.append(i)
                    trials.append(dict(index=i,legal=landed,contact=strike is not None,
                                       departure=None if strike is None else list(strike["velocity"]),
                                       location=None if strike is None else list(strike["ball"]),
                                       fault=end.get("reason"),events=ep if landed else None))
                print(c["id"], tick, "legal",accepted, flush=True)
                output["rows"].append(dict(card=c,tick=tick,camera=obs.tolist(),trials=trials))
                _write(args.output,output)
        finally:
            stereo.close();engine.twin.close();a.close()


class Pulse:
    """Promote physical practice, then use only currently sensed optical context."""
    scale = np.array([.078,.126,.046,.91,.64,.88])

    def __init__(self, practice, minimum=1):
        votes=np.zeros(len(MOVES),dtype=int)
        totals=np.zeros(len(MOVES),dtype=int)
        self.support={}
        for e in practice["rows"]:
            stock = next((t for t in e["trials"] if t["index"]==0),None)
            for trial in e["trials"]:
                j=trial["index"]
                votes[j]+=trial["legal"]
                totals[j]+=1
                if trial["legal"] and stock is not None and not stock["legal"] and "camera" in e:
                    self.support.setdefault(j,[]).append(np.asarray(e["camera"][:6]))
        self.learned=np.flatnonzero((votes>=votes[0]+minimum) & (np.arange(len(votes))>0))
        self.rank=sorted(self.learned,key=lambda k:(votes[k]/max(totals[k],1),-np.linalg.norm(MOVES[k])))[::-1]
        self.votes=votes.tolist()

    def plan(self,engine,surf,obs,selected,score,fit,eta):
        note=dict(active=False,low_estimate=float(obs[2]))
        status=dict(tape=None,score=None,eta=None,trace=[],note=note)
        if (not self.rank or not (LOW[0]<obs[2]<LOW[1] and .062<eta<.184)
                or (score is not None and score[12]<.055)):
            return status
        # The stock chooses the basic schedule. The learned pulse mode gets
        # vetoed outside the optical support of actual successful practice.
        current = np.asarray(obs[:6])
        for choice in self.rank:
            references = self.support.get(choice,[])
            if not len(references):
                continue
            near = float(np.min(np.linalg.norm((np.asarray(references)-current)/self.scale,axis=1)))
            if near > 1.8:
                continue
            dt=engine.twin.control_dt
            n=max(1,round((eta+.075)/dt))
            note.update(active=True,issued=True,pulse_id=int(choice),distance=near,
                        supporting_play_wins=self.votes[int(choice)])
            tape=pulse(*fit,selected,eta,dt,surf.ready,MOVES[choice],n=n)
            return dict(tape=tape,score=score,eta=eta,trace=[],note=note)
        return status


def held(args):
    group=core._read(args.plan)[args.group][:args.limit]
    d=_read(args.training)
    skill=Pulse(d,minimum=2)
    if not skill.rank:
        raise ValueError("No separately scored pulse motions earned more than the stock practice motion")
    if {c["id"] for c in group} & {r["card"]["id"] for r in d["rows"]}:
        raise ValueError("Evaluation release repeats pulse practice")
    core.joint.RECEIVE_Y=core.TRAIN_DEPTH
    surf=core.setup(core.TRAIN_DEPTH)
    shots=core.measured_launches(args.feeds)
    model=_read(core.PUPIL)
    checkpoint=_read(core.ROOT/"tt37/independent-cold-value.json.gz")
    judge=core.ROOT/"tt37/motor-value.json.gz"
    with np.load(core.ROOT/"tt37/self-combined.npz") as bank:
        pupil=core.DeepField(bank,"local")
    params=core.clinic.generate(35)
    doc=dict(cards=group,profile="reach",protocol="random",cap=args.cap,training=str(args.training),
             training_sha256=_hash(args.training),rows=[],practice_votes=skill.votes,
             training_repeat_wins_required=2, source_code_sha256=_hash(Path(__file__)),
             excludes_calibration=[r["card"]["id"] for r in d["rows"]])
    with core.centered(surf.ready):
        a=core.arena()
        engine=core.OpticalClinic(body=True)
        engine.motor_value=core.MotorValue(_read(judge))
        stereo,rig,stride,lo,hi=core.joint.resources(a)
        try:
            for c in group:
                rolls=[]
                for add in (False,True):
                    r=core.clinic.point(a,stereo,rig,model,surf,stride,lo,hi,c,shots[c["id"]],
                        list(checkpoint["bank"]),params,"optical-fixed",args.cap,.18,3,True,None,None,True,
                        False,None,engine,core.OPTICS,pupil=pupil,economy=True,impact=skill if add else None)
                    rolls.append(r)
                stock,armed=rolls
                first=next((k for k,(x,y) in enumerate(zip(stock["actions"],armed["actions"])) if x!=y),None)
                doc["rows"].append(dict(card=c,baseline=stock,special=armed,first_change=first))
                _write(args.output,doc)
                print(c["id"],stock["score"]["learner"],armed["score"]["learner"],first,
                    [x["impact_margin"]["pulse_id"] for x in armed["records"] if x.get("impact_margin",{}).get("issued")],flush=True)
        finally:
            stereo.close();engine.twin.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("practice","held"))
    p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--cap",type=int,default=200)
    p.add_argument("--plan",type=Path,default=core.ROOT/"tt39/endurance-plan.json")
    p.add_argument("--group",default="assessment")
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--training",type=Path,default=core.ROOT/"tt40/low-pulse-practice.json.gz")
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    (try_moves if a.mode=="practice" else held)(a)

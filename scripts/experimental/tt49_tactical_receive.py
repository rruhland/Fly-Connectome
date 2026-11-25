"""Play the next actual reply after each matched physical stroke with one ordinary RGB receiver."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt37_opponent as live
from . import tt42_learning_course as course
from . import tt45_tactical_coverage as study
from . import tt49_goal_retrieval as retrieval
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt39_opportunity_map import projected_height
from .tt44_paired_credit import State

ROOT = retrieval.ROOT
MATRIX = study.ROOT / "physical-arc-plan.json.gz"
PLAN = ROOT / "receiver-plan.json"
IDS = (6619, 2920, 8627)


def plan():
    original = _read(MATRIX)
    selected = [next(r for r in original["cases"] if r["id"] == i) for i in IDS]
    seen = {r["card"]["id"] for p in [mpc.BANK, course.PRACTICE]
            for r in _read(p.with_suffix(".json"))["practice"]}
    if seen.intersection(IDS):
        raise ValueError("Receiver's frozen motor practice contains a challenge release")
    new = np.random.default_rng(490611).integers(1, 2**30, 4).tolist()
    if set(new) & set(original["draw_seeds"]):
        raise ValueError("Challenge redraw overlaps previous height assay")
    _write(PLAN, dict(previous_matrix_sha256=_hash(MATRIX), cases=selected, reply_seeds=new,
                      source_receipts=original["sources"],
                      sources={str(p):_hash(p) for p in [mpc.BANK,course.PRACTICE,course.OPTIC]},
                      protocol="Three already frozen physical stroke contrasts. Four common new continuous TT37 draws per pair. Same ordinary RGB encoder receiver, no updates or world information after the actual prior arm action."))


def sensor(a, res):
    eye = mpc.Vision(res[1], a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, course.OPTIC)
    return eye


def inbound(x):
    return x is not None and x[4] < -.9 and x[1] < -.90


class ReceiverSnapshot:
    """Restore native body and the ordinary receiver's causal sensor history."""
    def __init__(self, a, model, sources, original, hit, ready, res):
        cam, rig, stride, lo, hi = res
        a.launch(original["shot"], card=sources)
        a.partner.reset(seed=sources["seed"])
        model.begin()
        eye, ref = sensor(a, res), np.asarray(ready).copy()
        self.sensed = False
        for tick in range(hit):
            q = a.joints
            x = mpc.observed(eye, tick, cam.capture() if tick % 2 == 0 and tick >= hit-32 else None, q)
            self.sensed |= inbound(x)
            model.engine.observe(tick, eye.eye, q, ref.copy())
            sent = np.asarray(original["actions"][tick])
            ref = np.clip(ref + stride*sent, lo, hi)
            a.exchange(sent, mpc.FAR_COMMAND)
        original_events = [e for e in original["events"] if e["time"] <= a.data.time+1e-9]
        if digest_json(a.events) != digest_json(original_events):
            raise ValueError("Predeparture physical action tape did not reproduce")
        self.public_prefix = digest_json(a.events)
        self.eye = deepcopy(eye, {id(rig): rig})
        self.ref = ref.copy()
        self.real = State(a)
        engine = model.engine
        self.encoders = (engine.previous.copy(), engine.rate.copy(), bool(engine.bounced), engine.tick)
        self.body = mujoco.MjData(engine.body_model)
        mujoco.mj_copyData(self.body, engine.body_model, engine.body_state)

    def restore(self, a, model, rig):
        self.real.restore(a)
        model.begin()
        eye = deepcopy(self.eye, {id(rig): rig})
        q, v, bounced, tick = self.encoders
        e = model.engine
        e.previous, e.rate, e.bounced, e.tick = q.copy(), v.copy(), bounced, tick
        e.reference = self.ref.copy()
        e.tracker, e.latest = eye.eye, eye.eye.latest
        mujoco.mj_copyData(e.body_state, e.body_model, self.body)
        return eye, self.ref.copy(), self.sensed


def play(a, model, res, snapshot, row, old, ready, seed):
    cam, rig, stride, lo, hi = res
    eye, ref, sensed = snapshot.restore(a, model, rig)
    home = np.asarray(ready)
    handoff = row["departure_tick"] + 2
    leg = row["leg"] + 1
    pick = live.draw

    def change(card, index, protocol, rng):
        return pick(dict(card, reply_seed=seed) if index == leg else card, index, protocol, rng)

    done, commands, owned, takeover = False, [], False, None
    t0 = row["impact_tick"] - 2
    with patch.object(live, "draw", change):
        for tick in range(t0, handoff+300):
            q = a.joints
            x = mpc.observed(eye, tick, cam.capture() if tick % 2 == 0 else None, q)
            sensed |= inbound(x)
            if tick < handoff:
                action = np.asarray(old["actions"][tick])
                model.engine.observe(tick, eye.eye, q, ref.copy())
            else:
                if takeover is None:
                    takeover = dict(tick=tick, q=q.tolist(), ref=ref.tolist())
                desired = home
                if x is not None and (x[4] < -.9 or (sensed and x[1] < .50)):
                    target, confidence = model.raw.get(x, q)
                    if target is not None:
                        desired = home + confidence*(target-home)
                desired = model.step(tick, eye.eye, x, q, ref, desired)
                action = np.clip((np.clip(desired,lo,hi) - ref)/stride, -.8, .8)
            ref = np.clip(ref+stride*action,lo,hi)
            commands.append(action.tolist())
            _, _, done, cap, inf = a.exchange(action, mpc.FAR_COMMAND)
            if any(e["kind"] == "legal_return" and e["side"]==1 for e in inf["events"]):
                delivered = a.point_placements()["delivered"]
                if len(delivered) > leg and delivered[leg]["near_landed"]:
                    owned = True
                    break
            if done or cap:
                break
    if not owned and not done:
        raise ValueError("Actual next-reply point was censored")
    if not any(e.get("collider")=="learner_racket" and digest_json(e)==row["departure_sha256"]
               for e in a.events):
        raise ValueError("Draw or receiver changed the original outgoing departure")
    d = a.point_placements()
    if len(d["requested"]) <= leg or any(e["target"] != list(change(row["card"],j,"random",None)[0])
                                  for j,e in enumerate(d["requested"])):
        raise ValueError("Own adopted stroke did not receive the unmodified draw")
    received = d["delivered"][leg] if len(d["delivered"]) > leg else None
    ray = received.get("receive_line") if received else None
    return dict(seed=seed, target=d["requested"][leg]["target"], answered=received is not None,
                own_next=owned, touched=bool(received and received["near_contact"]),
                height=projected_height(ray) if ray is not None else None,
                incoming=received["far_contact"] if received else None,
                bounce=received["near_bounce"] if received else None,
                native_sha256=digest_json(a.events), native_prefix_sha256=snapshot.public_prefix,
                takeover=takeover, memory=model.priors.digest(), commands=commands,
                start_tick=t0, prior_own_landed=d["delivered"][leg-1]["near_landed"],
                terminal=None if owned else a.events[-1].get("reason"))


def run(output):
    d = _read(PLAN)
    if d["previous_matrix_sha256"] != _hash(MATRIX) or any(_hash(Path(p))!=v for p,v in d["sources"].items()):
        raise ValueError("Frozen plan or receiver input changed")
    doc = _read(output) if output.exists() else dict(plan_sha256=_hash(PLAN), rows=[], complete=False)
    if doc["plan_sha256"] != _hash(PLAN):
        raise ValueError("Different ongoing experiment")
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    for case in d["cases"][len(doc["rows"]):]:
        branches = {}
        for name in ("low_risk","high_risk"):
            row = case[name]
            f = Path("artifacts") / (row["source"]+".json.gz")
            if _hash(f)!=d["source_receipts"][row["source"]]:
                raise ValueError("Real scored source changed")
            source = _read(f)
            old = source["rows"][row["row"]]
            with mpc.centered(source["ready"]):
                a = live.arena(protocol="random")
                res = mpc.resources(a)
                with patch.object(mpc,"Fragments",lambda p,e: Carried(p,e,practice=[course.PRACTICE],held=set(IDS))):
                    player = mpc.MPC(mpc.BANK, raw, variant="elastic", budget=28)
                try:
                    check = ReceiverSnapshot(a,player,row["card"],old,row["impact_tick"]-2,source["ready"],res)
                    branches[name] = [play(a,player,res,check,row,old,source["ready"],seed)
                                      for seed in d["reply_seeds"]]
                finally:
                    player.close()
                    res[0].close()
                    a.close()
        for a,b in zip(*branches.values(),strict=True):
            if a["target"] != b["target"] or a["memory"]!=b["memory"]:
                raise ValueError("Two scopes had different opponent requests or issued memory")
        doc["rows"].append(dict(release=case["id"], branches=branches))
        doc["complete"] = len(doc["rows"])==len(d["cases"])
        _write(output,doc)
        print(case["id"],[(sum(a["own_next"] for a in x),sum(a["answered"] for a in x),
                          [a["height"] for a in x]) for x in branches.values()],flush=True)


def summarize(source):
    data = _read(source)
    rows = []
    for r in data["rows"]:
        a,b = (r["branches"][k] for k in ("low_risk","high_risk"))
        rows.append(dict(id=r["release"], paired=[[x["own_next"],y["own_next"]] for x,y in zip(a,b,strict=True)],
                         status=[[x["answered"],y["answered"]] for x,y in zip(a,b,strict=True)],
                         height=[[x["height"],y["height"]] for x,y in zip(a,b,strict=True)],
                         prior_landed=[[x["prior_own_landed"],y["prior_own_landed"]] for x,y in zip(a,b,strict=True)]))
    if not rows:
        return
    a = np.array([p for r in rows for p in r["paired"]])
    _write(ROOT/"receiver-report.json",dict(rows=rows, complete=data["complete"],
            own=a.sum(axis=0).tolist(), paired_wins=[int(sum(a[:,0]&~a[:,1])),int(sum(a[:,1]&~a[:,0]))],
            source_sha256=_hash(source)))
    print(a.sum(axis=0), flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("plan","run","report"))
    p.add_argument("--out",type=Path,default=ROOT/"receiver-live.json.gz")
    a=p.parse_args()
    {"plan": lambda _:plan(), "run":run, "report":summarize}[a.phase](a.out)

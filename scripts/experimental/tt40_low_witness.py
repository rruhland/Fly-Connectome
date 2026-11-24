"""Reissue the twice-rewarded lift and wait for the actual opponent's next return."""
import argparse
from pathlib import Path

import numpy as np

from . import tt40_low_lift as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt40_low_pulse import MOVES, Pulse, pulse, source_cases


def run(args):
    source = _read(core.SCHOOL)
    training = _read(args.practice)
    pulse_id = int(Pulse(training, minimum=2).rank[0])
    earned = {
        r["card"]["id"]: r for r in training["rows"]
        if r["trials"][pulse_id]["legal"]
    }
    core.joint.RECEIVE_Y = source["depth"]
    surface = core.setup(source["depth"])
    model = _read(core.PUPIL)
    out = dict(
        practice=str(args.practice), practice_sha256=_hash(args.practice),
        source=str(core.SCHOOL), source_sha256=_hash(core.SCHOOL),
        pulse_id=pulse_id, pulse=MOVES[pulse_id], protocol="random", rows=[],
    )
    with core.centered(source["ready"]):
        world = core.arena()
        axes = world._arm_ids[1][2]
        stride = world._joint_increments[axes] * world.control_dt / world.model.opt.timestep
        lower = world._joint_low[axes]
        upper = world._joint_high[axes]
        try:
            for card,row,decision in source_cases(source):
                if card["id"] not in earned:
                    continue
                witness = earned[card["id"]]
                o = np.array(witness["camera"])
                tick = witness["tick"]
                world.sample_clearance(live=args.cad)
                world.launch(row["shot"],card=card)
                world.partner.reset(seed=card["seed"])
                actor = core.joint.JointPolicy(model|{"ready":source["ready"]},stride,lower,upper)
                for u in row["actions"][:tick]:
                    z=np.asarray(u)
                    world.exchange(z,core.clinic.FAR)
                    actor.reference=np.clip(actor.reference+z*stride,lower,upper)
                if digest_json(world.events) != digest_json([e for e in row["events"] if e["step"] <= tick]):
                    raise RuntimeError("Physical practice prefix differs")
                v0 = len(world.events)
                n = max(1,round((o[0]+.075)/world.control_dt))
                motion = pulse(*core.clinic.geometry(surface,o),np.asarray(decision["action"]),o[0],
                               world.control_dt,surface.ready,MOVES[pulse_id],n=n)
                actions = []
                for k in range(150):
                    goal = motion[k] if k < len(motion) else surface.ready
                    action = actor.drive(goal,world.joints.copy(),.8)
                    actions.append(action.tolist())
                    _,_,done,cap,_ = world.exchange(action,core.clinic.FAR)
                    if done or cap or core.audit(world.events,opening_side=1)["opponent"] > 1:
                        break
                score = core.audit(world.events,opening_side=1)
                target = witness["trials"][pulse_id]["events"]
                if digest_json(world.events[v0:v0+len(target)]) != digest_json(target):
                    raise RuntimeError("Physical first receiving attempt differs from practice")
                if score["learner"] != 1:
                    raise RuntimeError("Previously rewarded pulse did not score")
                out["rows"].append(dict(
                    card=card,tick=tick,camera=o.tolist(),verified_practice_events=True,score=score,
                    opponent_answered=bool(score["opponent"] > 1),actions=actions,
                    events=world.events,motor=world.motor_audit(),clearance=world.clearance_result(),
                ))
                _write(args.output,out)
                print(card["id"],score["learner"],score["opponent"],world.clearance_result()["penetration_samples"],flush=True)
        finally:
            world.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--practice",type=Path,default=core.ROOT/"tt40/low-pulse-practice.json.gz")
    p.add_argument("--cad",action="store_true")
    run(p.parse_args())

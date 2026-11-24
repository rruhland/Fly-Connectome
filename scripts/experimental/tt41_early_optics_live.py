"""Live painter-independent paired optical ablation on the existing actuator policy."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc, tt41_early_optics as optics
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_visual_inertia import LearnedOptics

PLAN=Path("artifacts/tt39/endurance-plan.json")


def run(args):
    deck=_read(args.plan)["assessment"][args.start:args.start+args.limit]
    with np.load(optics.DEVELOP) as f:
        if set(f["groups"][:,0])&{c["id"] for c in deck}:
            raise ValueError("Promised release-disjoint optical test")
    prior=mpc.BANK
    ready=_read(prior.with_suffix(".json"))["ready"]
    with np.load(prior) as bank:
        raw=mpc.DeepField(bank,"local")
    launches=mpc.measured_launches(args.feeds)
    doc=dict(held=list(deck),plan_sha256=_hash(args.plan),bank_sha256=_hash(prior),
             optic_sha256=_hash(optics.CHECKPOINT),estimator_sha256=_hash(Path(optics.__file__)),
             cap=args.cap,protocol="full random",mode="same frozen elastic MPC, only camera readout changes",
             rows=[])
    with mpc.centered(ready):
        a=mpc.arena()
        camera,rig,stride,lo,hi=mpc.resources(a)
        body=mpc.MPC(prior,raw,variant="elastic")
        try:
            for c in deck:
                arms=[]
                for alternate in (False,True):
                    monitors=[]
                    def factory(tracker, path):
                        if not alternate:
                            return LearnedOptics(tracker,path)
                        estimate=optics.EarlyReadout(path)
                        monitors.append(estimate)
                        return LearnedOptics(tracker,estimate)
                    with patch.object(mpc,"LearnedOptics",factory):
                        r=mpc.play(a,camera,rig,stride,lo,hi,ready,c,launches[c["id"]],raw,body,
                                   optics.CHECKPOINT,"frozen",args.cap)
                    r["inserted_optical_cpu_s"]=sum(m.additional_cpu for m in monitors)
                    r["new_optical"]=list(n for m in monitors for n in m.causal_notes)
                    arms.append(r)
                before,after=arms
                cut=next((k for k,(u,v) in enumerate(zip(before["actions"],after["actions"])) if u!=v),None)
                doc["rows"].append(dict(card=c,stock=before,early=after,first_action_change=cut))
                _write(args.output,doc)
                print(c["id"],before["score"]["learner"],after["score"]["learner"],cut,
                      len(after["new_optical"]),round(after["inserted_optical_cpu_s"],4),flush=True)
        finally:
            body.close();camera.close();a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--plan",type=Path,default=PLAN)
    p.add_argument("--cap",type=int,default=24)
    p.add_argument("--start",type=int,default=6)
    p.add_argument("--limit",type=int,default=3)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    run(a)

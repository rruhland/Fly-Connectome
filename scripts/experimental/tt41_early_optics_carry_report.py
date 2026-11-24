"""Score the optical-only carry trial without mistaking changed memory for a matched action."""
import argparse
import hashlib
from ast import literal_eval
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt41_early_optics_carry import CONTROL


def summarize(tape, output):
    current=_read(tape)
    source=_read(CONTROL)
    checkpoint=tape.with_suffix("").with_suffix(".npz")
    if current["cards"]!=source["cards"] or _hash(CONTROL)!=current["baseline_sha256"]:
        raise ValueError("Course or original comparator changed")
    if not current["rows"]:
        raise ValueError("No completed matched point yet")
    if current["checkpoint_sha256"]!=_hash(checkpoint):
        raise RuntimeError("Read occurred across two point-boundary updates or checkpoint changed")
    with np.load(checkpoint) as model:
        if model["serial"]!=len(current["rows"]):
            raise ValueError("The checkpoint serial and completed recorded points differ")
        digest=hashlib.sha256()
        for state,goals,tag in zip(model["states"],model["goals"],model["tags"],strict=True):
            digest.update(np.asarray(state,dtype="<f8").tobytes())
            digest.update(np.asarray(goals,dtype="<f8").tobytes())
            digest.update(repr(literal_eval(tag)).encode())
        if digest.hexdigest()!=current["rows"][-1]["memory"]["after"]["state_sha256"]:
            raise ValueError("The checkpoint's own memory-sequence digest did not roundtrip")
    rows=[]
    for c,old,new in zip(current["cards"],source["rows"],current["rows"]):
        recorded=new["first_action_diff_from_archived_stock"]
        exact=next((i for i,(a,b) in enumerate(zip(old["actions"],new["actions"])) if a!=b),None)
        if exact!=recorded:
            raise ValueError("First actual action difference does not reproduce")
        if exact is None and old["actions"]!=new["actions"]:
            raise ValueError("One version ended with an unpaired action tail")
        to=len(new["actions"]) if exact is None else exact
        near=[]; both=[]; after=[]; answered=[]
        for r in (old,new):
            actual=audit(r["events"],opening_side=1)
            if digest_json(actual)!=digest_json(r["score"]):
                raise ValueError("Physical score mismatch")
            if any(k["draw"]!="random" for k in r["placements"]["requested"]):
                raise ValueError("Non-random opponent draw")
            near.append(actual["learner"])
            both.append(actual["returns"])
            after.append(r["memory"]["after"]["rows"])
            answered.append(actual["opponent"])
        prefix=[digest_json([e for e in r["events"] if e["step"]<=to]) for r in (old,new)]
        if exact is None and prefix[0]!=prefix[1]:
            raise ValueError("Identical commands did not yield identical actual events")
        rows.append(dict(card=c,own=near,alternating=both,opponent=answered,
                         paired_prefix_physical_events_exact=prefix[0]==prefix[1],
                         first_different_tick=exact,same_starting_memory=old["memory"]["before"]==new["memory"]["before"],
                         memory_after=after,
                         cpu_s=[old["runtime_cpu_s"],new["runtime_cpu_s"]],
                         added_optical_cpu_s=new["inserted_optical_cpu_s"],
                         exposure_count=(len(new["actions"])+1)//2))
    deltas=[r["own"][1]-r["own"][0] for r in rows]
    _write(output,dict(source=str(tape),source_sha256=_hash(tape),baseline_sha256=_hash(CONTROL),
        checkpoint_sha256=_hash(checkpoint),completed_count=len(rows),planned_count=len(current["cards"]),
        complete=(len(rows)==len(current["cards"])),rows=rows,
        total_own=np.sum([r["own"] for r in rows],axis=0).tolist(),
        total_alternating=np.sum([r["alternating"] for r in rows],axis=0).tolist(),
        longest_own=np.max([r["own"] for r in rows],axis=0).tolist(),
        longest_alternating=np.max([r["alternating"] for r in rows],axis=0).tolist(),
        total_cpu_s=np.sum([r["cpu_s"] for r in rows],axis=0).tolist(),
        signs=dict(Counter("win" if y>0 else "loss" if y<0 else "tie" for y in deltas)),
        optical_cpu_s=sum(r["added_optical_cpu_s"] for r in rows),
        exposures=sum(r["exposure_count"] for r in rows)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("tape",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    summarize(a.tape,a.output)

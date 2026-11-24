"""Scorer-only error at the actual early strike planes in three recorded rising receipts."""
import argparse
from pathlib import Path

import numpy as np

from . import tt41_early_optics as model
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

SOURCE=Path("artifacts/tt40/long-low-diverse-optical.json.gz")
CAPTURE=Path("artifacts/tt41/early-optics-rising30.npz")


def audit(output):
    with np.load(CAPTURE) as f:
        sensor,groups=f["sensor"],f["groups"]
    pred,flags,turn,cpu,n=model.playback(sensor,groups,model.CHECKPOINT,
                                       dict(threshold=.0042,h_sigma=.0031))
    doc=_read(SOURCE)
    casts={p:_read(SOURCE.with_name(p)) for p in doc["source_sha256"]}
    rows=[]
    for key in np.unique(groups,axis=0):
        release,seed,reply=key.tolist()
        matching=next(g for g in doc["groups"] if g["card"]["id"]==release and
                       g["card"]["seed"]==seed and g["card"]["reply_seed"]==reply)
        branch=next(row for row in casts[matching["origin"]]["rows"]
                    if row["id"]==release and row["leg"]==matching["leg"])
        validated=next(r for r in branch["wins"] if digest_json(r["events"])==matching["events_sha256"])
        offer=validated["placements"]["delivered"][matching["leg"]]
        strike=offer["near_contact"]
        actual_t=float(strike["time"])
        actual_xyz=np.array(strike["ball"])
        case=np.flatnonzero(np.all(groups==key,axis=1))
        first=int(case[np.searchsorted(sensor[case,0],offer["near_bounce"]["time"])])
        first_planes=[]
        for k in case:
            if k<first-1 or sensor[k,0]+.008>=actual_t:
                continue
            t=sensor[k,0]
            current=np.r_[sensor[k,4:10]]
            values=[("raw",current,bool(sensor[k,11]))]+[
                (name,pred[j,k],bool(flags[j,k])) for j,name in enumerate(("stock","early"))]
            receipts={}
            for name,x,flag in values:
                future=model.plane(x,t,flag,depth=actual_xyz[1])
                receipts[name]=dict(bounced=flag,physical_readout=None if not np.isfinite(x).all() else x.tolist(),
                                    errors=None if future is None else dict(
                                        signed_time_ms=float((t+future[0]-actual_t)*1000),
                                        signed_xz_mm=((future[1:3]-actual_xyz[[0,2]])*1000).tolist()))
            first_planes.append(dict(relative_k=int(k-first), time_s=float(t),**receipts))
        rows.append(dict(id=release,leg=matching["leg"],bounce_time=offer["near_bounce"]["time"],
                         contact_time=actual_t,contact_position=actual_xyz.tolist(),reads=first_planes))
    _write(output,dict(source_sha256=_hash(SOURCE),capture_sha256=_hash(CAPTURE),
        method="Actual physical plane is evaluator-only; neither timing nor plane was supplied to the causal estimator.",
        rows=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)

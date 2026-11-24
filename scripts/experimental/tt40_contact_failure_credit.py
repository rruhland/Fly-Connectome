"""Correct a contact effect model with a played, still-scoreless trial."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt40_contact_inverse import OUT, physical, project

LIMITS=Path("artifacts/tt38-body-relative/servo-effects.npz")


def update(folder,output):
    prior=folder/"inverse.json"
    incumbent=folder/"practice-inverse.json.gz"
    stock=folder/"paired-camera.json.gz"
    doc=_read(prior)
    old={r["card"]["id"]:r["trials"]["raw"] for r in _read(stock)["rows"]}
    own=_read(incumbent)
    if own["mode"]!="inverse" or own["checkpoint_sha256"]!=_hash(prior):
        raise ValueError("The failure was not earned with the declared model")
    with np.load(LIMITS) as limits:
        step,lower,upper=limits["step"]/.8,limits["lower"],limits["upper"]
    updates=[]
    for c,r in zip(own["cards"],own["rows"],strict=True):
        before=old[c["id"]]
        if r["score"]["learner"] or not r["modified_ticks"] or r["first_camera_contact"] is None or before["impact"] is None:
            continue
        a=r["first_causal_tick"]
        if a!=before["at"]:
            raise ValueError("The compared failed readout started at a different optical event")
        ref=np.asarray(own["ready"][:5])
        commands=[]
        for i,u in enumerate(r["actions"]):
            ref=np.clip(ref+step*np.array(u[:5]),lower,upper)
            if i>=a:
                commands.append(ref)
        after=dict(at=a,impact=r["first_camera_contact"],reference=commands)
        measured=project(before,after)
        if measured is None:
            continue
        theta,rmse,frames=measured
        vb,va=(np.asarray(e["impact"]["outgoing"]) for e in (before,after))
        updates.append(dict(context=r["context"],executed_contrast=theta.tolist(),measured_change=(va-vb).tolist(),
                            observed_baseline=vb.tolist(),frames=frames,projection_error_rad=rmse,release_documentation=c["id"]))
    if len(updates)!=1 or updates[0]["release_documentation"]!=own["cards"][0]["id"]:
        raise ValueError("Single still-unresolved practice feedback must match the frozen first-row recheck")
    doc.update(owned_secants=updates,feedback_sources={p.name:_hash(p) for p in (prior,incumbent,stock,LIMITS)},
               credit="One actual clipped command difference and its signed stereo response. Both recorded trials scored zero.")
    _write(output,doc)
    print("first step failed, stereo change",updates[0]["measured_change"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("update","recheck"))
    p.add_argument("output",type=Path)
    p.add_argument("--folder",type=Path,default=OUT)
    args=p.parse_args()
    if args.phase=="update":
        update(args.folder,args.output)
    else:
        physical(args.folder/"own-failure-updated.json","practice","inverse",args.output,1)

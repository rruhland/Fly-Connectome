"""Read first-contact action contrasts directly from before/after RGB in real tape pairs."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt36_impact import contacts
from .tt36_impact_refine import Frames
from .tt37_opponent import arena

ROOT = Path("artifacts")
OUT = ROOT / "tt40-contact-effect"
STOCK = ROOT / "tt39/hindsight-practice.json.gz"
CONTRAST = ROOT / "tt39/hindsight-verification.json.gz"
CAMERA = ROOT / "tt36/impact-model.json"
DT = 1 / 60


def gate(row):
    """The common visible pre-intervention stage, not any contact location."""
    for i, sample in enumerate(row["sensory"]):
        x = sample["x"]
        if x is not None and x[4]<-.9 and -1.05<x[1]<-.545:
            return i, np.r_[x,sample["q"][:5],np.asarray(sample["ref"][:5])-sample["q"][:5]]
    return None


def observe(a, renderer, rig, stride, lo, hi, ready, card, row, at, calibration):
    a.launch(row["shot"],card=card)
    a.partner.reset(seed=card["seed"])
    rgb=Frames(rig)
    rig.begin()
    q=[]
    refs=[np.asarray(ready)]
    for i, u in enumerate(row["actions"]):
        enc=a.joints
        if at-11 <= i < at+39:
            rgb.observe(renderer.capture(),i*DT,enc)
        a.exchange(u,FAR_COMMAND)
        q.append(enc.tolist())
        refs.append(np.clip(refs[-1]+stride*np.asarray(u),lo,hi))
    if digest_json(a.events)!=digest_json(row["events"]):
        raise ValueError("Replay disagreed with recorded physical point")
    labels=[e for e in rgb.estimates(calibration) if .025 < e["time"]-at*DT < .43]
    first=labels[0] if labels else None
    last=min(len(q),at+34)
    # Simulator contact metadata is separated here and is never used to choose a fit or controls.
    audit=contacts(row)
    return dict(stereo=[[t,p.tolist(),None if b is None else b.tolist()] for t,p,b in rgb.stereo],
                at=at,q=q[at:last],reference=np.asarray(refs[at+1:last+1]).tolist(),impact=first,
                score=row["score"]["learner"],native_evaluation=audit[:1],
                event_sha256=digest_json(a.events), action_count=len(q))


def collect(output):
    fixed, paired=(_read(p) for p in (STOCK,CONTRAST))
    if paired["practice_sha256"]!=_hash(STOCK):
        raise ValueError("Changing a practice tape invalidates the paired intervention")
    source={c["id"]:(c,r) for c,r in zip(fixed["cards"],fixed["rows"],strict=True)}
    calibration=_read(CAMERA)["models"]["60"]
    ready=fixed["ready"]
    result=dict(sources={p.name:_hash(p) for p in (STOCK,CONTRAST,CAMERA)},ready=ready,rows=[],
                label="Stereo and monocular physical observations only, no private physics trials used in this fit.")
    with centered(ready):
        a=arena(protocol="random")
        renderer,rig,stride,lo,hi=resources(a)
        try:
            for pair in paired["rows"]:
                c, baseline=source[pair["id"]]
                stage=gate(baseline)
                if stage is None:
                    continue
                at,context=stage
                seen={digest_json(baseline["actions"])}
                modes={"raw":baseline}
                for index,trial in enumerate(pair["results"]):
                    for mode in ("causal","hindsight"):
                        row=trial[mode]["physical"]
                        if row is None:
                            continue
                        code=digest_json(row["actions"])
                        if code in seen:
                            continue
                        if row["actions"][:at] != baseline["actions"][:at]:
                            raise ValueError("Training pair does not share initial encoder and flight state")
                        seen.add(code)
                        modes[f"{index}-{mode}"]=row
                pass_={}
                for tag,tape in modes.items():
                    measurement=observe(a,renderer,rig,stride,lo,hi,ready,c,tape,at,calibration)
                    pass_[tag]=measurement
                    print(c["id"],tag,measurement["score"],None if measurement["impact"] is None else measurement['impact']['outgoing'],flush=True)
                result["rows"].append(dict(card=c,context=context.tolist(),trials=pass_))
                _write(output,result)
        finally:
            renderer.close()
            a.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path,default=OUT/"paired-camera.json.gz",nargs="?")
    a=p.parse_args()
    collect(a.output)

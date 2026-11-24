"""Verify real departure labels and count matched physical conversions."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_joint_report import read_complete, replay_case
from .tt40_contact_inverse import Effect, OUT, SCALE, STOCK, VEL


def validate(out):
    ckpt=out/"inverse.json"
    camera=out/"paired-camera.json.gz"
    packed=_read(ckpt)
    pairs=packed["records"]
    y=np.array([r["delta"] for r in pairs])
    estimates=np.empty_like(y)
    average=np.empty_like(y)
    for release in packed["groups"]:
        use=[p for p in pairs if p["release"]!=release]
        model=Effect(dict(records=use))
        mean=np.mean([p["delta"] for p in use],axis=0)
        for i,p in enumerate(pairs):
            if p["release"]!=release:
                continue
            d=np.linalg.norm(model.x-np.asarray(p["state"])/SCALE,axis=1)
            weights=np.exp(-.5*(d/1.05)**2)*model.prior*model.quality
            x=model.com
            learned=np.linalg.solve(x.T@(weights[:,None]*x)+.075*np.eye(10),x.T@(weights[:,None]*model.response))
            estimates[i]=(np.asarray(p["theta"])/.23)@learned*VEL
            average[i]=mean
    actual=_read(camera)
    photo_errors=[]
    photo_times=[]
    for r in actual["rows"]:
        for shot in r["trials"].values():
            if shot["impact"] is None or not shot["native_evaluation"] or shot["native_evaluation"][0]["outgoing"] is None:
                continue
            native=shot["native_evaluation"][0]
            photo_errors.append(np.linalg.norm(np.asarray(shot["impact"]["outgoing"])-native["outgoing"]))
            photo_times.append(abs(shot["impact"]["time"]-native["time"]))
    result=dict(contrasts=len(pairs),release_groups=len(packed["groups"]),
                held_mean_magnitude_error_mps={k:float(np.mean(np.linalg.norm(e,axis=1))) for k,e in (
                    ("zero",y),("constant_training_mean",y-average),("command_contrast_regression",y-estimates))},
                optical_audit_rows=len(photo_errors),
                optical_median_error_mps=float(np.median(photo_errors)),
                optical_median_time_error_s=float(np.median(photo_times)),
                source_sha256={p.name:_hash(p) for p in (ckpt,camera)},
                note="Entire release held out in each contrast fold. Native strike/separation is only a post-hoc camera audit.")
    _write(out/"validation.json",result)
    return result


def compare(a,b,names):
    if len(a["rows"])!=len(b["rows"]) or any(c["id"]!=d["id"] or c["seed"]!=d["seed"]
        for c,d in zip(a["cards"],b["cards"],strict=True)):
        raise ValueError("Expected exact paired opening cards")
    rows=[]
    for card,x,y in zip(a["cards"],a["rows"],b["rows"],strict=True):
        dx,dy=x["placements"]["delivered"],y["placements"]["delivered"]
        same=(not dx and not dy) or (bool(dx) and bool(dy) and all(
            digest_json(dx[0][key])==digest_json(dy[0][key]) for key in ("target","far_contact","near_bounce","receive_line")))
        contact0=x.get("first_camera_contact")
        contact1=y.get("first_camera_contact")
        rows.append(dict(id=card["id"],scores=[x["score"]["learner"],y["score"]["learner"]],
                         first_delivery_exact=same,modified_ticks=y.get("modified_ticks",0),
                         first_optical_departure=[p["outgoing"] if p else None for p in (contact0,contact1)]))
    return dict(conditions=names,first=sum(r["scores"][0]>0 for r in rows),
                improved_first=sum(r["scores"][1]>0 for r in rows),
                own=[sum(r["scores"][i] for r in rows) for i in range(2)],
                exact_deliveries=sum(r["first_delivery_exact"] for r in rows),
                wins=sum(r["scores"][1]>r["scores"][0] for r in rows),
                losses=sum(r["scores"][1]<r["scores"][0] for r in rows),
                rows=rows)


def standard(out,replay=False):
    data={key:read_complete(out/f"{key}.json.gz") for key in ("practice-inverse","practice-reversed",
          "fresh-fixed","fresh-inverse","observed-miss-carry","practice-continuations")}
    prior=_read(STOCK)
    ids={c["id"] for c in data["practice-inverse"]["cards"]}
    rows=[]
    records={r["card"]["id"]:r for r in _read(out/"paired-camera.json.gz")["rows"]}
    cards=[]
    for c,r in zip(prior["cards"],prior["rows"],strict=True):
        if c["id"] in ids:
            cards.append(c)
            obs=records[c["id"]]["trials"]["raw"]["impact"]
            rows.append(dict(r,first_camera_contact=obs))
    control=dict(prior,rows=rows,cards=cards)
    tested=dict(
        aligned=compare(control,data["practice-inverse"],["raw","fitted inverse"]),
        reversed=compare(control,data["practice-reversed"],["raw","sign reversal"]),
        fresh=compare(data["fresh-fixed"],data["fresh-inverse"],["raw","fitted inverse"]),
        observed=compare(data["fresh-fixed"],data["observed-miss-carry"],["raw","measured departure carry"]),
        continued=compare(control,data["practice-continuations"],["raw","fitted inverse then stock continuation"]),
        released_followthrough_prefix_exact=sum(a["actions"]==b["actions"][:len(a["actions"])]
            for a,b in zip(data["practice-inverse"]["rows"],data["practice-continuations"]["rows"],strict=True)),
        evidence_sha256={name:_hash(out/f"{name}.json.gz") for name in data},
    )
    if replay:
        changed=[]
        for key in ("fresh-inverse","practice-continuations","observed-miss-carry"):
            test=data[key]
            if key=="fresh-inverse":
                winners=[i for i,r in enumerate(tested["fresh"]["rows"]) if r["scores"][1]>r["scores"][0]]
            elif key=="observed-miss-carry":
                winners=[i for i,r in enumerate(tested["observed"]["rows"]) if r["scores"][1]>r["scores"][0]]
            else:
                winners=[i for i,r in enumerate(tested["continued"]["rows"]) if r["scores"][1]>r["scores"][0]]
            if winners:
                index=max(winners,key=lambda i:test["rows"][i]["score"]["learner"])
                receipt=replay_case(test["ready"],"random",test["cards"][index],test["rows"][index])
                changed.append(dict(mode=key,**receipt))
        tested["cad_replays"]=changed
    _write(out/"physical-results.json",tested)
    return tested


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("validate","physical"))
    p.add_argument("--folder",type=Path,default=OUT)
    p.add_argument("--replay",action="store_true")
    args=p.parse_args()
    print(validate(args.folder) if args.mode=="validate" else standard(args.folder,replay=args.replay)["fresh"]["own"])

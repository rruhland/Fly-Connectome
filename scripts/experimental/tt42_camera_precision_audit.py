"""Summarize individually matched impact effects and reproduce selected treatment points in CAD."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt42_camera_precision import DT, LIVE_ARCHIVE, OUT


def verdict(d, tick):
    t=tick*DT
    previous=None
    for p in d["placements"]["delivered"]:
        hit, bounce = p["far_contact"], p["near_bounce"]
        if hit is None:
            continue
        if hit["time"] > t+1e-7:
            return p
        previous=p
        contact=p["near_contact"]
        if contact is None or contact["time"]>t:
            return p
    return previous


def contrast(before,after,tick):
    if tick is None:
        return None
    a,b=verdict(before,tick),verdict(after,tick)
    if a is None or b is None:
        return None
    # Everything in this body is outcome evaluation, not a live observation.
    match=all(digest_json(a[k])==digest_json(b[k]) for k in ("far_contact","near_bounce","target"))
    r=dict(same_incoming=match,leg=[p["leg"] for p in (a,b)],
           touches=[p["near_contact"] is not None for p in (a,b)],
           landed=[p["near_landed"] for p in (a,b)])
    if match and a["receive_line"] is not None:
        r["native_receive_line"]=a["receive_line"]
    return r


def report(path,out,*, cad_enabled):
    data=_read(path)
    base=_read(LIVE_ARCHIVE)
    if data["stock_sha256"]!=_hash(LIVE_ARCHIVE):
        raise ValueError("Expected archived stock source")
    records=[]
    for i,s in enumerate(data["rows"]):
        before=base["rows"][i] if data["archived_control_verified"] else s["control_fresh"]
        rs=s["comparison"]
        item=dict(id=s["card"]["id"],**rs,first_offer=contrast(
            before,s["precision"],rs["first_changed_action_tick"]),additional_camera=s["add_camera"])
        records.append(item)
    total=lambda k:[sum(s[k][j] for s in records) for j in (0,1)]
    audit=dict(source_sha256=_hash(path),complete=data["complete"],n=len(records),
               legal_near=total("legal_near"),alternating=total("alternating"),
               longest_near=[max(s["legal_near"][j] for s in records) for j in (0,1)],
               gained=sum(s["legal_near"][1]>s["legal_near"][0] for s in records),
               lost=sum(s["legal_near"][1]<s["legal_near"][0] for s in records),
               exact_initial_control=data["archived_control_verified"],points=records,cad=[])
    _write(out,audit)
    if not cad_enabled:
        return
    ids=[i for i,r in enumerate(records) if r["legal_near"][1]>r["legal_near"][0]]
    # Baseline strict CAD has already checked nominal play. Prefer the biggest
    # two treatment gains, or audit the longest changed treatment if none gain.
    ids=sorted(ids,key=lambda i:records[i]["legal_near"][1]-records[i]["legal_near"][0],reverse=True)[:2]
    if not ids:
        changed=[i for i,r in enumerate(records) if r["first_changed_action_tick"] is not None]
        if changed:
            ids=[max(changed,key=lambda i:records[i]["legal_near"][1])]
    with centered(data["ready"]):
        a,b=arena(),arena(detail="cad")
        clear=CourtClearance(b.model,articulated=True)
        try:
            for index in ids:
                c=data["rows"][index]["card"]
                r=data["rows"][index]["precision"]
                a.launch(r["shot"],card=c)
                a.partner.reset(seed=c["seed"])
                guarded(a,clear)
                contacts=[]
                with contact_observer(a,contacts):
                    for u in r["actions"]:
                        a.exchange(u,FAR_COMMAND)
                h=digest_json(a.events)
                if h!=digest_json(r["events"]):
                    raise ValueError("Held physical action replay changed in CAD audit")
                verify_clearance(a)
                audit["cad"].append(dict(id=c["id"],scored=r["score"]["learner"],events_sha256=h,
                                         samples=a.clearance_result(),native_contacts=len(contacts)))
                _write(out,audit)
        finally:
            a.close();b.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",type=Path,default=OUT/"live.json.gz")
    p.add_argument("--output",type=Path,default=OUT/"outcomes.json")
    p.add_argument("--cad",action="store_true")
    args=p.parse_args()
    report(args.live,args.output,cad_enabled=args.cad)

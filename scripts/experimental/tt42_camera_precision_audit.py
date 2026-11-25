"""Summarize individually matched impact effects and reproduce selected treatment points in CAD."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt42_camera_precision import LIVE_ARCHIVE, OUT
from .tt42_camera_precision_live import match
from .tt42_feedback_verification import action_prefix, offer_contrast, verify_checkpoint, verify_row


def report(path,out,*, cad_enabled, sensor):
    data=_read(path)
    base=_read(LIVE_ARCHIVE)
    fit=_read(sensor)
    if (data["stock_sha256"]!=_hash(LIVE_ARCHIVE) or data["sensor_report_sha256"]!=_hash(sensor)
            or data["sensor_variant"]!=fit["selection"]["mode"]):
        raise ValueError("Stock or frozen sensor provenance changed")
    if any(_hash(Path(__file__).with_name(n))!=h for n,h in data["source_code"].items()):
        raise ValueError("Treatment source no longer matches the archived run")
    n=len(data["rows"])
    if (not n or n>len(data["cards"]) or data["complete"]!=(n==len(data["cards"]))
            or data["cards"]!=base["cards"][:len(data["cards"])]):
        raise ValueError("Paired card roster or completion receipt changed")
    fresh=data["rows"][0]["control_fresh"]
    keys=("actions","events","placements","score","memory")
    exact=all(digest_json(fresh[k])==digest_json(base["rows"][0][k]) for k in keys)
    if data["archived_control_verified"]!=exact:
        raise ValueError("Fresh control receipt does not match archived control equality")
    checkpoint=path.with_suffix("").with_suffix(".npz")
    memory=verify_checkpoint(checkpoint,[s["precision"] for s in data["rows"]],data["checkpoint_sha256"])
    records=[]
    previous=base["rows"][0]["memory"]["before"]
    for i,s in enumerate(data["rows"]):
        before=base["rows"][i] if data["archived_control_verified"] else s["control_fresh"]
        after=s["precision"]
        card=data["cards"][i]
        if s["card"]!=card or after["memory"]["before"]!=previous:
            raise ValueError("Unexpected card or discontinuous treatment memory")
        previous=after["memory"]["after"]
        for row in (before,after):
            verify_row(card,row)
        fork,prefix=action_prefix(before,after)
        rs=match(before,after)
        if rs!=s["comparison"] or rs["first_changed_action_tick"]!=fork:
            raise ValueError("Archived paired comparison differs on recomputation")
        item=dict(id=s["card"]["id"],**rs,first_offer=offer_contrast(
            before,after,fork),prefix_events=len(prefix),additional_camera=s["add_camera"])
        records.append(item)
    total=lambda k:[sum(s[k][j] for s in records) for j in (0,1)]
    audit=dict(source_sha256=_hash(path),sensor_sha256=_hash(sensor),stock_sha256=_hash(LIVE_ARCHIVE),
               checkpoint=memory,complete=data["complete"],n=len(records),
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
    p.add_argument("--sensor",type=Path,default=OUT/"practice-frozen.json")
    p.add_argument("--cad",action="store_true")
    args=p.parse_args()
    report(args.live,args.output,cad_enabled=args.cad,sensor=args.sensor)

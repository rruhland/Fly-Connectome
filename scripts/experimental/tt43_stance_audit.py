"""Independently replay the optical fixed-hub pilot and its archived stock scores."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena, draw
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt43_stance import ROOT


def compare(path, stock, output):
    d,base=(_read(p) for p in (path,stock))
    if not d.get("complete") or d["ready"]!=base["ready"] or d["mode"]!=base["mode"]:
        raise ValueError("Need a complete pilot and metadata-matched baseline")
    if d["carry_sha256"]!=base["carry_sha256"] or d["optics_sha256"]!=base["optical_sha256"]:
        raise ValueError("Baseline uses a different frozen policy history")
    controls={c["key"]:(c,r) for c,r in zip(base["cards"], base["rows"],strict=True)}
    result=dict(schema="tt43-optical-hub-replay-1",source_sha256=_hash(path),control_sha256=_hash(stock),rows=[])
    with centered(d["ready"]):
        a=arena()
        c=arena(detail="cad")
        guard=CourtClearance(c.model,articulated=True)
        try:
            for card,row in zip(d["cards"],d["rows"],strict=True):
                control,stock_row=controls[card["key"]]
                if card!=control or row["shot"]!=stock_row["shot"]:
                    raise ValueError("Mismatched releases or opponent draws")
                first=row["placements"]["delivered"][:1]
                original=stock_row["placements"]["delivered"][:1]
                common=all(same_trace(o[key],p[key]) for p,o in zip(first,original)
                           for key in ("far_contact","near_bounce")) and len(first)==len(original)
                a.launch(row["shot"],card=card)
                a.partner.reset(seed=card["seed"])
                guarded(a,guard)
                notes=[]
                with contact_observer(a,notes):
                    for cmd in row["actions"]:
                        a.exchange(cmd,FAR_COMMAND)
                verify_clearance(a)
                if (digest_json(a.events)!=digest_json(row["events"])
                    or digest_json(a.point_placements())!=digest_json(row["placements"])
                    or digest_json(audit(a.events,opening_side=1))!=digest_json(row["score"])):
                    raise ValueError("Native action replay diverged")
                draws=[r["target"] for r in a.point_placements()["requested"]]
                if any(list(draw(card,i,"random",None)[0]) != x for i,x in enumerate(draws)):
                    raise ValueError("One of the replayed opponent goals was not the assigned draw")
                r=dict(first_delivery_numeric_epsilon=1e-8,id=card["id"],stock_own=stock_row["score"]["learner"],own=row["score"]["learner"],
                       opponent=row["score"]["opponent"],event_sha256=digest_json(a.events),
                       unchanged_first_delivery=common,contacts=notes,clearance=a.clearance_result(),draws=len(draws))
                result["rows"].append(r)
                print(card["id"],r["stock_own"],r["own"],common,r["clearance"]["penetration_samples"],flush=True)
                _write(output,result)
        finally:
            a.close();c.close()
    result["complete"]=True
    _write(output,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--stock",type=Path,default=ROOT/"tt42/after12.json.gz")
    a=p.parse_args()
    compare(a.source,a.stock,a.out)

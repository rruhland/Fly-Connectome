"""Summarize scored, answered, and damaging local sensitivity transfers."""
from collections import Counter

import numpy as np

from . import tt67_action_sensitivity as tt
from .tt4_transfer_trial import _hash, _read, _write

REPORT = tt.ROOT / "report.json"


def returning(result):
    clearance = result.get("clearance", {})
    return bool(result["landed"] and not clearance.get("penetration_samples", True)
                and clearance["nearest"]["signed_m"] > 0.)


def summarize():
    plan, model, result = _read(tt.PLAN), _read(tt.MODEL), _read(tt.OUT)
    if (result["fit"] != _hash(tt.MODEL) or result["scouts"] != _hash(tt.SCOUT)
            or result["protocol"] != _hash(tt.PLAN)):
        raise ValueError("Result and sealed fit/physical assignments differ")
    targets = {r["id"]: r for r in plan["cases"]}
    modes = ("replay", "forward")
    report = dict(fit=_hash(tt.MODEL), protocol=_hash(tt.PLAN), receipt=_hash(tt.OUT),
                  complete=result["complete"], modes=modes, rows=[], counts={}, min_clearance_mm=None,
                  cad_samples=0)
    for group in ("source", "neighbor", "easy_neighbor", "low_neighbor"):
        report["counts"][group] = {m: Counter() for m in modes}
    for r in result["rows"]:
        source = targets[r["id"]]
        for t in r["trials"]:
            matched = t.get("results")
            original = t["order"]["seed"] is None
            groups = ["source"] if original else ["neighbor", r["phase"]+"_neighbor"]
            if matched is None:
                for g in groups:
                    for mode in modes:
                        report["counts"][g][mode]["no_public_gate"] += 1
                continue
            shift = np.array(t["order"]["state"])-source["state"]
            margin = float(np.linalg.norm(shift[:2]))
            m = dict(id=r["id"], phase=r["phase"], original=original, dx_cm=float(shift[0]*100),
                     dz_cm=float(shift[1]*100), shift_cm=margin*100., seed=t["order"]["seed"], modes={})
            for mode in modes:
                outcome = matched[mode]
                safe = returning(outcome)
                clear = outcome.get("clearance")
                actual = outcome.get("actual")
                arm = dict(landed=outcome["landed"], safe=safe, answered=bool(safe and outcome["answered"]),
                           matched=not outcome.get("unmatched"), touched=bool(actual),
                           contact_radius_cm=None if not actual else actual["radius"]*100.)
                m["modes"][mode] = arm
                for g in groups:
                    report["counts"][g][mode].update(
                        issued=1, touched=int(arm["touched"]), legal=int(arm["landed"]), safe=int(safe),
                        answered=int(arm["answered"]), matched=int(arm["matched"]))
                if clear is not None:
                    report["cad_samples"] += clear["samples"]
                    signed = clear["nearest"]["signed_m"]*1000
                    report["min_clearance_mm"] = min(signed, report["min_clearance_mm"] if report["min_clearance_mm"] is not None else signed)
            m["recovered"] = not m["modes"]["replay"]["safe"] and m["modes"]["forward"]["safe"]
            m["lost"] = m["modes"]["replay"]["safe"] and not m["modes"]["forward"]["safe"]
            report["rows"].append(m)
    report["native_identification_count"] = sum(m.get("worlds", 0) for m in model["rows"])
    _write(REPORT,report)
    print(report["complete"], report["counts"], flush=True)
    print([(r["id"],r["seed"],round(r["shift_cm"],2),r["recovered"],r["lost"])
           for r in report["rows"] if r["recovered"] or r["lost"]], flush=True)


if __name__ == "__main__":
    summarize()

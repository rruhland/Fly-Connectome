"""Summarize the designated current live stream and independently replay changed points."""
import argparse
from collections import Counter

from . import tt59_online_corrections as live
from .tt4_transfer_trial import _hash, _read, _write
from .tt40_long_low import same_trace

REPORT = live.carry.ROOT / "online.report.json"


def offer_for(note, deliveries):
    tick = note["tick"]
    possible = [o for o in deliveries if o.get("far_contact") and o["far_contact"]["step"] <= tick]
    return max(possible, key=lambda x: x["leg"], default=None)


def action_diff(a, b):
    return next((i for i, (x,y) in enumerate(zip(a, b)) if x != y), None)


def summary(verify):
    doc = _read(live.OUT)
    plan = _read(live.PLAN)
    if doc["plan"] != _hash(live.PLAN) or doc["operator"] != _hash(live.Path(live.__file__)):
        raise ValueError("Live stream has a different planned source")
    if len(doc["rows"]) > len(plan["cases"]):
        raise ValueError("An unassigned game entered the online stream")
    prior = _read(REPORT) if REPORT.exists() else dict(rows=[])
    verified = {(r["id"], m): a["replay"] for r in prior["rows"]
                for m, a in r["arms"].items() if "replay" in a}
    result = dict(stream=_hash(live.OUT), protocol=_hash(live.PLAN), complete=doc["complete"], rows=[],
                  scores={m: Counter() for m in live.MODES})
    for case, recorded in zip(plan["cases"], doc["rows"]):
        if recorded["card"] != case["card"] or recorded["shot"] != case["shot"]:
            raise ValueError("Ordered point cards or measured launches changed")
        card = recorded["card"]
        case_results = {}
        arms = recorded["arms"]
        stock = arms["stock"]["actions"]
        changed = any(action_diff(stock, a["actions"]) is not None for a in arms.values())
        for mode, arm in recorded["arms"].items():
            delivered = arm["placements"]["delivered"]
            offers = []
            for note in arm["repairs"]:
                if not note["issued"]:
                    continue
                offer = offer_for(note, delivered)
                if offer is None:
                    offers.append(dict(tick=note["tick"], delivered=False, contacted=False, scored=False, answered=False))
                    continue
                near = offer.get("near_contact")
                close = near and 0 <= near["step"] - note["tick"] <= live.base.H+2
                stock_delivered = arms["stock"]["placements"]["delivered"]
                prior_leg = stock_delivered[offer["leg"]] if len(stock_delivered) > offer["leg"] else None
                same = bool(prior_leg and same_trace(prior_leg["far_contact"], offer["far_contact"])
                            and same_trace(prior_leg["near_bounce"], offer["near_bounce"]))
                offers.append(dict(tick=note["tick"], leg=offer["leg"], delivered=True, contacted=bool(close),
                                   matched_stock_incoming=same, stock_scored=prior_leg["near_landed"] if same else None,
                                   scored=bool(close and offer["near_landed"]),
                                   answered=bool(close and offer["near_landed"] and len(delivered) > offer["leg"]+1),
                                   source=note["selected"]["repair_release"], candidate=note["selected"]["candidate"]))
            first_diff = action_diff(stock, arm["actions"])
            learning_diff = (action_diff(arms["frozen23"]["actions"], arm["actions"])
                             if mode == "adaptive23" else None)
            data = dict(returns=arm["score"]["learner"], delivered=len(delivered),
                        contacts=sum(o.get("near_contact") is not None for o in delivered),
                        offered_repairs=len(arm["repairs"]), issued=offers,
                        issued_contacts=sum(o["contacted"] for o in offers),
                        issued_scores=sum(o["scored"] for o in offers),
                        issued_answers=sum(o["answered"] for o in offers),
                        learned=arm["motor_lessons"], complete_point=arm["complete_point"],
                        terminating_reason=arm["events"][-1].get("reason"), first_diff=first_diff,
                        first_diff_from_frozen=learning_diff, exact_offer_changes=arm.get("same_offer_changes", []))
            if (card["id"], mode) in verified:
                data["replay"] = verified[(card["id"], mode)]
            elif verify and (first_diff is not None or mode == "stock" and changed):
                with live.school.enabled(cameras=False):
                    data["replay"] = live.game.check.replay(plan["ready"], card, arm)
            if "replay" in data:
                clear = data["replay"]["clearance"]
                data["audited_safe"] = not clear["penetration_samples"] and clear["nearest"]["signed_m"] > 0.
            case_results[mode] = data
            result["scores"][mode].update(
                points=1, returns=data["returns"], deliveries=data["delivered"], contacts=data["contacts"],
                offered_repairs=data["offered_repairs"], issued=len(offers), issued_contacts=data["issued_contacts"],
                issued_scores=data["issued_scores"], issued_answers=data["issued_answers"],
                learned=data["learned"], complete=int(data["complete_point"]),
                audited=int("replay" in data), audited_safe=int(data.get("audited_safe", False)),
                learned_sources=sum(o.get("source") is not None and o["source"] < 0 for o in offers))
        result["rows"].append(dict(id=card["id"], arms=case_results))
    _write(REPORT, result)
    print(result["complete"], result["scores"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    summary(p.parse_args().verify)

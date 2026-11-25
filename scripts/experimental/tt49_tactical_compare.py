"""Compare the tactical recall arm's actual outgoing shapes and the next deliveries."""
import argparse
from pathlib import Path

import numpy as np

from . import tt49_goal_retrieval as trial
from . import tt45_tactical_coverage as science
from . import tt45_tactical_goal as tactical
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_opportunity_map import projected_height


def outcomes(row, envelope):
    deliveries = row["placements"]["delivered"]
    departures = [e for e in row["events"] if e.get("collider") == "learner_racket"]
    built = []
    for j, r in enumerate(deliveries):
        if not r["near_landed"]:
            continue
        later = deliveries[j+1] if j+1 < len(deliveries) else None
        time = r["near_contact"]["time"]
        e = next((e for e in departures if time < e["time"] < time+.09), None)
        if e is None:
            continue
        arc = science.arc_shape(e["ball"], e["velocity"])
        ray = later["receive_line"] if later else None
        h = projected_height(ray) if ray else None
        built.append(dict(leg=j, in_region=envelope.contains(arc) if np.isfinite(arc).all() else False,
                          next_playable=None if h is None else h>=science.THRESHOLD, next_z=h,
                          arc=arc.tolist() if np.isfinite(arc).all() else None))
    return built


def first_diff(a, b):
    k = next((i for i,(u,v) in enumerate(zip(a["actions"],b["actions"])) if u!=v), None)
    if k is None:
        return None
    q, r = (w["placements"]["delivered"] for w in (a,b))
    for i,(x,y) in enumerate(zip(q,r)):
        if x["far_contact"]!=y["far_contact"] or x["near_bounce"]!=y["near_bounce"]:
            return dict(tick=k, identical_offer_claimed=False)
        hits = [o["near_contact"] for o in (x,y) if o["near_contact"]]
        if not hits or max(c["step"] for c in hits) <= k:
            continue
        z = [a_list[i+1] if i+1 < len(a_list) else {} for a_list in (q,r)]
        line = [w.get("receive_line") for w in z]
        return dict(tick=k, leg=i, identical_offer_claimed=True,
                    first_landed=[bool(o["near_landed"]) for o in (x,y)],
                    next_draw_same=None if not all(z) else z[0]["target"]==z[1]["target"],
                    next_h=[projected_height(k) if k else None for k in line])
    return dict(tick=k, identical_offer_claimed=False)


def report(source, output):
    fingerprint = _hash(source)
    control, actor = map(_read, (trial.REF, source))
    if (_hash(source) != fingerprint or actor["tactic_plan"] != _hash(trial.PLAN)
        or actor["cards"] != control["cards"][:len(actor["cards"])]):
        raise ValueError("Different or actively changing official actor snapshot")
    env = tactical.Envelope(_read(trial.GOAL))
    rows, arms = [], [[], []]
    for card, s, r in zip(actor["cards"], control["rows"], actor["rows"]):
        q = [outcomes(x,env) for x in (s,r)]
        for aggregate, additional in zip(arms,q,strict=True):
            aggregate.extend(additional)
        rows.append(dict(id=card["id"], count=[s["score"]["learner"],r["score"]["learner"]],
                         first=first_diff(s,r), live_earned=r["goal_retrieval"]["earned_online"],
                         new_goal_slots=r["goal_retrieval"]["extra_goal_seeds"],
                         stock_trace=q[0], actor_trace=q[1]))
    agg = []
    for x in arms:
        agg.append(dict(known_arcs=len(x), in_region=sum(r["in_region"] for r in x),
                        later_tracked=sum(r["next_z"] is not None for r in x),
                        later_lows=sum(r["next_playable"] is False for r in x),
                        lows_from_region=sum(r["next_playable"] is False and r["in_region"] for r in x)))
    _write(output, dict(complete=len(rows)==len(actor["cards"]), source_sha256=fingerprint,
                        opponent_and_season_reference=_hash(trial.REF), measured=agg, rows=rows))
    print(agg, [(r["id"], r["count"],len(r["live_earned"])) for r in rows], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=trial.ROOT / "official-live.json.gz")
    p.add_argument("--output", type=Path, default=trial.ROOT / "achieved-shape-report.json")
    a = p.parse_args()
    report(a.source, a.output)

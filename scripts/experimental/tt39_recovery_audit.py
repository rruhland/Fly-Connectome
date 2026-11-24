"""Diagnose action-induced reply changes and when the recovering arm gets ready."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_joint_capture import OUTPUT as TT37
from .tt37_opponent import arena

OUT = TT37.parent / "tt39-recovery"


def merge_specs(paths):
    for path in paths:
        d = _read(path)
        for c, r in zip(d["cards"], d["rows"]):
            yield path.name, d["ready"], c, r


def offer_rows(paths):
    for source, ready, card, row in merge_specs(paths):
        own = [s for s in row["score"]["shots"] if s["side"] == 1]
        deliveries = row["placements"]["delivered"]
        for i, offer in enumerate(deliveries):
            if i == 0 or offer["receive_line"] is None or i - 1 >= len(own):
                continue
            previous = own[i-1]
            d = row["placements"]["requested"][i]
            recv = offer["receive_line"]
            yield dict(
                source=source, card=[card["id"], card["seed"], card["reply_seed"]], leg=i,
                request=offer["target"], outbound_velocity=d["observed_incoming"], outbound_bounce=previous["landing_xyz"],
                outbound_seconds=offer["far_contact"]["time"]-previous["hit"],
                receive=recv["xyz"], incoming=recv["velocity"], landed=offer["near_landed"],
            )


def matched(paths, dst):
    by_key = {}
    for r in offer_rows(paths):
        by_key.setdefault((*r["card"], r["leg"]), []).append(r)
    rows = []
    for key, group in by_key.items():
        for a, b in combinations(group, 2):
            if a["request"] != b["request"]:
                raise ValueError("Same continuation drew two aims")
            va, vb = [np.r_[r["outbound_bounce"], r["outbound_velocity"]] for r in (a, b)]
            if np.array_equal(va, vb):
                continue
            pa, pb = [np.asarray(r["receive"]) for r in (a, b)]
            return_a, return_b = [r["incoming"][1] for r in (a, b)]
            rows.append(dict(key=list(key), pair=[a, b], outgoing_hold_s=abs(a["outbound_seconds"] - b["outbound_seconds"]),
                             reply_x_m=abs(pa[0] - pb[0]), reply_z_m=abs(pa[2] - pb[2]),
                             reply_forward_speed_mps=abs(return_a-return_b)))
    def distribution(key):
        values = [r[key] for r in rows]
        return dict(p50=float(np.median(values)), p90=float(np.quantile(values, .9)), maximum=float(max(values)))
    summary = {k: distribution(k) for k in ("outgoing_hold_s", "reply_x_m", "reply_z_m", "reply_forward_speed_mps")} if rows else {}
    _write(dst, dict(sources={path.name: _hash(path) for path in paths}, comparable_keys=len({tuple(r["key"]) for r in rows}),
                     paired_changed_outgoing=len(rows), summary=summary, rows=rows,
                     note="Same independently drawn target after differing own shots. Mediation by the real contacted ball, not random target prediction."))
    print(len(rows), summary, flush=True)


def changed_reply_fit(source, output):
    """Calibrate only differences under the same draw. An unseen release stays out of its fold."""
    rows = _read(source)["rows"]
    scale = np.array([.14, .19, .50, .50, .50])

    def output_shot(r):
        return np.r_[r["outbound_bounce"][:2], r["outbound_velocity"]]

    def response(r):
        return np.r_[r["outbound_seconds"], np.take(r["receive"], [0, 2]), r["incoming"]]

    groups = np.array([row["key"][0] for row in rows])
    x, y = (np.array([transform(r["pair"][1]) - transform(r["pair"][0]) for r in rows])
            for transform in (output_shot, response))
    x /= scale

    def fit(features, label):
        return np.linalg.solve(features.T @ features + 6. * np.eye(len(scale)), features.T @ label)

    forecast = np.empty_like(y)
    for group in np.unique(groups):
        test = groups == group
        forecast[test] = x[test] @ fit(x[~test], y[~test])
    cols = ["outgoing_flight_s", "next_x_m", "next_z_m", "next_vx_mps", "next_vy_mps", "next_vz_mps"]
    fitted = fit(x, y) / scale[:, None]
    _write(output, dict(
        source_sha256=_hash(source), paired_changed_return_count=len(rows), releases=len(set(groups)),
        predictors=["outgoing_bounce_x", "outgoing_bounce_y", "outgoing_vx", "outgoing_vy", "outgoing_vz"],
        responses=cols, held_mae_unchanged=dict(zip(cols, abs(y).mean(axis=0).tolist(), strict=True)),
        held_mae_fitted=dict(zip(cols, abs(forecast-y).mean(axis=0).tolist(), strict=True)),
        empirical_slopes=fitted.tolist(),
        note="Archived independent aims match post hoc only. All effect estimates await new stroke interventions. Both sides of a pair had a legal next reply."))


def readiness(t, q, t_contact, t_far, ready):
    if t_far <= t_contact:
        raise ValueError("Readiness interval is reversed")
    entering = max(0, int(np.searchsorted(t, t_contact, side="left")))
    until = int(np.searchsorted(t, t_far, side="right"))
    error = np.max(abs(q[entering:until, :5] - ready[:5]), axis=1)
    available = np.flatnonzero(error <= .18)
    return dict(first_neutral_s=(float(t[entering + available[0]]-t_contact) if len(available) else None),
                maximum_error_past_100ms_rad=float(max(error[-6:], default=0.)),
                time_far_minus_contact_s=float(t_far-t_contact))


def encoders(paths, out, *, per_source=None):
    rows = []
    count = {}
    for source, ready, card, row in merge_specs(paths):
        later = row["placements"]["delivered"][1:]
        if not later or (per_source is not None and count.get(source, 0) >= per_source):
            continue
        count[source] = count.get(source, 0) + 1
        own = [s for s in row["score"]["shots"] if s["side"] == 1]
        with centered(ready):
            a = arena(protocol="random")
            try:
                a.launch(row["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                t, q = [0.], [a.joints]
                for action in row["actions"]:
                    a.exchange(action, FAR_COMMAND)
                    t.append(float(a.data.time))
                    q.append(a.joints)
                if digest_json(a.events) != digest_json(row["events"]):
                    raise ValueError(f"Action replay diverged for {source} {card['id']}")
            finally:
                a.close()
        t, q, ready = map(np.asarray, (t, q, ready))
        def interpolate(at):
            return np.array([np.interp(at, t, column) for column in q.T])
        for i, received in enumerate(later, 1):
            t0 = own[i-1]["hit"]
            tf = received["far_contact"]["time"]
            tb = received["near_bounce"]["time"]
            qf, qb = interpolate(tf), interpolate(tb)
            rows.append(dict(source=source, id=card["id"], leg=i, landed=received["near_landed"],
                             near_contact=received["near_contact"] is not None,
                             goal=received["target"], receive=received["receive_line"],
                             neutral_at_far_rad=float(np.max(abs(qf[:5]-ready[:5]))),
                             neutral_at_bounce_rad=float(np.max(abs(qb[:5]-ready[:5]))),
                             incoming_bounce_after_far_s=float(tb-tf),
                             q_far=qf.tolist(), q_bounce=qb.tolist(),
                             **readiness(t, q, t0, tf, ready)))
        _write(out, dict(sources={p.name: _hash(p) for p in paths}, rows=rows))
        print(source, card["id"], len(later), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("matched", "encoders", "influence"))
    p.add_argument("output", type=Path)
    p.add_argument("tapes", type=Path, nargs="+")
    p.add_argument("--per-source", type=int)
    a = p.parse_args()
    if a.phase == "matched":
        matched(a.tapes, a.output)
    elif a.phase == "influence":
        if len(a.tapes) != 1:
            raise ValueError("The influence fit requires one paired-difference archive")
        changed_reply_fit(a.tapes[0], a.output)
    else:
        encoders(a.tapes, a.output, per_source=a.per_source)

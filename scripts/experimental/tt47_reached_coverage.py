"""Qualify a body-conditioned distribution of future feasible movement on earned shots."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .tt43_outgoing_coverage import design as outgoing_design
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts")
OUT = ROOT / "tt47_reached_coverage"
PACKET = ROOT / "tt45_context/earned.json"
SOURCES = tuple(ROOT / s for s in (
    "tt42/learning-48.json.gz",
    "tt39/credit-carry.json.gz",
    "tt39/credit-novel.json.gz",
    "tt39/commitment-replication-stock.json.gz",
    "tt41/early-optics-carry.json.gz",
))
DT = 1. / 60.
SEED = 470226
RIDGE = 32.
O_SCALE = np.array([.15, .16, .23, .16, .17, .16, .60])
Q_SCALE = np.array([.28, .16, .22, .25, .28])
R_SCALE = np.array([.15, .25, .13, .50, .85, .80])
# Finite solver/training floor for constant channels from all families.
GATES = {
    "min_oracle_brier_gain": .004,
    "min_forecast_rms_gain_rad": .012,
    "min_marginal_brier_gain": .003,
}


def references(row, home):
    """Commanded targets, not hidden encoders. Keep this proxy distinction explicit."""
    stride = np.array(row["motor_audit"]["target_rate_limit_rad_s"]) * DT / .8
    return np.r_[home[None], home + np.cumsum(np.asarray(row["actions"]) * stride, axis=0)]


def previous_prediction(row, sent):
    contact = sent.get("near_contact")
    if contact is None or not sent.get("near_landed"):
        return None
    impact = contact["time"] / DT
    motions = [n for n in row["searches"] if n.get("target_errors") and 0 < impact - n["tick"] < 8]
    if not motions:
        return None
    m = motions[-1]
    if abs(impact - m["tick"] - m["contact_delay_s"] / DT) > 4:
        return None
    return m["target_errors"], contact


def feature_line(line):
    p, v = np.asarray(line["xyz"]), np.asarray(line["velocity"])
    return np.r_[p, v]


def build_encoder():
    packet = _read(PACKET)
    if not packet["complete"]:
        raise ValueError("Archived encoder packet is unfinished")
    files = list(packet["inputs"])
    games = []
    for p in files:
        path = Path(p)
        if _hash(path) != packet["inputs"][p]:
            raise ValueError(f"Archived source changed: {p}")
        d = _read(path)
        games.append({c["id"]: (r, np.asarray(d["ready"])) for c, r in zip(d["cards"], d["rows"], strict=True)})
    seen = {}
    for data in packet["rows"]:
        if data["leg"] < 1:
            continue
        group = data["release"]
        row, home = games[data["arm"]][group]
        legs = row["placements"]["delivered"]
        sent, reply = legs[data["leg"]-1:data["leg"]+1]
        prev = previous_prediction(row, sent)
        if prev is None:
            continue
        outgoing, contact = prev
        refs = references(row, home)
        post_tick = contact["step"] + 12
        next_tick = data["tick"]
        if not 0 <= post_tick < next_tick < len(refs):
            continue
        context = np.asarray(data["context"])
        key = (group, *np.round(context[:11], 7), bool(data["landed"]))
        seen.setdefault(key, dict(
            release=group, flight=context[:6], recovery=context[6:11]-home[:5],
            velocity=context[11:16], sender=refs[post_tick, :5]-home[:5],
            sender_velocity=(refs[post_tick]-refs[post_tick-6])[:5],
            incoming_time=(next_tick - post_tick)*DT, outgoing=outgoing,
            hit=int(reply["near_landed"]),
        ))
    return list(seen.values()), {**packet["inputs"], str(PACKET): _hash(PACKET)}


def build_reference():
    seen = {}
    receipts = {}
    for path in SOURCES:
        d = _read(path)
        receipts[str(path)] = _hash(path)
        home = np.asarray(d["ready"])
        for c, row in zip(d["cards"], d["rows"], strict=True):
            refs = references(row, home)
            legs = row["placements"]["delivered"]
            for prev, new in zip(legs, legs[1:]):
                earlier = previous_prediction(row, prev)
                line = new.get("receive_line")
                if earlier is None or not line or not new.get("far_contact"):
                    continue
                outgoing, contact = earlier
                send_tick = contact["step"]+12
                read_tick = new["far_contact"]["step"]+8
                if not 6 <= send_tick < read_tick < len(refs):
                    continue
                flight = feature_line(line)
                key = (c["id"], *np.round(contact["ball"]+contact["incoming"], 6),
                       *np.round(flight, 6), *np.round(outgoing, 5))
                seen.setdefault(key, dict(
                    release=c["id"], flight=flight, recovery=refs[read_tick, :5]-home[:5],
                    velocity=(refs[read_tick]-refs[read_tick-6])[:5],
                    sender=refs[send_tick, :5]-home[:5],
                    sender_velocity=(refs[send_tick]-refs[send_tick-6])[:5],
                    incoming_time=(read_tick-send_tick)*DT, outgoing=outgoing,
                    hit=int(new["near_landed"]),
                ))
    return list(seen.values()), receipts


def columns(rows):
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def designs(flight, q=None, dq=None):
    z = np.clip((flight - [0., -.4, .95, 0., -4.2, .4])/R_SCALE, -4, 4)
    ball = np.c_[z, np.abs(z), np.maximum(0., -z[:, 2]), z[:, 0]*z[:, 3]]
    if q is None:
        return ball
    s = np.clip(q / Q_SCALE, -4, 4)
    v = np.clip(dq / Q_SCALE, -3, 3)
    return np.c_[ball, s, v, np.abs(s), s[:, 0]*z[:, 0], v[:, 0]*z[:, 0]]


def readout(x, y, l2=RIDGE):
    x = np.atleast_2d(x)
    at = x.mean(axis=0)
    mu = np.mean(y, axis=0)
    u = x-at
    w = np.linalg.solve(u.T @ u + l2*np.eye(u.shape[1]), u.T @ (y-mu))
    return at, mu, w


def apply(fit, x):
    at, mu, w = fit
    return mu + (x - at) @ w


def q_design(flight, time, out, send=None, speed=None):
    r = np.clip((flight - [0., -.4, .95, 0., -4.2, .4])/R_SCALE, -4, 4)
    o = np.clip((out - [0., .63, .35, 0., 1.29, 1.185, 2.30])/O_SCALE, -4, 4)
    t = np.clip((time-.75)/.24,-4,4)
    base = np.c_[r, np.abs(r), o, t]
    if send is None:
        return base
    q = np.clip(send/Q_SCALE,-4,4)
    d = np.clip(speed/Q_SCALE,-4,4)
    return np.c_[base, q, d, q*t[:, None]]


def next_neighbors(train, keys):
    if len(train) < 6:
        raise ValueError("Too few actually answered outgoing strokes")
    distance = cdist(keys/O_SCALE, train/O_SCALE, "sqeuclidean")
    rank = np.argsort(distance, axis=1, kind="stable")[:, :min(len(train), 24)]
    mass = np.exp(-.5*np.take_along_axis(distance, rank, axis=1) / .9**2)
    support = mass.sum(axis=1)
    return rank, mass/np.maximum(support[:, None], 1e-20), support


def scores(pred, y):
    n = len(y)
    pred = np.clip(pred, 0, 1)
    rank = rankdata(pred)
    hit = int(y.sum())
    auc = (float(rank[y==1].sum() - hit*(hit+1)/2)/(hit*(n-hit))
           if hit and hit < n else None)
    tail = np.argsort(pred, kind="stable")[:max(1, round(n/5))]
    return dict(brier=float(np.mean((pred-y)**2)), auc=auc,
                misses_at_riskiest_fifth=int((1-y[tail]).sum()), coverage=float(np.mean(pred)))


def uncertainty(z, group):
    ids = np.unique(group)
    by_group = [np.flatnonzero(group == i) for i in ids]
    mean_g = np.array([z[use].mean() for use in by_group])
    rg = np.random.default_rng(SEED+1)
    draw = rg.integers(len(ids), size=(6000,len(ids)))
    avg = mean_g[draw].mean(axis=1)
    return dict(release_weighted_mean=float(mean_g.mean()),
                release_bootstrap_95=np.quantile(avg,[.025,.975]).tolist(),
                releases_improved=int(np.sum(mean_g>0)))


def cv(rows):
    a = columns(rows)
    g, y, f, q, dq, z, v, time, out = (a[k] for k in (
        "release","hit","flight","recovery","velocity","sender","sender_velocity","incoming_time","outgoing"))
    groups = np.random.default_rng(SEED).permutation(np.unique(g))
    folds = np.array_split(groups, min(7,len(groups)))
    bins = {n: np.zeros(len(g)) for n in ("frequency", "flight", "same_flight_reached_body",
            "outgoing_only", "marginal_reply_only", "marginal_reply_and_reached_body")}
    q_pred = np.zeros((2,len(q),10))
    support = np.zeros(len(g))
    X0, X1 = designs(f), designs(f,q,dq)
    Xq0, Xq1 = q_design(f,time,out), q_design(f,time,out,z,v)
    Xo = outgoing_design(out)
    E = np.c_[q,dq]
    for groups in folds:
        test = np.isin(g,groups)
        train = ~test
        n = int(test.sum())
        bins["frequency"][test] = y[train].mean()
        base, body = readout(X0[train],y[train]), readout(X1[train],y[train])
        bins["flight"][test] = apply(base, X0[test])
        bins["same_flight_reached_body"][test] = apply(body, X1[test])
        raw = readout(Xo[train], y[train], l2=30.)
        bins["outgoing_only"][test] = apply(raw, Xo[test])
        qbase, qfull = readout(Xq0[train],E[train]), readout(Xq1[train],E[train])
        q_pred[0,test] = apply(qbase,Xq0[test])
        q_pred[1,test] = apply(qfull,Xq1[test])
        nbr, weight, support[test] = next_neighbors(out[train],out[test])
        k = nbr.shape[1]
        relative_reply = f[train][nbr].reshape(n*k,6)
        relative_t = time[train][nbr].ravel()
        relative_out = np.repeat(out[test],k,axis=0)
        receive = apply(qfull,q_design(relative_reply, relative_t, relative_out,
                                        np.repeat(z[test],k,axis=0),np.repeat(v[test],k,axis=0)))
        bins["marginal_reply_only"][test] = np.sum(
            weight * apply(base, designs(relative_reply)).reshape(n,k).clip(0,1),axis=1)
        bins["marginal_reply_and_reached_body"][test] = np.sum(
            weight * apply(body, designs(relative_reply,receive[:,:5],receive[:,5:])).reshape(n,k).clip(0,1),axis=1)
    metrics = {k: scores(p,y) for k,p in bins.items()}
    gains = {}
    for tag, x0, x1 in (("oracle", "flight","same_flight_reached_body"),
                        ("marginal", "marginal_reply_only","marginal_reply_and_reached_body")):
        p0, p1 = bins[x0].clip(0,1), bins[x1].clip(0,1)
        diff = (p0-y)**2-(p1-y)**2
        gains[tag+"_brier_gain"] = dict(offer_weighted_mean=float(diff.mean()), **uncertainty(diff,g))
    rms = np.sqrt(np.mean((q_pred[:,:,:5]-q)**2,axis=(1,2)))
    return dict(offers=len(g), releases=len(np.unique(g)), misses=int(len(y)-sum(y)),
                metrics=metrics, gains=gains, next_q_rmse_rad=rms.tolist(),
                q_rmse_order=["reply_elapsed_and_outgoing","plus_sending_joints"],
                sender_gain_rad=float(rms[0]-rms[1]),
                reply_kernel_support_p10_p50_p90=np.quantile(support,[.10,.5,.90]).tolist(),
                body_to_home_rms_rad=float(np.sqrt(np.mean(q**2))))


def run(output):
    sets = {"encoder": build_encoder(), "reference": build_reference()}
    r = {}
    for name, (rows,sources) in sets.items():
        a = cv(rows)
        a["sources"] = sources
        r[name] = a
        print(name, a["offers"], a["releases"], a["misses"], a["gains"], a["next_q_rmse_rad"], flush=True)
    def supported(d):
        return (d["gains"]["oracle_brier_gain"]["offer_weighted_mean"] > GATES["min_oracle_brier_gain"]
                and d["gains"]["marginal_brier_gain"]["offer_weighted_mean"] > GATES["min_marginal_brier_gain"]
                and d["sender_gain_rad"] > GATES["min_forecast_rms_gain_rad"]
                and d["gains"]["marginal_brier_gain"]["release_bootstrap_95"][0] > 0)
    output.parent.mkdir(exist_ok=True,parents=True)
    _write(output,dict(
        study="release-disjoint retrospective qualification, no issued controls",
        schema="tt47-body-joint-reply-1", source_sha256=_hash(Path(__file__)),
        data=r, gates=GATES, bounded_physical_test_recommended=all(supported(d) for d in r.values()),
        scope={
            "encoder":"Actual camera p/v and own encoders at first reported approach in TT45 exact paired tapes. Prior sender is command reference at 12 ticks after physical hit. Current body oracle is observed, not promised at deployment.",
            "reference":"Larger practice data. Next ball uses evaluator receive-line geometry and both arm states are command references, not encoder telemetry. Current proxy is at eight control ticks after the next far impact. A diagnostic upper-information comparison, not a deployable model.",
            "prediction":"Next reply distribution is a weighted empirical mixture of actual prior reply/elapsed rows, using only the previously issued virtual outgoing signature. Receiver q prediction and risk use the same mixture in both arms.",
            "statistics":"Identical splits hold out all descendants of a measured release. Brier gains positive mean lower error, release bootstrap weights each entire release equally.",
        },
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, default=OUT / "qualification.json", nargs="?")
    run(p.parse_args().out)

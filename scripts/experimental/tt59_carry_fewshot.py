"""Carry the five newly earned physical repairs into release-disjoint ordinary losses."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt59_fewshot_joints as base
from .tt4_transfer_trial import _hash, _read, _write

ROOT = base.ROOT / "carry"
PLAN = ROOT / "protocol.json"
MODEL = ROOT / "readout.npz"
OUT = ROOT / "forks.json.gz"
SOURCES = tuple(Path("artifacts") / (x + ".json.gz") for x in (
    "tt42/after12", "tt42/after36", "tt42/contingent", "tt42/no-more-practice",
    "tt44_combined/stock", "tt48_transfer/continued-64", "tt42/diverse-learning",
))
SCHOOL = base.school
mpc = SCHOOL.mpc


def freeze():
    pre = base.validate_plan()
    if not _read(base.OUT)["complete"]:
        raise ValueError("Seal the first disjoint course before carrying its credit")
    roots = _read(base.ROOT / "games.protocol.json")
    exclude = set(pre["excluded"]) | {r["id"] for r in pre["cases"]} | {r["card"]["id"] for r in roots["cases"]}
    opts = [dict(r, source=str(p)) for p in SOURCES
            for r in SCHOOL.course_rows(_read(p), exclude, "practice")]
    rng = np.random.default_rng(590757)
    opts = [opts[i] for i in rng.permutation(len(opts))]
    rows = []
    for lane in ("middle", "left", "right") * 7:
        eligible = [r for r in opts if r["id"] not in exclude and r["lane"] == lane]
        if not eligible:
            continue
        v = min(eligible, key=lambda r: (r["height"] >= 1.025, r["tick"] // 480))
        rows.append(v)
        exclude.add(v["id"])
        if len(rows) >= 12:
            break
    if len(rows) < 12:
        raise ValueError("Not enough independent recorded ordinary misses")
    ROOT.mkdir(exist_ok=True, parents=True)
    proto = {k: pre[k] for k in ("school_protocol", "school_practice", "earned", "optics", "bank", "budgets")}
    _write(PLAN, dict(
        proto, schema="tt59-next-release-carry-1", cases=rows,
        sources={str(path): _hash(path) for path in (*SOURCES, base.PLAN, base.OUT, base.MODEL,
                                                     Path(__file__), base.ROOT / "games.protocol.json")},
        scope="Disjoint recorded ordinary misses, with no TT57, earlier TT59, or prospective game release ID. Freeze on historical misses only. Original 18-lesson matched motor prior versus original 18 plus all five paired first newly earned physical successes. Both methods get the same optical/encoder ranker and four-attempt cap.",
    ))
    print("new release carry roster", [(r["id"], r["lane"]) for r in rows], flush=True)


def fit():
    p = _read(PLAN)
    seen = _read(base.OUT)
    if p["sources"][str(base.OUT)] != _hash(base.OUT) or seen["protocol"] != _hash(base.PLAN):
        raise ValueError("Changed earlier independently audited acquisitions")
    with np.load(base.MODEL) as f:
        old = {k: f[k] for k in f.files}
    keys, states, ids, failures, flat, events = [], [], [], [], [], []
    with SCHOOL.enabled(cameras=False):
        a = mpc.arena(width=64, height=48)
        try:
            axes = a._arm_ids[1][2]
            stride = a._joint_increments[axes] * a.control_dt / a.model.opt.timestep
            lo, hi = a._joint_low[axes], a._joint_high[axes]
            for row in seen["rows"]:
                move = next((r for r in row["arms"]["paired"]["trials"] if r["reward"]), None)
                if move is None:
                    continue
                cid = row["case"]["id"]
                if (cid in {r["id"] for r in p["cases"]} or row["stock"]["landed"]
                        or not move["matched_far_hit"] or not move["matched_near_bounce"]
                        or move["clearance"]["nearest"]["signed_m"] <= 0):
                    raise ValueError("Invalid or leaked second-stage lesson")
                x, ref = (np.asarray(row["optics"][k]) for k in ("x", "ref"))
                stocks = SCHOOL.atlas.reference_path(ref, row["stock"]["controls"][:base.H], stride, lo, hi)[1:] - ref
                earn = SCHOOL.atlas.reference_path(ref, move["controls"][:base.H], stride, lo, hi)[1:] - ref
                scale = base.transfer.eta(x) / base.CANONICAL
                fail = SCHOOL.warp(mpc.tail(stocks, base.H), scale, 0.)
                residual = SCHOOL.warp(mpc.tail(earn, base.H), scale, 0.) - fail
                keys.append(row["optics"]["feature"])
                states.append(x)
                ids.append(cid)
                failures.append(fail)
                flat.append(residual.reshape(-1))
                events.append(move["event_sha256"])
        finally:
            a.close()
    if not ids:
        raise ValueError("No physically rewarded credit to carry")
    # Keep the already frozen motor basis and exact fit for the original 18.
    # The new observations just add coefficients and public retrieval contexts.
    new = dict(old)
    for k, values in dict(features=keys, x=states, failures=failures, ids=ids, lesson_events=events,
                          weights=np.asarray(flat) @ old["basis"].T).items():
        new[k] = np.concatenate((old[k], np.asarray(values)), axis=0)
    new["permutation"] = np.arange(len(new["ids"]))
    new["protocol"] = _hash(PLAN)
    np.savez_compressed(MODEL, **new)
    print("carrying", ids, flush=True)


class Pair:
    def __init__(self, before, after):
        self.before, self.after = before, after

    def proposals(self, x, feature, baselines, mode):
        selected = self.before if mode == "base18" else self.after
        return selected.proposals(x, feature, baselines, "paired")


def forks(count):
    before = base.Corrections(base.MODEL)
    build = base.Corrections
    with patch.multiple(base, PLAN=PLAN, MODEL=MODEL, OUT=OUT, MODES=("base18", "carry23")):
        pair = Pair(before, build(MODEL))
        with patch.object(base, "Corrections", lambda: pair):
            base.score_forks(count)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "fit", "forks"))
    p.add_argument("--n", type=int, default=12)
    a = p.parse_args()
    if a.mode == "plan":
        freeze()
    elif a.mode == "fit":
        fit()
    else:
        forks(a.n)

"""Learn the chance of surviving the following exchange from actual completed games."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried

SCALE = np.array([0.36, 0.39, 0.12, 0.33, 0.20, 0.18, 0.9])
BASE = mpc.MPC


def examples(row):
    """Associate a previously issued private prediction with observed landing rewards."""
    last_offer = None
    last_own = None
    pairs = []
    controls = row["searches"]
    for e in row["events"]:
        if e["kind"] == "legal_return" and e["side"] == -1:
            last_offer = e["time"]
            continue
        terminal = e["kind"] == "point"
        if not terminal and (e["kind"] != "legal_return" or e["side"] != 1):
            continue
        if last_offer is None:
            continue
        # The preceding near reward provides a conservative left boundary
        # even if the next movement started before the delivery's table bounce.
        floor = last_own if last_own is not None else 0.0
        candidates = [
            n
            for n in controls
            if n.get("target_errors")
            and floor < n["tick"] / 60 < e["time"]
            and last_offer - 0.15
            < n["tick"] / 60 + n["contact_delay_s"]
            < e["time"] + 0.025
        ]
        feature = candidates[-1]["target_errors"] if candidates else None
        pairs.append((feature, not terminal))
        last_offer = None
        if not terminal:
            last_own = e["time"]
    result = []
    for i, (f, legal) in enumerate(pairs):
        if f is None:
            continue
        following = pairs[i + 1][1] if i + 1 < len(pairs) else None
        result.append(
            (
                f,
                int(legal),
                int(legal and following)
                if following is not None or not legal
                else None,
            )
        )
    return result


class Value:
    def __init__(self, rows):
        self.rows = list(rows)
        self.fit()

    def fit(self):
        if not self.rows:
            return
        self.x = np.array([r[0] for r in self.rows]) / SCALE
        self.own = np.array([r[1] for r in self.rows], float)
        self.future = np.array([np.nan if r[2] is None else r[2] for r in self.rows])

    def probability(self, goals):
        if not self.rows:
            return np.full(len(goals), 0.5)
        x = np.asarray(goals) / SCALE
        ds = np.mean((x[:, None, :] - self.x) ** 2, axis=2)
        w = np.exp(-2.3 * ds)
        # The model predicts two actual arm landings from a candidate's
        # simulated outbound, so misses and early point endings remain visible.
        mask = np.isfinite(self.future)
        future = (w[:, mask] @ self.future[mask] + 5 * 0.65) / (
            5 + w[:, mask].sum(axis=1)
        )
        own = (w @ self.own + 5 * 0.80) / (5 + w.sum(axis=1))
        return 0.43 * own + 0.57 * future


class Rally(BASE):
    def __init__(self, *args, value, **kw):
        super().__init__(*args, **kw)
        self.value = value

    def robust(self, goals, rank, nominal):
        agreement, risk = self.uncertainty(goals, nominal)
        _, legal, _, y, _ = self.trial(goals, long=True)
        if not legal.all():
            return super().robust(goals, rank, nominal)
        p = self.value.probability(y[:, 3:10])
        winner = int(np.argmin(risk - 8 * p))
        self.diag["rally_choices_changed"] += winner != int(np.argmin(risk))
        return int(rank[winner]), int(agreement[winner]), float(risk[winner])


def run(a):
    files = a.sources
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    training = {}
    for p in files:
        d = _read(p)
        for card, row in zip(d["cards"], d["rows"], strict=True):
            if card["id"] in held:
                raise ValueError("Practice release overlaps assessment")
            for x, own, following in examples(row):
                training[(card["id"], tuple(x), own, following)] = (x, own, following)
    value = Value(training.values())
    n0 = len(value.rows)

    def model(file, raw, **kw):
        return Rally(file, raw, value=value, **kw)

    def bank(file, engine):
        return Carried(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )

    base_play = mpc.play

    def play(*args, **kw):
        row = base_play(*args, **kw)
        if a.update:
            value.rows.extend(examples(row))
            value.fit()
        return row

    with (
        patch.object(mpc, "MPC", model),
        patch.object(mpc, "Fragments", bank),
        patch.object(mpc, "play", play),
    ):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=240,
        )
    d = _read(a.out)
    d.update(
        training_initial=n0,
        training_final=len(value.rows),
        sources={str(p): _hash(p) for p in files},
        continuation_value=value.rows,
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=[
            mpc.ROOT / "tt39/credit-novel.json.gz",
            mpc.ROOT / "tt38/joint-mpc-development-elastic.json.gz",
            mpc.ROOT / "tt38/joint-mpc-guarded-challenge-elastic.json.gz",
        ],
    )
    p.add_argument("--update", action="store_true")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())

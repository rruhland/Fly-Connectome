"""Locally adapt earned joint commands to the currently observed flight and body."""

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt46_continuous_season as season
from . import tt49_cumulative_memory as history
from .tt4_transfer_trial import _hash, _read, _write

course = season.course
mpc = course.mpc
ROOT = Path("artifacts/tt54_local_motor")


def reference(x):
    return x[..., 6:11] * mpc.OPEN_NORM[6:11] + x[..., 11:] * mpc.OPEN_NORM[11:]


class Library:
    def __init__(self, rows):
        self.states = np.array([r[0] for r in rows])
        self.goals = np.array([r[1] for r in rows])
        self.group = [r[2] for r in rows]
        self.relative = self.goals[:, :, :5] - reference(self.states)[:, None, :]
        self.tree = cKDTree(self.states)

    def pairs(self, x, count):
        d, ids = self.tree.query(x, k=min(len(self.states), 160))
        seen = set()
        out = []
        for distance, i in zip(d, ids, strict=True):
            if self.group[i] in seen:
                continue
            seen.add(self.group[i])
            out.append((distance, i))
            if len(out) >= count:
                break
        return out

    def generalize(self, key, i, variant):
        g = self.goals[i].copy()
        dx = key - self.states[i]
        new_ref = reference(key)
        anchored = self.relative[i] + new_ref
        if variant == "absolute":
            return g
        if variant == "relative":
            g[:, :5] = anchored
            return g
        # The donor specifies the movement family. Its neighboring examples
        # must represent other rewarded strokes with similar motor shape.
        ds, ids = self.tree.query(self.states[i], k=min(220, len(self.states)))
        delta_motor = self.relative[ids] - self.relative[i]
        motor_distance = np.mean(delta_motor[:, (2, 6, 11), :] ** 2, axis=(1, 2))
        proximity = ds**2 / 1.4**2 + motor_distance / 0.22**2
        keep = []
        groups = {self.group[i]}
        for j in np.argsort(proximity, kind="stable"):
            group = self.group[ids[j]]
            if group not in groups:
                groups.add(group)
                keep.append(j)
            if len(keep) == 32:
                break
        if len(keep) < 6:
            return g
        u = self.states[ids[keep]] - self.states[i]
        w = np.exp(-0.5 * proximity[keep])
        # Solve for a state adjustment around this demonstrated path. The
        # weights are fitted using other events only; no outcome of this
        # incoming ball is available before issuing a command.
        coeff = np.linalg.solve((u.T * w) @ u + np.eye(u.shape[1]) * 0.45, dx)
        movement = (w * (u @ coeff)) @ delta_motor[keep].reshape(len(keep), -1)
        corrected = anchored + movement.reshape(anchored.shape)
        g[:, :5] += np.clip(corrected - g[:, :5], -0.32, 0.32)
        return g


def errors(path, ref, actual):
    # What the slew-limited actuator would be sent, including the first tick.
    values = np.zeros_like(path[:, :5])
    for t, goal in enumerate(path[:, :5]):
        ref = ref + np.clip(goal - ref, -0.8 * 5.2 / 60, 0.8 * 5.2 / 60)
        values[t] = ref
    return np.array(
        [np.sqrt(np.mean((values[:n] - actual[:n, :5]) ** 2)) for n in (1, 4, 10, 18)]
    )


def evaluate():
    earlier = history.load_rows(season.CARRY)
    full, proof = history.restore_history(24)
    lib = Library(earlier)
    new = full[len(earlier) :]
    # One sample for each scored future stroke. No target stroke, adjacent
    # phases of it, or any other stroke in its point enters the donor bank.
    grouped = {}
    for row in new:
        grouped.setdefault(row[2], []).append(row)
    totals = {k: [] for k in ("absolute", "relative", "local")}
    for event in grouped.values():
        k, actual, _ = event[len(event) // 2]
        neighbors = lib.pairs(k, 7)
        for variant in totals:
            candidate = [
                errors(lib.generalize(k, i, variant), reference(k), actual)
                for _, i in neighbors
            ]
            totals[variant].append([candidate[0], np.min(candidate, axis=0)])
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(
        ROOT / "reproduction.json",
        dict(
            prior=_hash(season.CARRY),
            proof=proof,
            n=len(grouped),
            unique_games=len(set(r[1] for r in grouped)),
            ranges=[1, 4, 10, 18],
            mean_rms_rad={k: np.mean(v, axis=0).tolist() for k, v in totals.items()},
        ),
    )


class Generalized(course.Carried):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.demo = Library(self.rows)
        self.stamp = None
        self.counts = Counter()

    def priors(self, x, q, ref, limit):
        key = mpc.query(x, q, ref)
        stamp = (len(self.online), id(self.online[-1][0]) if self.online else None)
        if stamp != self.stamp:
            self.learned = Library(list(self.online)) if self.online else None
            self.stamp = stamp
        choices = [(d, self.demo, i) for d, i in self.demo.pairs(key, limit)]
        if self.learned is not None:
            choices.extend(
                (max(0.0, d - 0.10), self.learned, i)
                for d, i in self.learned.pairs(key, limit)
            )
        selected = []
        labels = set()
        for d, bank, i in sorted(choices, key=lambda r: r[0]):
            if d > 3.1 and len(selected) >= 2:
                break
            if bank.group[i] in labels:
                continue
            labels.add(bank.group[i])
            # Keep the nearest absolute exemplar from each search. The
            # remaining donors specialize their command for the current body.
            mode = "absolute" if not selected else self.variant
            selected.append((bank.generalize(key, i, mode), bank is self.learned))
            self.counts[mode] += 1
            if len(selected) == limit:
                break
        return selected


def play(a):
    Generalized.variant = a.variant
    actor = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    with (
        patch.object(course, "PLAN", season.PLAN),
        patch.object(course, "Carried", Generalized),
        patch.object(
            course.mpc, "MPC", lambda *x, **kw: season.Player(*x, coverage=actor, **kw)
        ),
    ):
        course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", choices=("metric", "local", "relative", "absolute"))
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live", deck="practice", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    a.out = ROOT / f"{a.variant}-games.json.gz"
    evaluate() if a.variant == "metric" else play(a)

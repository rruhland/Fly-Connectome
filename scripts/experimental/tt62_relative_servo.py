"""Retrieve and adapt paid body-object relationships when the current search has no legal motion."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt43_temporal_servo as servo
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

season = dose.season
course = dose.course
mpc = course.mpc
ROOT = Path("artifacts/tt62_relative_servo")
MODEL = ROOT / "relationships.npz"
SCALE = np.array([0.11, 0.077, 0.057, 0.40, 0.60, 0.66, 0.88, 0.62, 0.75, 0.83, 0.99])
LOOK = 6


def windows(samples, token):
    examples = []
    for i, (t, x, q, ref, sent) in enumerate(samples):
        if x is None or x[4] >= -0.9 or not -1.62 < x[1] < -0.22:
            continue
        k = min(i + LOOK, len(samples) - 1)
        while k > i and (
            samples[k][0] != t + k - i
            or samples[k][1] is None
            or samples[k][1][4] >= -0.9
        ):
            k -= 1
        if k <= i:
            continue
        _, _, qf, _, target = samples[k]
        examples.append((np.r_[x, q[:5]], qf[:5], target[:5], k - i, token))
    return examples


def fit():
    rows = []
    token = 0
    ids = set()
    for path in servo.SOURCES:
        meta = _read(path.with_suffix(".json"))
        if _hash(path) != meta["bank_sha256"]:
            raise ValueError("Optical practice changed")
        with np.load(path) as z:
            states, marks, targets = (z[k] for k in ("states", "marks", "targets_now"))
        start = 0
        for sample in meta["practice"]:
            ids.add(sample["card"]["id"])
            end = start + sample["frames"]
            ticks = marks[start:end, 0].astype(int)
            age = marks[start:end, 1]
            tag = np.rint((ticks * mpc.DT - age) * 480).astype(int)
            for contact in np.unique(tag[age < 0]):
                idx = np.flatnonzero((tag == contact) & (age < 0))
                log = [
                    (
                        ticks[i],
                        states[start + i, :6],
                        states[start + i, 6:11],
                        None,
                        targets[start + i],
                    )
                    for i in idx
                ]
                rows.extend(windows(log, token))
                token += 1
            start = end
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL,
        **{
            k: np.array(v)
            for k, v in zip(
                ("state", "q", "ref", "lag", "token"),
                zip(*rows, strict=True),
                strict=True,
            )
        },
    )
    _write(
        ROOT / "fit.json",
        dict(
            model=_hash(MODEL),
            paid_segments=token,
            examples=len(rows),
            ids=sorted(ids),
            sources={str(p): _hash(p) for p in servo.SOURCES},
            scope="The controller can issue a direct relationship adjustment only following a real-time search that failed to find any nominal legal movement. Retrieval sees current RGB motion and encoders. The ordinary arm decides all other frames. Thirty-two prespecified complete points with the usual broad opponent.",
        ),
    )


class Player(season.Player):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        with np.load(MODEL) as z:
            self.entries = list(
                zip(*(z[k] for k in ("state", "q", "ref", "lag", "token")), strict=True)
            )
        self.tree = cKDTree(np.asarray([a[0] for a in self.entries]) / SCALE)
        self.learned = deque(maxlen=400)
        self.reliability = {}
        self.serial_rel = len(self.entries) + 1
        keep = self.priors.keep

        def earned(tape, leg):
            self.update(1)
            latest = next(
                (
                    i
                    for i in range(len(tape) - 1, -1, -1)
                    if tape[i][1] is not None
                    and tape[i][1][4] < -0.9
                    and tape[i][1][1] < -0.9
                ),
                None,
            )
            if latest is not None:
                # Use just the approaching ball that immediately preceded this reward.
                start = latest
                while (
                    start > 0
                    and latest - start < 40
                    and not (
                        tape[start - 1][1] is not None and tape[start - 1][1][4] > 0.9
                    )
                ):
                    start -= 1
                self.serial_rel += 1
                self.learned.extend(
                    windows(tape[start : latest + 1], self.serial_rel)[::2]
                )
            return keep(tape, leg)

        self.priors.keep = earned

    def begin(self):
        super().begin()
        self.selected = set()
        self.empty_at = -100

    def update(self, won):
        for token in self.selected:
            a, b = self.reliability.get(token, (0, 0))
            self.reliability[token] = (a + won, b + 1 - won)
        self.selected.clear()

    def missed(self, tape):
        self.update(0)

    def search(self, t, x, q, ref, base):
        super().search(t, x, q, ref, base)
        note = self.notes[-1] if self.notes else {}
        self.empty_at = (
            t if note.get("tick") == t and note.get("legal_trials") == 0 else -100
        )

    def goal(self, x, q):
        state = np.r_[x, q[:5]]
        d, ii = self.tree.query(state / SCALE, k=32)
        available = [(dist, self.entries[j]) for dist, j in zip(d, ii, strict=True)]
        if self.learned:
            ds = np.linalg.norm(
                (np.asarray([r[0] for r in self.learned]) - state) / SCALE, axis=1
            )
            available.extend((ds[j], self.learned[j]) for j in np.argsort(ds)[:8])

        def penalty(item):
            a, b = self.reliability.get(item[1][-1], (0, 0))
            return item[0] + 0.45 * (b - a) / (a + b + 2)

        dist, row = min(available, key=penalty)
        if dist > 3.25:
            return None
        observed, body, target, lag, token = row
        # At the *same phase of that rewarded movement*, compensate the
        # measured difference in the two object tracks. No intercept location
        # or desired ball departure is prescribed.
        drift = x[:3] - observed[:3] + lag * mpc.DT * (x[3:6] - observed[3:6])
        _, _, linear, angular = self.geometry.pose(body, jacobian=True)
        J = np.vstack([linear, 0.085 * angular])
        correction = np.linalg.solve(
            J.T @ J + 0.053**2 * np.eye(5), linear.T @ np.clip(drift, -0.115, 0.115)
        )
        self.selected.add(token)
        return target + np.clip(correction, -0.36, 0.36)

    def step(self, t, eye, x, q, ref, base):
        ordinary = super().step(t, eye, x, q, ref, base)
        if (
            x is None
            or x[4] >= -0.9
            or not -1.62 < x[1] < -0.49
            or not 0 <= t - self.empty_at <= 4
        ):
            return ordinary
        if (
            self.notes
            and self.best is not None
            and 0 <= t - self.tick < self.lease
            and self.tick != self.empty_at
        ):
            return ordinary
        target = self.goal(x, q)
        if target is None:
            return ordinary
        out = ordinary.copy()
        out[:5] += 0.88 * np.clip(target - ordinary[:5], -0.38, 0.38)
        self.diag["relationship_servo_ticks"] += 1
        return out


def play(a):
    meta = _read(ROOT / "fit.json")
    if meta["model"] != _hash(MODEL) or set(meta["ids"]) & {
        c["id"] for c in _read(dose.PLAN)["practice"]
    }:
        raise ValueError("Changed or overlapping motor history")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    actors = []

    def make(*args, **kw):
        actor = Player(*args, coverage=rank, **kw)
        actors.append(actor)
        return actor

    with (
        enabled(),
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(mpc, "MPC", make),
    ):
        course.run(a)
    p = actors[-1]
    np.savez_compressed(
        ROOT / "new.npz",
        **{
            k: np.array([r[i] for r in p.learned])
            for i, k in enumerate(("state", "q", "ref", "lag", "token"))
        },
        reliability=np.array([[k, *v] for k, v in p.reliability.items()]),
        last_token=p.serial_rel,
    )
    d = _read(a.out)
    d.update(
        learned_relationships=_hash(ROOT / "new.npz"), base_relationships=_hash(MODEL)
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        cap=240,
        carry=season.CARRY,
        out=ROOT / "live.json.gz",
        resume=False,
    )
    a = p.parse_args()
    fit() if a.phase == "fit" else play(a)

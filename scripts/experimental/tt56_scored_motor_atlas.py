"""Recover direct actuator choices from the full positive and negative RGB-controlled game archive."""

import argparse
from collections import deque
from hashlib import blake2b
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt46_continuous_season as season
from . import tt48_practice_transfer as transfer
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

mpc = season.course.mpc
ROOT = Path("artifacts/tt56_scored_motor_atlas")
MODEL = ROOT / "actions.npz"
H = mpc.HORIZON
P_SCALE = np.array([0.095, 0.135, 0.07])
V_SCALE = np.array([0.51, 0.85, 0.83])
Q_SCALE = np.array([0.45, 0.35, 0.44, 0.45, 0.55])


def key(t, p, bounced, ref, old):
    p = np.asarray(p)
    velocity, motor = np.zeros(3), np.zeros(5)
    has_history = False
    if old is not None and 2 <= t - old[0] <= 11:
        previous_tick, prev_p, prev_ref = old
        dt = (t - previous_tick) * mpc.DT
        velocity = (p - prev_p) / dt
        motor = (ref[:5] - prev_ref[:5]) / dt
        has_history = True
    return np.r_[
        p / P_SCALE,
        velocity / V_SCALE,
        ref[:5] / Q_SCALE,
        motor / 3.5,
        float(bounced),
        2.0 * has_history,
    ]


def limits():
    plant = mpc.arena()
    try:
        ids = plant._arm_ids[1][2]
        return (
            plant._joint_increments[ids]
            * (plant.control_dt / plant.model.opt.timestep),
            plant._joint_low[ids],
            plant._joint_high[ids],
        )
    finally:
        plant.close()


def reference_path(first, commands, stride, low, high):
    delta = np.asarray(commands) * stride
    path = np.vstack((first, first + np.cumsum(delta, axis=0)))
    if path.min(0).min() < -10 or not np.isfinite(path).all():
        raise ValueError("Invalid action tape")
    if (path < low - 1e-9).any() or (path > high + 1e-9).any():
        for j, u in enumerate(delta):
            path[j + 1] = np.clip(path[j] + u, low, high)
    return path


def fit():
    ROOT.mkdir(exist_ok=True, parents=True)
    reserved = {
        c["id"] for k in ("practice", "assessment") for c in _read(season.PLAN)[k]
    }
    reserved.update(c["id"] for c in _read(transfer.PLAN)["practice"])
    inventory = _read(Path("artifacts/tt52_matched_movements/pairs.json.gz"))["sources"]
    scale = limits()
    X, U, rewards, ids = [], [], [], []
    seen = set()
    sources = {}
    for filename, checksum in inventory.items():
        path = Path(filename)
        if _hash(path) != checksum:
            raise ValueError(f"Changed source at {path}")
        d = _read(path)
        previous = len(X)
        for card, r in zip(d["cards"], d["rows"]):
            if (
                card["id"] in reserved
                or card.get("profile") != "reach"
                or any(g["draw"] != "random" for g in r["placements"]["requested"])
            ):
                continue
            refs = reference_path(d["ready"], r["actions"], *scale)
            ticks = np.array([n["tick"] for n in r["searches"]])
            for o in r["placements"]["delivered"]:
                if not o.get("near_bounce") or not o.get("far_contact"):
                    continue
                # The referee segments the practice commands. No contact
                # coordinate, time to impact, opponent draw or score is an input.
                lo = o["far_contact"]["step"]
                hi = (o["near_contact"] or r["events"][-1])["step"]
                old = None
                for j in np.flatnonzero((ticks >= lo) & (ticks < hi)):
                    n = r["searches"][j]
                    if not n.get("p"):
                        continue
                    t = n["tick"]
                    state = key(t, n["p"], n["bounced"], refs[t], old)
                    old = (t, np.asarray(n["p"]), refs[t])
                    move = mpc.tail(refs[t + 1 : t + 1 + H], H) - refs[t]
                    won = bool(o["near_landed"])
                    uid = blake2b(
                        np.round(np.r_[state, move.ravel()], 5).tobytes()
                        + bytes([won]),
                        digest_size=16,
                    ).digest()
                    if uid in seen:
                        continue
                    seen.add(uid)
                    X.append(state)
                    U.append(move)
                    rewards.append(won)
                    ids.append(card["id"])
        if len(X) != previous:
            sources[filename] = checksum
    np.savez_compressed(
        MODEL,
        keys=np.asarray(X, np.float32),
        moves=np.asarray(U, np.float32),
        reward=rewards,
        release=ids,
    )
    _write(
        ROOT / "fit.json",
        dict(
            samples=len(X),
            successes=sum(rewards),
            misses=len(X) - sum(rewards),
            release_count=len(set(ids)),
            model=_hash(MODEL),
            sources=sources,
        ),
    )


class Player(season.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(MODEL) as d:
            won = d["reward"]
            self.success = cKDTree(d["keys"][won])
            self.failed = cKDTree(d["keys"][~won])
            self.good = d["moves"][won]
            self.bad = d["moves"][~won]
            self.release = d["release"][won]
        self.earned = deque(maxlen=256)
        self.observed_fails = deque(maxlen=128)
        keep = self.priors.keep

        def earn(tape, leg):
            self.credit(tape, True)
            return keep(tape, leg)

        self.priors.keep = earn

    def begin(self):
        super().begin()
        self.recent = []
        self.previous = None

    def credit(self, tape, won):
        log = {t: (r, u) for t, _, _, r, u in tape}
        for tick, state in self.recent:
            if tick not in log:
                continue
            issued = [log[t][1] for t in range(tick, tick + H) if t in log]
            if len(issued) >= 4:
                target = mpc.tail(np.array(issued), H) - log[tick][0]
                (self.earned if won else self.observed_fails).append((state, target))
        self.recent.clear()

    def missed(self, tape):
        self.credit(tape, False)

    def offers(self, state):
        d, idx = self.success.query(state, k=32)
        candidates = [
            (float(v), self.good[i], self.release[i])
            for v, i in zip(d, idx, strict=True)
        ]
        candidates.extend((np.linalg.norm(s - state), m, -1) for s, m in self.earned)
        choices = []
        groups = set()
        for dist, m, release in sorted(candidates, key=lambda c: c[0]):
            if len(choices) >= 5 or (dist > 3.2 and choices):
                break
            if release in groups and len(groups) < 3:
                continue
            if any(np.max(np.abs(m[::4] - c[::4])) < 0.035 for c in choices):
                continue
            groups.add(release)
            choices.append(m)
        fd, i = self.failed.query(state)
        neg = self.bad[i]
        for s, m in self.observed_fails:
            dist = np.linalg.norm(s - state)
            if dist < fd:
                fd, neg = dist, m
        if choices and fd < 2.5:
            # Extrapolate in actuator space away from the nearby failed movement.
            # The optical model must validate even a previously earned alternative.
            for good in choices[:2]:
                choices.append(good + 0.4 * np.clip(good - neg, -0.38, 0.38))
        return choices

    def search(self, tick, x, q, ref, base):
        super().search(tick, x, q, ref, base)
        if not self.notes or self.notes[-1]["tick"] != tick:
            return
        n = self.notes[-1]
        state = key(tick, x[:3], n["bounced"], ref, self.previous)
        self.previous = tick, x[:3].copy(), ref.copy()
        self.recent.append((tick, state))
        if n.get("robust", 0) == 2:
            return
        paths = [ref + move for move in self.offers(state)]
        if not paths:
            return
        loss, legal, _, _, _ = self.trial(paths, long=True)
        self.diag["atlas_searches"] += 1
        ids = np.flatnonzero(legal)
        if len(ids):
            ids = ids[np.argsort(loss[ids])[:4]]
            votes, risk = self.uncertainty([paths[i] for i in ids], loss[ids])
            use = np.where(votes >= max(1, n.get("robust", 0) + 1), risk, np.inf)
            j = np.argmin(use)
            if np.isfinite(use[j]):
                self.best = paths[ids[j]]
                self.tick, self.lease = tick, mpc.LEASE
                self.diag["atlas_installed"] += 1
                n.update(atlas_votes=int(votes[j]), atlas_cost=float(use[j]))


def run(a):
    meta = _read(ROOT / "fit.json")
    if _hash(MODEL) != meta["model"]:
        raise ValueError("Changed atlas")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    players = []

    def make(*args, **kwargs):
        p = Player(*args, coverage=rank, **kwargs)
        players.append(p)
        return p

    with (
        enabled(),
        patch.object(season.course, "PLAN", dose.PLAN),
        patch.object(mpc, "MPC", make),
    ):
        season.course.run(a)
    p = players[-1]
    memory = ROOT / "new-movements.npz"
    np.savez_compressed(
        memory,
        **{
            prefix + attr: np.array([r[i] for r in rows])
            for prefix, rows in (("earned_", p.earned), ("missed_", p.observed_fails))
            for i, attr in enumerate(("keys", "moves"))
        },
    )
    d = _read(a.out)
    d.update(motor_model=_hash(MODEL), acquired=_hash(memory))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode_name", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        out=ROOT / "games.json.gz",
        mode="live",
        deck="practice",
        carry=season.CARRY,
        cap=240,
        resume=False,
    )
    a = p.parse_args()
    fit() if a.mode_name == "fit" else run(a)

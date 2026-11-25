"""Test whole, reward-earned joint approaches from earlier in the observed incoming flight."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt46_continuous_season as season
from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

mpc = season.course.mpc
ROOT = Path("artifacts/tt56_long_motor_memory")
FRAMES = 43
GATES = (0.21, -0.19)
BANK = ROOT / "practice.npz"


def packed_state(x, q, ref):
    return mpc.query(x, q, ref)


def collect(logs, group):
    """The scalar reward authorizes the movement. Labels use only observed states and applied commands."""
    result = []
    for i, (tick, x, q, ref, _) in enumerate(logs):
        if x is None or x[4] >= -0.95 or not -0.39 < x[1] < 0.43 or tick % 2:
            continue
        window = logs[i : i + FRAMES]
        time = np.array([row[0] - tick for row in window])
        if time[-1] < 17 or np.max(np.diff(time)) > 4:
            continue
        goals = np.array([row[-1] for row in window])
        # Repeat the last installed command when there is less than one full horizon.
        interpolated = np.stack(
            [np.interp(np.arange(FRAMES), time, col) for col in goals.T], axis=1
        )
        result.append((packed_state(x, q, ref), interpolated, group))
    return result


def fit():
    ROOT.mkdir(parents=True, exist_ok=True)
    samples = []
    provenance = {}
    excluded = {c["id"] for c in _read(dose.PLAN)["practice"]}
    group = 0
    plant = mpc.arena()
    axes = plant._arm_ids[1][2]
    stride = plant._joint_increments[axes] * (
        plant.control_dt / plant.model.opt.timestep
    )
    low, high = plant._joint_low[axes], plant._joint_high[axes]
    plant.close()
    for path in season.fresh.servo.SOURCES:
        meta = _read(path.with_suffix(".json"))
        if meta["bank_sha256"] != _hash(path):
            raise ValueError("Changed practice source")
        provenance[str(path)] = _hash(path)
        with np.load(path) as f:
            states, goals, marks = (f[k] for k in ("states", "targets_now", "marks"))
        start = 0
        tapes = {}
        for ep in meta["practice"]:
            if ep["card"]["id"] in excluded:
                raise ValueError("Practice overlaps prospective course")
            tape = path.with_name(ep["source"])
            if tape not in tapes:
                if _hash(tape) != meta["sources"][ep["source"]]:
                    raise ValueError("Changed issued commands")
                tapes[tape] = _read(tape)
            archive = tapes[tape]
            if archive["cards"][ep["source_row"]] != ep["card"]:
                raise ValueError("Reward and command tapes refer to different games")
            actions = archive["rows"][ep["source_row"]]["actions"]
            issued = [np.asarray(meta["ready"])]
            for u in actions:
                issued.append(np.clip(issued[-1] + stride * np.asarray(u), low, high))
            stop = start + ep["frames"]
            if not np.allclose(
                np.array(issued)[marks[start:stop, 0].astype(int) + 1],
                goals[start:stop],
                atol=1e-5,
            ):
                raise ValueError(
                    "Issued targets disagree with observed training labels"
                )
            m = marks[start:stop]
            credited = np.rint((m[:, 0] * mpc.DT - m[:, 1]) * 480).astype(int)
            for event in np.unique(credited):
                rows = np.flatnonzero(credited == event) + start
                samples_at = {int(marks[i, 0]): states[i] for i in rows}
                logs = []
                for t in range(min(samples_at), max(samples_at) + 1):
                    s = samples_at.get(t)
                    logs.append(
                        (
                            t,
                            s[:6] if s is not None else None,
                            s[6:11] if s is not None else None,
                            issued[t],
                            issued[t + 1],
                        )
                    )
                samples.extend(collect(logs, group))
                group += 1
            start = stop
    keys, goals, groups = map(np.asarray, zip(*samples, strict=True))
    np.savez_compressed(BANK, keys=keys, goals=goals, groups=groups)
    _write(
        ROOT / "fit.json",
        dict(
            samples=len(keys),
            scored_movements=len(np.unique(groups)),
            model=_hash(BANK),
            sources=provenance,
        ),
    )


class Player(season.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with np.load(BANK) as d:
            self.tree = cKDTree(d["keys"])
            self.paths, self.groups = d["goals"], d["groups"]
        self.new = deque(maxlen=768)
        self.failures = deque(maxlen=96)
        self.group = -1
        a = self.engine.twin
        ia = self.engine.actuators
        self.stride = a._joint_increments[ia] * self.engine.predict.micro
        self.low, self.high = a._joint_low[ia], a._joint_high[ia]
        keep = self.priors.keep

        def earn(history, tag):
            # A previous hit can still be in the rolling buffer at the next reward.
            # Walk backward from the final approach, then include its actually issued follow-through.
            incoming = [
                i
                for i, (_, x, *_) in enumerate(history)
                if x is not None and x[4] < -0.95 and x[1] < -0.72
            ]
            if incoming:
                last = incoming[-1]
                earlier = [
                    i
                    for i, (_, x, *_) in enumerate(history[:last])
                    if x is not None and x[4] > 0.95
                ]
                segment = history[(earlier[-1] + 1 if earlier else 0) : last + 8]
                additions = collect(segment, self.group)
                self.group -= 1
                self.new.extend(additions)
                self.diag["earned_early_samples"] += len(additions)
            self.applied = None
            return keep(history, tag)

        self.priors.keep = earn

    def begin(self):
        super().begin()
        self.applied = None
        self.reset_approach()

    def reset_approach(self):
        self.next_gate = 0
        self.approach = None

    def missed(self, history):
        if self.applied is not None:
            tick, key, reference = self.applied
            for t, _, _, _, issued in history:
                if t == tick + 4:
                    self.failures.append((key, issued[:5] - reference[:5]))
                    self.diag["failed_early_movements"] += 1
                    break

    def candidates(self, key):
        dist, ids = self.tree.query(key, min(100, len(self.groups)))
        rows = [
            (d, self.paths[i], self.groups[i]) for d, i in zip(dist, ids, strict=True)
        ]
        rows.extend((np.linalg.norm(key - k), p, g) for k, p, g in self.new)
        seen, options = set(), []
        for d, path, group in sorted(rows, key=lambda r: r[0]):
            if group in seen:
                continue
            seen.add(group)
            if d > 3.5 and options:
                break
            options.append(path)
            if len(options) >= 7:
                break
        return options

    def screen(self, tick, x, q, ref, base):
        if not self.engine.sync():
            return
        key = packed_state(x, q, ref)
        paths = self.candidates(key)
        if not paths:
            return
        t = np.arange(FRAMES)
        # Vary timing independently of which experience is retrieved.
        variants = list(paths)
        for p in paths[:4]:
            for rate in (0.89, 1.12):
                variants.append(
                    np.stack([np.interp(t * rate, t, c) for c in p.T], axis=1)
                )

        def assess(ids):
            g = np.asarray([mpc.tail(variants[i], FRAMES + mpc.TAIL) for i in ids])
            feedback = np.zeros((len(g), 9))
            _, y = self.engine.trajectories(
                g, mpc.AIM, FRAMES * mpc.DT, feedback=feedback
            )
            return mpc.cost(y, feedback)

        loss, valid = assess(range(len(variants)))
        self.diag["early_screens"] += 1
        ids = np.flatnonzero(valid)
        if not len(ids):
            return
        ids = ids[np.argsort(loss[ids])[:4]]
        a = self.engine.twin
        p, v = a.ball.copy(), a.velocity.copy()
        area = []
        try:
            for s in (-1, 1):
                a.ball[:] = p + s * np.array([0.003, 0.004, 0.005])
                a.velocity[:] = v + s * np.array([0.038, 0.060, 0.085])
                cost, landed = assess(ids)
                area.append(np.where(landed, cost, 9.0))
        finally:
            a.ball[:] = p
            a.velocity[:] = v
        score = np.asarray(area).max(axis=0)
        # Penalize similar *actually issued* early command segments following a miss.
        for i, k in enumerate(ids):
            predicted = ref.copy()
            for g in variants[k][:5]:
                predicted += np.clip(
                    np.clip(g, self.low, self.high) - predicted,
                    -0.8 * self.stride,
                    0.8 * self.stride,
                )
            prefix = predicted[:5] - ref[:5]
            penalty = max(
                (
                    np.exp(
                        -np.sum((f - key) ** 2) / 0.8
                        - np.sum((u - prefix) ** 2) / 0.018
                    )
                    for f, u in self.failures
                ),
                default=0.0,
            )
            score[i] += 2.5 * penalty
        i = int(np.argmin(score))
        if score[i] >= 3:
            return
        path = variants[ids[i]]
        self.approach = (tick, path)
        self.applied = (tick, key.copy(), ref.copy())
        self.diag["early_selected"] += 1

    def step(self, tick, eye, x, q, ref, base):
        incoming = x is not None and x[4] < -0.95
        if x is not None and x[4] > 0.95 and self.next_gate:
            self.reset_approach()
        if incoming and x[1] < -0.49 and self.approach is not None:
            start, seq = self.approach
            age = tick - start
            if 0 <= age < len(seq) - 4:
                self.best = mpc.tail(seq[age:], mpc.HORIZON)
                self.tick, self.lease = tick, 3
            self.approach = None
        result = super().step(tick, eye, x, q, ref, base)
        if (
            incoming
            and self.next_gate < len(GATES)
            and -0.41 < x[1] <= GATES[self.next_gate]
        ):
            self.next_gate += 1
            self.screen(tick, x, q, ref, base)
        if self.approach is not None and incoming and x[1] > -0.49:
            start, seq = self.approach
            if tick - start < min(FRAMES, 18):
                self.diag["early_control_ticks"] += 1
                return seq[tick - start]
        return result


def run(a):
    meta = _read(ROOT / "fit.json")
    if meta["model"] != _hash(BANK):
        raise ValueError("Practice moved")
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    actors = []

    def make(*args, **kw):
        p = Player(*args, coverage=rank, **kw)
        actors.append(p)
        return p

    with (
        enabled(),
        patch.object(season.course, "PLAN", dose.PLAN),
        patch.object(mpc, "MPC", make),
    ):
        season.course.run(a)
    p = actors[-1]
    model = ROOT / "online-approaches.npz"
    rows = list(p.new)
    np.savez_compressed(
        model,
        keys=np.array([r[0] for r in rows]),
        paths=np.array([r[1] for r in rows]),
        groups=np.array([r[2] for r in rows]),
        failed_keys=np.array([r[0] for r in p.failures]),
        failed_moves=np.array([r[1] for r in p.failures]),
    )
    d = _read(a.out)
    d.update(initial_approaches=_hash(BANK), final_approaches=_hash(model))
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("activity", choices=("fit", "play"))
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
    fit() if a.activity == "fit" else run(a)

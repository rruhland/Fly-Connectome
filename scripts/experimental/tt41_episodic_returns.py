"""Use earned future returns to rank similar direct movements, without network gradients."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_carry_joints import Carried


def prior_values(file):
    doc = _read(file.with_suffix(".json"))
    tapes = {name: _read(file.with_name(name)) for name in doc["sources"]}
    values = {}
    for i, p in enumerate(doc["practice"]):
        r = tapes[p["source"]]["rows"][p["source_row"]]
        own = [
            e["time"]
            for e in r["events"]
            if e["kind"] == "legal_return" and e["side"] == 1
        ]
        values[i] = own
    return values


def survival(n):
    return 1 - 0.72 ** min(8, max(0, n))


class Episodic(Carried):
    def __init__(self, file, engine, *, practice, held):
        super().__init__(file, engine, practice=practice, held=held)
        self.scores = {}
        self.last = []
        self.failed = []
        for src in [file, *practice]:
            values = prior_values(src)
            keys = (
                {r[2] for r in self.rows}
                if src == file
                else {t for _, _, t in self.online if t[1] == src.stem}
            )
            for tag in keys:
                index, stamp = tag[-2:]
                count = sum(t > stamp / 480 + 0.025 for t in values[index])
                self.scores[tag] = survival(count - 1)
        self.q_effects = 0

    def begin_game(self):
        self.last = []

    def keep(self, archive, leg):
        # Value changes only when the next actual legal arm landing arrives.
        for n, prior in enumerate(reversed(self.last[-8:]), 1):
            self.scores[prior] = survival(n)
        self.last.append(leg)
        self.scores[leg] = 0.36  # Censored until the opponent redelivers.
        return super().keep(archive, leg)

    def learn_miss(self, archive):
        if self.last:
            self.scores[self.last[-1]] = 0.0
        usable = [
            (i, r)
            for i, r in enumerate(archive)
            if r[1] is not None and r[1][4] < -0.9 and r[1][1] < -0.65
        ]
        for i, (_, x, q, ref, _) in usable[-15::4]:
            seq = mpc.tail(
                np.asarray([r[-1] for r in archive[i : i + mpc.HORIZON]]), mpc.HORIZON
            )
            self.failed.append((mpc.query(x, q, ref), seq))
        self.failed = self.failed[-160:]

    def context(self, x, q, ref):
        self.key = mpc.query(x, q, ref)
        d, i = self.tree.query(self.key, min(len(self.rows), 100))
        pool = [(ds, *self.rows[j]) for ds, j in zip(d, i, strict=True)]
        pool.extend((np.linalg.norm(s - self.key), s, u, t) for s, u, t in self.online)
        lookup = {}
        for dist, s, u, t in sorted(pool, key=lambda r: r[0]):
            if t not in lookup and dist < 3.1:
                lookup[t] = (dist, u)
            if len(lookup) >= 26:
                break
        self.local = [(d, u, self.scores.get(t, 0.36)) for t, (d, u) in lookup.items()]
        self.local.extend(
            (np.linalg.norm(s - self.key), u, -0.35)
            for s, u in self.failed
            if np.linalg.norm(s - self.key) < 2.5
        )

    def evaluate(self, moves):
        if not self.local:
            return np.full(len(moves), 0.36)
        radius, goals, labels = map(np.asarray, zip(*self.local, strict=True))
        a = np.asarray(moves)[:, None, 3:16:3, :5]
        b = goals[None, :, 3:16:3, :5]
        md = np.mean(((a - b) / [0.22, 0.21, 0.25, 0.25, 0.31]) ** 2, axis=(-2, -1))
        w = np.exp(-0.65 * radius**2 - 0.85 * md)
        return (0.75 * 0.36 + w @ labels) / (0.75 + w.sum(axis=1))


class Actor(mpc.MPC):
    def begin(self):
        super().begin()
        self.priors.begin_game()

    def search(self, tick, x, q, ref, base):
        self.priors.context(x, q, ref)
        return super().search(tick, x, q, ref, base)

    def robust(self, goals, rank, nominal):
        agr, risk = self.uncertainty(goals, nominal)
        value = self.priors.evaluate(goals)
        winner = int(np.argmin(risk - 3.4 * value))
        self.diag["episodic_decisions_changed"] += winner != int(np.argmin(risk))
        return int(rank[winner]), int(agr[winner]), float(risk[winner])

    def missed(self, archive):
        self.priors.learn_miss(archive)


def run(a):
    held = {c["id"] for c in _read(a.plan)[a.deck][: a.limit]}
    objs = []

    def build(file, engine):
        v = Episodic(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )
        objs.append(v)
        return v

    with patch.object(mpc, "MPC", Actor), patch.object(mpc, "Fragments", build):
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
    model = objs[0]
    ck = a.out.with_suffix("").with_suffix(".npz")
    np.savez_compressed(
        ck,
        states=[r[0] for r in model.online],
        goals=[r[1] for r in model.online],
        tags=[repr(r[2]) for r in model.online],
        value=[model.scores.get(r[2], 0.36) for r in model.online],
        failed_states=[r[0] for r in model.failed],
        failed_goals=[r[1] for r in model.failed],
    )
    doc = _read(a.out)
    doc.update(checkpoint_sha256=_hash(ck), episode_values=len(model.scores))
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())

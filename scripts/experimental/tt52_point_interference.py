"""Separate the consequences of adding and evicting motor memories during the 63-return point."""

import argparse
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import tt49_cumulative_memory as history
from . import tt51_transfer_after_point as boundary
from . import tt46_continuous_season as season
from .tt4_transfer_trial import _hash, _read, _write

ROOT = boundary.ROOT
course = season.course


def factorial():
    p = history.load_rows(ROOT / "after-09.json.gz")
    q = history.load_rows(ROOT / "after-10.json.gz")
    n = _read(ROOT / "after-10.json.gz")["rows"][-1]["planner"]["credited_fragments"]
    if (
        n <= 0
        or n >= len(q)
        or not all(history.identical(u, v) for u, v in zip(p[n:], q[:-n], strict=True))
    ):
        raise ValueError(
            "Could not separate exact point-acquired and point-evicted fragments"
        )
    return p, q, p + q[-n:], p[n:]


class Balanced(history.Growing):
    def priors(self, x, q, ref, limit):
        key = course.mpc.query(x, q, ref)
        stamp = (len(self.online), id(self.online[-1][0]) if self.online else 0)
        if stamp != getattr(self, "balanced_stamp", None):
            self.balanced_stamp = stamp
            self.arr = list(self.online)
            self.states = np.asarray([r[0] for r in self.arr]).reshape(-1, len(key))
            self.frequency = Counter(r[2][1] for r in self.arr if r[2][0] == "live")
        near, ids = self.tree.query(key, min(len(self.rows), 160))
        options = [
            (float(d), self.rows[i][1], self.rows[i][2], False)
            for d, i in zip(near, ids, strict=True)
        ]
        distances = np.maximum(0.0, np.linalg.norm(self.states - key, axis=1) - 0.10)
        options.extend(
            (float(d), g, tag, True)
            for d, (_, g, tag) in zip(distances, self.arr, strict=True)
        )
        result = []
        used = set()
        points = Counter()
        for d, g, tag, is_online in sorted(options, key=lambda row: row[0]):
            if tag in used:
                continue
            if len(result) >= 2 and d > 3.1:
                break
            old_point = is_online and tag[0] == "live" and tag[1] != self.current_game
            if (
                old_point
                and self.frequency[tag[1]] > 0.10 * len(self.arr)
                and points[tag[1]] >= 2
            ):
                continue
            used.add(tag)
            if old_point:
                points[tag[1]] += 1
            result.append((g.copy(), is_online))
            if len(result) >= limit:
                break
        return result


class Player(season.Player):
    def begin(self):
        super().begin()
        self.priors.current_game = self.serial


def run(a):
    pre, post, union, cut = factorial()
    a.carry = ROOT / "after-10.json.gz"
    a.deck = "assessment" if a.mode == "frozen" else "practice"
    a.n = len(_read(boundary.PLAN)[a.deck])
    bank = {"addition": union, "eviction": cut, "balanced": union}[a.condition]
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    from .tt41_early_optics_carry import resume

    def restore(actor, path, doc):
        resume(
            actor, path, {**doc, "rows": [None] * doc["parent_serial"] + doc["rows"]}
        )
        actor.priors.online = deque(bank)

    with (
        patch.object(course, "PLAN", boundary.PLAN),
        patch.object(course, "resume", restore),
        patch.object(
            course,
            "Carried",
            Balanced if a.condition == "balanced" else history.Growing,
        ),
        patch.object(
            course.mpc, "MPC", lambda *x, **kw: Player(*x, coverage=rank, **kw)
        ),
    ):
        course.run(a)
    d = _read(a.out)
    d.update(
        condition=a.condition,
        archive=_hash(a.carry),
        initial_size=len(bank),
        definition=_hash(Path(__file__)),
    )
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("condition", choices=("addition", "eviction", "balanced"))
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=("live", "frozen"), default="frozen")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(resume=False, cap=240)
    run(p.parse_args())

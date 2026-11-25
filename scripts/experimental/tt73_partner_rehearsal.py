"""Practice teacher previews several physically generated returns before choosing an exercise."""

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import tt71_delivered_curriculum as prior
from .tt4_transfer_trial import _hash, _read, _write
from .tt44_paired_credit import State

ROOT = Path("artifacts/tt73_partner_rehearsal")
body = prior.body


def plan(folder, budget):
    pool = {c["id"]: c for c in _read(body.dose.PLAN)["practice"]}
    rng = np.random.default_rng(730102 if budget == 32 else 730103)
    cards = [
        dict(
            pool[i],
            seed=int(rng.integers(1, 2**30)),
            reply_seed=int(rng.integers(1, 2**30)),
        )
        for _ in range(5)
        for i in rng.permutation(sorted(pool))
    ]
    folder.mkdir(parents=True, exist_ok=True)
    _write(
        folder / "course.json",
        dict(
            cards=cards,
            budget=budget,
            model=_hash(prior.MODEL),
            fixture=_hash(body.PLAN),
            base=_hash(body.season.CARRY),
            method="During practice only, the physical opponent can preview four different continuously sampled legal requests in a private copy of its current state. The student's action, memory, pictures and referee are only updated on the one real point. Compare 35 percent unaltered + 65 percent competence-targeted versus entirely uniform practice. All held points use the original independent broad law. First 32 exposures used solely to verify preview fidelity.",
        ),
    )


class Coach(prior.Partner):
    def __init__(self, kind):
        super().__init__(kind)
        self.shadow = None
        self.pending = []
        self.diagnostics = []
        self.peek_count = 0

    def close(self):
        if self.shadow is not None:
            self.shadow.close()

    def preview(self, saved, card, leg, target):
        b = self.shadow
        saved.restore(b)

        def draw(c, j, mode, rng):
            return (
                (target, "teacher_preview")
                if j == leg
                else prior.STANDARD(c, j, mode, rng)
            )

        with patch.object(prior.opponent, "draw", draw):
            cmd = np.zeros(6)
            for _ in range(150):
                _, _, done, capped, _ = b.exchange(cmd, body.mpc.FAR_COMMAND)
                if done or capped or leg + 1 in b.crossings:
                    break
        offers = b.point_placements()["delivered"]
        return prior.actual(offers[leg]) if leg < len(offers) else None

    def select(self, card, leg, mode, rng):
        key = (card["seed"], card["reply_seed"], leg)
        if key in self.cache:
            return self.cache[key]
        first, tag = prior.STANDARD(card, leg, mode, rng)
        rand = np.random.default_rng([*key, 73012])
        if self.kind == "uniform":
            return first, tag
        if self.shadow is None:
            self.shadow = body.mpc.arena(width=64, height=48)
        saved = State(self.plant)
        use = rand.random() >= 0.35
        candidates = [first] + (
            [
                tuple(s)
                for s in rand.uniform(prior.BOUNDS[:, 0], prior.BOUNDS[:, 1], (3, 4))
            ]
            if use
            else []
        )
        estimates = [self.preview(saved, card, leg, c) for c in candidates]
        self.peek_count += len(estimates)
        category = [int(prior.cell(v)) if v is not None else -1 for v in estimates]
        n = self.counts.sum(1)
        competence = (self.counts[:, 1] + 3) / (n + 4)
        target = 0.80 - 0.22 * min(1.0, self.offers / 160)
        priority = np.exp(-0.5 * ((competence - target) / 0.17) ** 2) / (n + 4) ** 0.50
        weights = np.array([priority[i] if i >= 0 else 0.0001 for i in category])
        k = int(rand.choice(len(weights), p=weights / weights.sum())) if use else 0
        result = (tuple(candidates[k]), "physical_practice" if use else tag)
        self.pending.append(
            dict(
                leg=leg,
                survey=[None if y is None else y.tolist() for y in estimates],
                chosen=k,
                requested=list(candidates[k]),
            )
        )
        self.cache[key] = result
        return result

    def observe(self, row):
        super().observe(row)
        if self.kind == "uniform":
            return
        real = row["placements"]["delivered"]
        for p in self.pending:
            k = p["leg"]
            v = prior.actual(real[k]) if k < len(real) else None
            p["actual"] = None if v is None else v.tolist()
            if k < len(real) and not np.allclose(real[k]["target"], p["requested"]):
                raise ValueError("Different practice request issued")
        self.diagnostics.append(self.pending)
        self.pending = []


def run(a):
    folder = ROOT / ("pilot" if a.budget == 32 else "course")
    if a.phase == "plan":
        plan(folder, a.budget)
        return
    with (
        patch.object(prior, "ROOT", folder),
        patch.object(prior, "PLAN", folder / "course.json"),
    ):
        prior.practice(a.phase, a.feeds, Coach)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "uniform", "rehearsal"))
    p.add_argument("--budget", type=int, choices=(32, 256), default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    run(p.parse_args())

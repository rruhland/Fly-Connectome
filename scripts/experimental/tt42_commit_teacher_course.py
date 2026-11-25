"""Release-disjoint chronological learning with stock, pilot and expanded visual teachers."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt41_bounce_joint import OPTIC, PRACTICE
from .tt42_commit_teacher import OUT, PLAN
from .tt42_commit_teacher_student import FRAGMENTS, WithTeacher

SLOTS = slice(2, 5)
KINDS = ("stock", "pilot", "all")
WINDOW = 9


def recent_missed(archive):
    """Sample the current optical arrival and the arm's actual issued target curve."""
    if len(archive) <= WINDOW:
        return []
    end = archive[-1][0]
    out = []
    for i, (t, x, q, ref, _) in enumerate(archive[: -WINDOW + 1]):
        if t < end - 52 or t % 3 or x is None or x[4] > -0.9 or x[1] > -0.57:
            continue
        goal = np.asarray([r[-1] for r in archive[i : i + WINDOW]], dtype=float)
        if len(goal) != WINDOW:
            continue
        out.append((mpc.query(x, q, ref), goal[:, :5] - ref[:5]))
    return out


class NegativeCredit:
    def __init__(self):
        self.samples = deque(maxlen=180)

    def keep(self, archive):
        rows = recent_missed(archive)
        self.samples.extend(rows)
        return len(rows)

    def resembles_failed(self, key, seq, ref):
        if not self.samples:
            return None
        states = np.asarray([s for s, _ in self.samples])
        d = np.linalg.norm(states - key, axis=1)
        ids = np.flatnonzero(d < 1.28)
        if not len(ids):
            return None
        local = np.asarray([self.samples[i][1] for i in ids])
        diff = local - (seq[:WINDOW, :5] - ref[:5])
        err = np.sqrt(np.mean(diff**2, axis=(1, 2)))
        z = int(np.argmin(err))
        if err[z] < 0.057:
            return [float(d[ids[z]]), float(err[z])]
        return None

    def digest(self):
        return {
            "rows": len(self.samples),
            "sha256": digest_json([[k.tolist(), v.tolist()] for k, v in self.samples]),
        }


class AdaptiveTeacher(WithTeacher):
    def __init__(self, bank, engine, teacher, *, held, use):
        super().__init__(bank, engine, teacher, held=held, use=use)
        self.recalls = []
        self.negative = NegativeCredit()

    def priors(self, x, q, reference, limit):
        audit = []
        attempt = {"query": [], "stock_miss_declines": [], "substituted": 0}
        if not self.use:
            audit.append(["no teacher bank enabled"])
        # Preselect two alternate library rows only when the carried miss
        # ledger contains experience. It cannot change the rollout budget.
        limit_search = limit + 2 if self.negative.samples else limit
        stock = Carried.priors(self, x, q, reference, limit_search)
        key = mpc.query(x, q, reference)
        defend = []
        weak = []
        for seq, online in stock:
            similarity = self.negative.resembles_failed(key, seq, reference)
            if similarity is None:
                defend.append((seq, online))
            else:
                attempt["stock_miss_declines"].append(similarity)
                weak.append((seq, online))
        safe = (defend + weak)[:limit]
        replacement = []
        if self.use:
            if len(safe) < 3:
                audit.append(["not enough regular candidates"])
            else:
                for seq, label, distance in self.offers(x, q, reference, audit):
                    similarity = self.negative.resembles_failed(key, seq, reference)
                    name = repr(label)
                    if similarity is not None:
                        audit.append(
                            ["matched a real issued miss", name, distance, similarity]
                        )
                        continue
                    audit.append(["admitted", name, distance])
                    self.admissions.append(distance)
                    replacement.append((seq, True))
                    if len(replacement) >= 2:
                        audit.append(["fixed slots filled"])
                        break
                if not audit:
                    audit.append(["no supported template"])
        attempt["query"] = audit
        attempt["substituted"] = len(replacement)
        self.recalls.append(attempt)
        return safe[: len(safe) - len(replacement)] + replacement


class Learner(mpc.MPC):
    def begin(self):
        super().begin()
        self.priors.recalls = []
        self.priors.admissions = []

    def search(self, tick, x, q, reference, base):
        before = len(self.priors.recalls)
        super().search(tick, x, q, reference, base)
        if len(self.priors.recalls) > before:
            self.priors.recalls[-1]["tick"] = tick

    def missed(self, archive):
        n = self.priors.negative.keep(archive)
        self.diag["observed_miss_fragments"] += n
        self.diag["physical_miss_updates"] += 1


def audited_roster(plan, sources):
    ids = {c["id"] for c in plan["assessment"][SLOTS]}
    practice_ids = {c["id"] for c in plan["practice"]}
    if ids & practice_ids:
        raise ValueError("Release lists are not disjoint")
    for p in sources:
        rec = _read(p.with_suffix(".json"))
        if rec["bank_sha256"] != _hash(p):
            raise ValueError(f"Changed archive {p}")
        used = {c["card"]["id"] for c in rec["practice"]}
        if ids & used:
            raise ValueError(f"Assessment releases found in {p}: {sorted(ids & used)}")
    return plan["assessment"][SLOTS]


def run(a):
    pilot = OUT / "earned-camera.npz"
    selected = OUT / "earned-camera-41.npz" if a.condition == "all" else pilot
    cards = audited_roster(_read(a.plan), (mpc.BANK, PRACTICE, selected))
    held = {c["id"] for c in cards}
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    hashes = {
        p.name: _hash(p) for p in (mpc.BANK, PRACTICE, OPTIC, a.plan, a.feeds, selected)
    }
    with np.load(mpc.BANK) as bank:
        raw = mpc.DeepField(bank, "local")
    shots = mpc.measured_launches(a.feeds)
    doc = {
        "role": "chronological online camera-only learner, on unseen full points",
        "condition": a.condition,
        "source_sha256": hashes,
        "teacher_has_no_native_policy_code": True,
        "cards": cards,
        "ready": ready,
        "cap": a.cap,
        "budget": 28,
        "rows": [],
    }
    with mpc.centered(ready):
        court = mpc.arena(protocol="random")
        res = mpc.resources(court)

        def make(bank, engine):
            return AdaptiveTeacher(
                bank,
                engine,
                FRAGMENTS(selected, engine),
                held=held,
                use=a.condition != "stock",
            )

        with patch.object(mpc, "Fragments", make):
            model = Learner(mpc.BANK, raw, variant="elastic", budget=28)
        try:
            for card in cards:
                before = model.priors.negative.digest()
                game = mpc.play(
                    court,
                    *res,
                    ready,
                    card,
                    shots[card["id"]],
                    raw,
                    model,
                    OPTIC,
                    "live",
                    a.cap,
                )
                requests = game["placements"]["requested"]
                if any(e["draw"] != "random" for e in requests) or len(
                    {tuple(e["target"]) for e in requests}
                ) != len(requests):
                    raise ValueError("A delivered game used a noncontinuous opponent")
                if doc["rows"]:
                    last = doc["rows"][-1]
                    if (
                        last["memory"]["after"] != game["memory"]["before"]
                        or last["miss_memory"]["after"] != before
                    ):
                        raise ValueError(
                            "Previous practice did not carry to the next point"
                        )
                game["miss_memory"] = {
                    "before": before,
                    "after": model.priors.negative.digest(),
                }
                game["recall_attempts"] = model.priors.recalls
                game["teacher_admissions"] = model.priors.admissions
                doc["rows"].append(game)
                _write(a.out, doc)
                print(
                    a.condition,
                    card["id"],
                    game["score"]["learner"],
                    len(model.priors.admissions),
                    game["planner"].get("updates", 0),
                    game["planner"].get("observed_miss_fragments", 0),
                    flush=True,
                )
        finally:
            model.close()
            res[0].close()
            court.close()
    doc["complete"] = True
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("condition", choices=KINDS)
    p.add_argument("out", type=Path)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--cap", type=int, default=200)
    run(p.parse_args())

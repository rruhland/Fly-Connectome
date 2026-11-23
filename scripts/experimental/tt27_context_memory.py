"""Reward-only local action memory, tested over a sequence of changed table surfaces."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt27_flight_feedback import OUT
from .tt27_surface_transfer import HEIGHTS, key


def similarity(a, b):
    if a is None or b is None:
        return 0.25
    return 0.02 + 0.98 * np.exp(-0.5 * np.sum(((a - b) / [0.12, 0.14]) ** 2))


class Memory:
    def __init__(self, practice, *, context, local, sparse=False):
        self.context = context
        self.local = local
        self.rows = []
        for block in practice:
            optical = key(block)
            for i, r in enumerate(block["rows"]):
                # Sparse source practice exposes one deterministic, balanced randomized choice per feed.
                if sparse:
                    action = int(np.random.default_rng(block["seed"] * 137 + i).integers(len(HEIGHTS)))
                    self.add(optical, r["origin"], action, r["scores"][action])
                else:
                    for a, v in enumerate(r["scores"]):
                        self.add(optical, r["origin"], a, v)

    def add(self, optical, origin, action, reward):
        self.rows.append((optical, origin, action, reward))

    def choose(self, optical, origin):
        wins = np.zeros(len(HEIGHTS))
        attempted = np.zeros(len(HEIGHTS))
        for o, g, a, v in self.rows:
            weight = similarity(o, optical) if self.context else 1.0
            if self.local:
                weight *= 1.0 if g == origin else 0.035
            attempted[a] += weight
            wins[a] += weight * v
        # Mild preference for an unchanged motor controller when the data are sparse.
        rate = (wins + np.array([1.15] + [1.0] * (len(wins) - 1))) / (attempted + 3.0)
        return int(np.argmax(rate))


def run(data, output=OUT / "context-memory.json.gz"):
    blocks = _read(data)["blocks"]
    practice = [b for b in blocks if b["label"] == "practice"]
    policies = {
        "context": (True, True, False, True, True),
        "reset_between": (True, True, False, True, False),
        "source_only": (True, True, False, False, False),
        "no_surface": (False, True, False, True, True),
        "no_trajectory": (True, False, False, True, True),
        "sparse_source": (True, True, True, True, True),
        "target_only": (True, True, True, True, True),
    }
    report = []
    for cohort in ("held", "raw"):
        targets = [b for b in blocks if b["label"] == cohort]
        for name, (context, local, sparse, update, carry) in policies.items():
            source = practice if name != "target_only" else []
            state = Memory(source, context=context, local=local, sparse=sparse)
            for b in targets:
                if not carry:
                    state = Memory(source, context=context, local=local, sparse=sparse)
                optical = key(b)
                steps = []
                for r in b["rows"]:
                    act = state.choose(optical, r["origin"])
                    reward = r["scores"][act]
                    steps.append(dict(action=act, legal=reward, contact=r["contacts"][act]))
                    if update:
                        state.add(optical, r["origin"], act, reward)
                report.append(dict(policy=name, cohort=cohort, block=b["number"], steps=steps))
            print(
                cohort,
                name,
                [
                    sum(s["legal"] for s in r["steps"])
                    for r in report
                    if r["cohort"] == cohort and r["policy"] == name
                ],
                flush=True,
            )
    _write(output, dict(source=str(data), policies=policies, records=report))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=OUT / "surface-transfer.json.gz")
    run(parser.parse_args().data)

"""Update a direct camera-to-joint field during play from self-issued references and delayed scores."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_joint_deep import DeepField, run
from .tt38_scored_joints import ScoredJoints


class RewardedField:
    def __init__(self, model, bank, path):
        self.model = model
        with np.load(bank) as z:
            self.x = z["states"]
            self.u = z["targets_now"]
        self.replay = ScoredJoints()
        self.checkpoint = path

    def begin_point(self):
        self.replay.begin_point()
        self.previous_score = 0
        self.diagnostics = {"start_frames": len(self.replay.states), "credits": 0}

    def record_command(self, tick, opt, q, reference):
        self.replay.step(tick, opt, q, reference)

    def score_feedback(self, score):
        if score <= self.previous_score:
            return
        self.previous_score = score
        size = len(self.replay.states)
        self.replay.credit()
        if size == len(self.replay.states):
            return
        self.diagnostics["credits"] += 1
        data = {
            "states": np.vstack([self.x, self.replay.states[-8192:]]),
            "targets_now": np.vstack([self.u, self.replay.targets[-8192:]]),
        }
        self.model = DeepField(data, "local")

    def end_point(self, *, terminal):
        self.diagnostics["end_frames"] = len(self.replay.states)
        tmp = self.checkpoint.with_suffix(".partial.npz")
        np.savez_compressed(
            tmp,
            states=np.asarray(self.replay.states).reshape(-1, 11),
            targets_now=np.asarray(self.replay.targets).reshape(-1, 6),
        )
        tmp.replace(self.checkpoint)

    def get(self, optical, q, *, reference=None):
        return self.model.get(optical, q, reference=reference)


def evaluate(source, plan, output, deck, feeds):
    checkpoint = output.with_suffix("").with_suffix(".npz")
    run(
        feeds,
        source,
        plan,
        deck,
        "local",
        output,
        cap=200,
        field_hook=lambda field: RewardedField(field, source, checkpoint),
        details={
            "online": "own optical reversal and scalar reward",
            "prior_sha256": _hash(source),
        },
    )
    d = _read(output)
    d.update(additional_checkpoint=checkpoint.name, additional_sha256=_hash(checkpoint))
    _write(output, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bank", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--deck", default="held_openings")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    a = p.parse_args()
    evaluate(a.bank, a.plan, a.output, a.deck, a.feeds)

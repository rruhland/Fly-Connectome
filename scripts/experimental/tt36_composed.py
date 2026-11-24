"""Compose learned present-time body feedback and a hit adjustment in a single joint target."""

import argparse
from pathlib import Path

import numpy as np

from . import tt36_reach as reach
from .tt4_transfer_trial import _hash, _read, _write
from .tt36_course import OUT
from .tt36_gate import Gate
from .tt36_hit import HitAdjustment


def position_preserving(delta, jacobian):
    """Remove joint motion that would move the blade center under the learned body map."""
    return delta - jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + 0.0004 * np.eye(3), jacobian @ delta
    )


def run(feeds, cohort, mode):
    hit_file = OUT / "hit-checkpoint.json"
    gate_file = OUT / "gate-checkpoint.json"
    hit = HitAdjustment(_read(hit_file))
    gate_state = _read(gate_file)

    class Composed(reach.Reach):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gate = Gate(gate_state)

        def bias(self, tick, optical, joints, frames):
            alignment = super().bias(tick, optical, joints, frames)
            if mode == "gated" and self.gate.choose(optical, joints) not in (
                "hit",
                "both",
            ):
                return alignment
            outgoing = hit.bias(optical, joints)
            if mode in ("projected", "scrambled") and np.any(outgoing):
                _, jacobian, support_distance = self.body.local(joints[:5])
                if support_distance > 1.5:
                    return alignment
                if mode == "scrambled":
                    jacobian = jacobian[:, [2, 0, 3, 4, 1]]
                outgoing = position_preserving(outgoing, jacobian)
            return outgoing if alignment is None else alignment + outgoing

    reach.run(
        feeds,
        OUT / "reach-observations.npz",
        OUT,
        cohort,
        [mode],
        controllers={mode: Composed},
    )
    path = OUT / f"reach-{cohort}-{mode}.json.gz"
    doc = _read(path)
    doc["hit_sha256"] = _hash(hit_file)
    if mode == "gated":
        doc["gate_sha256"] = _hash(gate_file)
    _write(path, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("composed", "gated", "projected", "scrambled"))
    p.add_argument("cohort", choices=("new", "replies", "confirm"))
    p.add_argument("feeds", type=Path)
    args = p.parse_args()
    run(args.feeds, args.cohort, args.mode)

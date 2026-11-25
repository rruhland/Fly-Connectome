"""Compare a newly earned motor choice to the same player's established joint policy."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt97_return_to_center as transfer
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt99_protected_learning")
PLAN = ROOT / "protocol.json"
ordinary = transfer.season.Player


class Protected(ordinary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = None
        self.updated = False
        reward = self.priors.keep

        def credit(tape, leg):
            n = reward(tape, leg)
            if n:
                self.updated = True
            return n

        self.priors.keep = credit

    def begin(self):
        super().begin()
        if self.anchor is None:
            # course.resume already installed the verified training checkpoint.
            self.anchor = tuple(self.priors.online)
            self.anchor_hash = self.priors.digest()
        self.secondary = []

    def search(self, tick, x, q, reference, base):
        live = self.priors.online
        saved = (self.best, self.tick, self.lease)
        initial_rng = deepcopy(self.rng.bit_generator.state)
        before = len(self.notes)
        self.priors.online = self.anchor
        try:
            super().search(tick, x, q, reference, base)
        finally:
            self.priors.online = live
        if not self.updated or len(self.notes) == before:
            return
        baseline = self.notes[-1]
        if baseline.get("chosen") or (
            self.best is not None and self.tick < tick < self.tick + self.lease
        ):
            return
        established = (self.best, self.tick, self.lease)
        primary_rng = deepcopy(self.rng.bit_generator.state)
        # Reuse exactly the same random proposal stream. A substitute must
        # earn support from acquired actions and current camera/body evidence.
        self.best, self.tick, self.lease = saved
        self.rng.bit_generator.state = initial_rng
        self.notes.pop()
        try:
            super().search(tick, x, q, reference, base)
            alt = self.notes.pop() if len(self.notes) > before else {}
        finally:
            self.rng.bit_generator.state = primary_rng
        use = bool(alt.get("chosen") and alt.get("robust") == 2)
        self.secondary.append(
            dict(
                tick=tick,
                baseline=baseline.get("chosen"),
                alternative=alt.get("chosen"),
                robust=alt.get("robust"),
                issued=use,
            )
        )
        if use:
            alt["protected_update"] = True
            self.notes.append(alt)
        else:
            self.best, self.tick, self.lease = established
            self.notes.append(baseline)


def plan():
    p = _read(transfer.PLAN)
    ROOT.mkdir(exist_ok=True)
    _write(
        PLAN,
        dict(
            training_bank=p["bank"],
            course=_hash(transfer.PLAN),
            games=32,
            rule="Thirty-two assigned ordinary points on the original centered installation after the identical shifted-clamp practice as TT97. Lock the verified starting policy as a reference. If its search finds no legal candidate and has no remaining earlier commitment, challenge with the continually updated bank from the same live sensor state and initial planning randomness. Issue a challenger only after both model perturbations land. Real earned fragments continue to accumulate through all games. Paired opponents, one launch, no scripted next trajectory.",
        ),
    )


def run(a):
    doc = _read(PLAN)
    if doc["course"] != _hash(transfer.PLAN):
        raise ValueError("Mismatched ordinary course")
    mpc = transfer.course.mpc
    original = mpc.play

    def play(*v, **kw):
        result = original(*v, **kw)
        model = v[10]
        result.update(challenges=model.secondary, protected_bank=model.anchor_hash)
        return result

    with (
        patch.object(transfer.season, "Player", Protected),
        patch.object(mpc, "play", play),
    ):
        transfer.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "live"))
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        n=32,
        carry=transfer.PRACTICE,
        cap=240,
        resume=False,
        out=ROOT / "live.json.gz",
    )
    a = p.parse_args()
    plan() if a.phase == "plan" else run(a)

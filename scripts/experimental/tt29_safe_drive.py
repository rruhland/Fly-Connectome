"""Retest successful strokes after rejecting mechanically overlapping preparations and paths."""


import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_so101 import GRIP_BODY, _GRASP_POS
from .tt29_arm_clearance import ArmClearance
from .tt29_conditions import CONDITIONS, apply
from .tt29_drive import OUT, play
from .tt29_rally_physics import BladeIK
from .tt29_ready_block import BlockArena
from .tt29_reward_practice import practice_feed


class GuardedArena(BlockArena):
    def __init__(self, **kwargs):
        self.check = None
        self.penetration = 0.0
        super().__init__(**kwargs)
        self.check = ArmClearance(self.model)

    def launch(self, shot):
        super().launch(shot)
        self.penetration = 0.0
        self._sample_clearance()

    def _sample_clearance(self):
        for _, _, d in self.check.violations(self.data):
            self.penetration = min(self.penetration, d)

    def _aerodynamics(self):
        if self.check is not None and self.penetration == 0.0:
            self._sample_clearance()
        super()._aerodynamics()


class PoseClearance:
    def __init__(self, arena):
        self.ik = BladeIK(arena)
        self.guard = ArmClearance(arena.model)
        self.body = arena.model.body(GRIP_BODY).id
        self.free = arena.model.joint("learner_paddle_free").qposadr[0]

    def violations(self, q):
        self.ik.pose(q)
        d = self.ik.data
        d.qpos[self.free : self.free + 3] = (
            d.xpos[self.body] + d.xmat[self.body].reshape(3, 3) @ _GRASP_POS
        )
        d.qpos[self.free + 3 : self.free + 7] = d.xquat[self.body]
        return self.guard.violations(d)


def main():
    data = _read(OUT / "driven-receiving.json.gz")
    a = BlockArena(frameskip=4)
    try:
        c = PoseClearance(a)
        low, high = c.ik.low, c.ik.high
        worked = [r["motor"] for r in data["trials"] if r["legal"]]
        clean = []
        for m in worked:
            q = np.array(m["ready"])
            v = np.array(m["rate"])
            if all(
                not c.violations(np.clip(q + t * v, low, high))
                for t in np.linspace(0, m["span"], 9)
            ):
                clean.append(m)
        print("candidate ready/strike clearance", len(clean), len(worked), flush=True)
        if not clean:
            _write(OUT / "safe-drive.json.gz", dict(legal=len(worked), clear=0))
            return
        rng = np.random.default_rng(2972)
        deck = [practice_feed(rng, 85900 + i) for i in range(24)]
        scores = [sum(play(a, s, m)["legal"] for s in deck) for m in clean]
        selected = clean[int(np.argmax(scores))]
    finally:
        a.close()
    by_condition = {}
    for name in ("stock", "materials", "loaded_arm"):
        a = GuardedArena(frameskip=4)
        try:
            apply(a, CONDITIONS[name])
            rng = np.random.default_rng(2973)
            held = []
            for i in range(24):
                shot = practice_feed(rng, 98100 + i)
                r = play(a, shot, selected, events=True)
                held.append(dict(shot=shot, **r, overlap=a.penetration))
            by_condition[name] = held
            print(
                name,
                sum(r["legal"] and r["overlap"] == 0.0 for r in held),
                len(held),
                flush=True,
            )
        finally:
            a.close()
    _write(
        OUT / "safe-drive.json.gz",
        dict(
            legal=len(worked),
            clear=len(clean),
            validation=scores,
            selected=selected,
            held=by_condition,
        ),
    )


if __name__ == "__main__":
    main()

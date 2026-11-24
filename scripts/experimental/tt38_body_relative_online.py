"""Learn a local motor correction from a paddle-relative miss seen in the pupil's own RGB."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_joint_deep as deep
from .tt4_transfer_trial import _hash, _read, _write
from .tt38_body_relative import BANK, OUT, PLAN, RelativeField

PRACTICE = PLAN.with_name("school-plan.json")
SCALE = np.array([.10, .09, .08, 1.1, 1.4, 1.6, .65, .7, .8])


def schedule(output, roster=PRACTICE):
    source = _read(roster)
    anchors = source["practice"][42:50]
    held = {c["id"] for c in source["held_openings"]}
    if len(anchors) != 8 or any(c["id"] in held for c in anchors):
        raise ValueError("Practice anchors intersect the untouched confirmation roster")
    _write(output, dict(roster_sha256=_hash(roster), practice=[
        dict(c, reply_seed=381914107 + 257 * b + 23 * i) for b in range(2) for i, c in enumerate(anchors)],
        carry=[dict(c, reply_seed=382136099 + 23 * i) for i, c in enumerate(anchors)],
        note="Same independent development releases, with fresh broad opponent continuation draws in each block."))


def stereo_ball(eye, time):
    """Use only two ordinary RGB rays from the same exposure."""
    samples = {r.stream: r for r in reversed(eye.rays) if abs(r.time - time) < 1.e-9 and r.depth_weight > 0}
    if "left" not in samples or "right" not in samples:
        return None
    a, b = (samples[n] for n in ("left", "right"))
    delta = b.eye - a.eye
    cos = a.direction @ b.direction
    if abs(cos) >= .985:
        return None
    ta = (a.direction @ delta - cos * (b.direction @ delta)) / (1. - cos**2)
    tb = (cos * (a.direction @ delta) - b.direction @ delta) / (1. - cos**2)
    pa, pb = a.eye + ta * a.direction, b.eye + tb * b.direction
    if min(ta, tb) <= 0 or np.linalg.norm(pa - pb) > .012:
        return None
    return .5 * (pa + pb)


def passage(t0, rel0, t1, rel1):
    """An interpolated body-relative crossing, after both sensed exposures arrive."""
    if not (0 < t1 - t0 <= 4 and 0. < rel0[1] < .23 and -.23 < rel1[1] <= 0.):
        return None
    weight = rel0[1] / (rel0[1] - rel1[1])
    error = (rel0 + weight * (rel1 - rel0))[[0, 2]]
    if not (.052 <= np.linalg.norm(error) <= .24):
        return None
    return t0 + weight * (t1 - t0), error


class PassageField(RelativeField):
    """All motor changes use current observations and a small memory of completed approaches."""

    def __init__(self, bank, receipt, weights, *, mode, memory=None):
        self.mode_credit = mode
        self.memory = [] if memory is None else memory
        self.examples = np.empty((0, len(SCALE)))
        self.targets = np.empty((0, 2))
        self.refresh()
        super().__init__(bank, receipt, weights, "position")

    def refresh(self):
        self.memory[:] = self.memory[-24:]
        if self.memory:
            self.examples = np.concatenate([np.asarray(s["states"]) for s in self.memory])
            self.targets = np.concatenate([np.asarray(s["targets_m"]) for s in self.memory])

    def begin_point(self):
        super().begin_point()
        self.early = deque(maxlen=16)
        self.previous_blade_gap = None
        self.gaps = []
        self.last_score = -1
        self.diagnostics.update(corrected_ticks=0, shift_m_sum=0., practice_misses=[], learned=False)

    def score_feedback(self, public_own_count):
        if public_own_count > 0:
            self.last_score = self.tick

    def observe(self, optical, eye, tick, q):
        if optical is None or optical[4] >= -.9 or optical[1] > -.95:
            return
        ball = stereo_ball(eye.eye, tick * eye.dt)
        if ball is None:
            return
        rel = ball - self.body.position(q)
        if self.previous_blade_gap is not None:
            t0, previous = self.previous_blade_gap
            found = passage(t0, previous, tick, rel)
            if found is not None:
                at, gap = found
                contexts = [(t, x, shift) for t, x, shift in self.early if 2 <= at - t < 11][::2]
                if contexts:
                    self.gaps.append((at, gap, contexts))
                    self.diagnostics["practice_misses"].append(dict(tick=at, transverse_gap_m=gap.tolist(),
                                                                      informed_ticks=len(contexts)))
        self.previous_blade_gap = tick, rel

    def correction(self, x):
        if len(self.examples) == 0 or self.mode_credit == "fixed":
            return np.zeros(2)
        distance = np.linalg.norm(self.examples - x, axis=1)
        short = distance < .66
        if not short.any():
            return np.zeros(2)
        w = np.exp(-.5 * (distance[short] / .25)**2)
        support = np.clip((.66 - np.min(distance[short])) / .32, 0., 1.)
        return support * ((w @ self.targets[short]) / w.sum())

    def at(self, optical, q, p, speed):
        base, confidence = super().at(optical, q, p, speed)
        gap = optical[:3] - p
        if base is None or optical[4] >= -.9 or not (-.03 < gap[1] < .50):
            return base, confidence
        x = np.r_[gap, optical[3:6] - speed, q[[0, 3, 4]]] / SCALE
        shift = self.correction(x)
        self.early.append((self.tick, x.copy(), shift.copy()))
        if np.any(shift):
            j = self.body.jacobian(base)[[0, 2]]
            gain = np.diag([1., .12, .8, .7, .48])
            weighted = j @ gain
            change = gain @ weighted.T @ np.linalg.solve(weighted @ weighted.T + .026**2 * np.eye(2), shift)
            adjustment = np.clip(change, -.15, .15)
            base[:5] += adjustment
            self.diagnostics["corrected_ticks"] += 1
            self.diagnostics["shift_m_sum"] += float(np.linalg.norm(j @ adjustment))
        return base, confidence

    def end_point(self, *, terminal):
        if self.mode_credit != "learn" or not terminal:
            return
        candidates = [(tick, gap, context) for tick, gap, context in self.gaps if tick > self.last_score]
        if not candidates:
            return
        tick, gap, support = candidates[-1]
        states = [x.tolist() for _, x, _ in support]
        targets = np.array([prior + .40 * gap for _, _, prior in support])
        self.memory.append(dict(states=states, targets_m=np.clip(targets, -.045, .045).tolist(),
                                transverse_gap_m=gap.tolist(), tick=tick))
        self.diagnostics["learned"] = True
        self.refresh()


def assess(feeds, body, bank, plan, output, phase, mode, memory_path):
    if (phase, mode) not in (("practice", "learn"), ("practice", "fixed"), ("carry", "fixed"), ("carry", "carry")):
        raise ValueError("Only practice may train and carry must freeze its memory")
    hashes = dict(body_sha256=_hash(body), bank_sha256=_hash(bank), practice_plan_sha256=_hash(plan))
    memory = None
    if mode == "carry":
        stored = _read(memory_path)
        if any(stored.get(key) != digest for key, digest in hashes.items()):
            raise ValueError("The learned memory belongs to another actor or course")
        memory = stored["points"]
    with np.load(body) as body_state, np.load(bank) as bank_state:
        field = PassageField(bank_state, _read(bank.with_suffix(".json")), body_state["weights"], mode=mode, memory=memory)
    cards = _read(plan)[phase]
    original = deep.observed

    def optical(eye, tick, frames, q):
        data = original(eye, tick, frames, q)
        if frames is not None:
            field.observe(data, eye, tick, q)
        return data

    with patch.object(deep, "observed", optical):
        deep.run(feeds, bank, PLAN, f"body_relative_{phase}", "local", output, cap=200, cards_override=cards,
                 field_hook=lambda _: field, details=dict(body_coordinates=f"passage-{mode}", **hashes))
    if mode == "learn":
        _write(memory_path, dict(points=field.memory, **hashes, tape_sha256=_hash(output)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    phases = p.add_subparsers(dest="command", required=True)
    plan_parser = phases.add_parser("plan")
    plan_parser.add_argument("output", type=Path)
    test = phases.add_parser("assess")
    test.add_argument("output", type=Path)
    test.add_argument("phase", choices=("practice", "carry"))
    test.add_argument("mode", choices=("fixed", "learn", "carry"))
    test.add_argument("--plan", type=Path, default=OUT / "passage-plan.json")
    test.add_argument("--bank", type=Path, default=BANK)
    test.add_argument("--body", type=Path, default=OUT / "body.npz")
    test.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    test.add_argument("--memory", type=Path, default=OUT / "passage-memory.json")
    a = p.parse_args()
    if a.command == "plan":
        schedule(a.output)
    else:
        assess(a.feeds, a.body, a.bank, a.plan, a.output, a.phase, a.mode, a.memory)

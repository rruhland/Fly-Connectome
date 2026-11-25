"""Learn which direct joint feedback works from RGB/encoder context and actual returns."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt42_learning_course as course
from . import tt43_temporal_servo as servo
from . import tt44_rally_course as fresh
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json

ROOT = Path("artifacts/tt45_context")
SOURCES = [course.ROOT / "after12.json.gz", servo.ROOT / "physical.json.gz"]
PRACTICE = ROOT / "earned.json"
DIRECT = course.mpc.MPC


class Approach:
    def __init__(self):
        self.states = deque(maxlen=7)
        self.chosen = False

    def observe(self, x, q):
        if x is None or x[4] >= -0.9:
            self.states.clear()
            if x is not None and x[4] > 0.9:
                self.chosen = False
            return None
        self.states.append(np.r_[x, q[:5]])
        if self.chosen or len(self.states) < 7 or not -0.57 < x[1] < 1.35:
            return None
        self.chosen = True
        # Observe motion in the same time window as the learned motor response.
        return np.r_[self.states[-1], (self.states[-1] - self.states[0])[6:]]


def collect(out, count):
    documents = [_read(path) for path in SOURCES]
    if documents[0]["cards"][:count] != documents[1]["cards"][:count]:
        raise ValueError("Feedback arms did not see assigned matched releases")
    ready = documents[0]["ready"]
    packet = dict(rows=[], inputs={str(p): _hash(p) for p in SOURCES}, complete=False)
    out.parent.mkdir(exist_ok=True, parents=True)
    with course.mpc.centered(ready):
        arena = course.mpc.arena()
        camera, rig, *_ = course.mpc.resources(arena)
        try:
            for index in range(count):
                for arm, document in enumerate(documents):
                    card, row = document["cards"][index], document["rows"][index]
                    arena.launch(row["shot"], card=card)
                    arena.partner.reset(seed=card["seed"])
                    eye = course.mpc.Vision(rig, arena.control_dt, raw=True)
                    eye.eye = course.mpc.LearnedOptics(eye.eye, course.OPTIC)
                    sensor = Approach()
                    decisions = []
                    for tick, u in enumerate(row["actions"]):
                        q = arena.joints
                        rgb = camera.capture() if tick % 2 == 0 else None
                        x = course.mpc.observed(eye, tick, rgb, q)
                        context = sensor.observe(x, q)
                        if context is not None:
                            decisions.append((tick, context))
                        arena.exchange(u, course.mpc.FAR_COMMAND)
                    if digest_json(arena.events) != digest_json(row["events"]):
                        raise RuntimeError("Camera replay changed the original physics")
                    # The evaluator assigns outcomes afterwards. No native event
                    # or shot geometry participates in deciding what to observe.
                    for leg in row["placements"]["delivered"]:
                        if not leg["near_bounce"] or not leg["far_contact"]:
                            continue
                        first = leg["far_contact"]["step"]
                        last = (leg["near_contact"] or row["events"][-1])["step"]
                        matching = [(t, f) for t, f in decisions if first <= t < last]
                        if len(matching) == 1:
                            tick, feature = matching[0]
                            packet["rows"].append(
                                dict(
                                    release=card["id"],
                                    arm=arm,
                                    leg=leg["leg"],
                                    tick=tick,
                                    context=feature.tolist(),
                                    landed=bool(leg["near_landed"]),
                                )
                            )
                    _write(out, packet)
                print(index + 1, len(packet["rows"]), flush=True)
        finally:
            camera.close()
            arena.close()
    packet["complete"] = True
    _write(out, packet)


class Selector:
    def __init__(self, rows):
        self.examples = list(rows)

    def chance(self, context):
        keys = np.array([r["context"] for r in self.examples])
        arms = np.array([r["arm"] for r in self.examples])
        reward = np.array([r["landed"] for r in self.examples], dtype=float)
        scale = np.maximum(
            np.std(keys, axis=0),
            [
                0.07,
                0.10,
                0.07,
                0.28,
                0.40,
                0.33,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.09,
                0.09,
                0.09,
                0.09,
                0.09,
            ],
        )
        distances = np.mean(np.minimum(((keys - context) / scale) ** 2, 12.0), axis=1)
        values = []
        for arm in (0, 1):
            use = np.flatnonzero(arms == arm)
            nearest = use[np.argsort(distances[use], kind="stable")[:18]]
            weights = np.exp(-1.7 * distances[nearest])
            values.append((3.4 + weights @ reward[nearest]) / (4.0 + weights.sum()))
        return np.asarray(values)


def fit_report(out):
    p = _read(PRACTICE)
    if not p["complete"]:
        raise ValueError("Unfinished paired camera replay")
    rows = p["rows"]
    checked = []
    for game in sorted({r["release"] for r in rows}):
        model = Selector([r for r in rows if r["release"] != game])
        for r in rows:
            if r["release"] == game:
                v = model.chance(r["context"])
                checked.append(
                    dict(
                        release=game,
                        arm=r["arm"],
                        probability=v.tolist(),
                        landed=r["landed"],
                    )
                )
    _write(out, dict(examples=len(rows), checked=checked, source=_hash(PRACTICE)))


class Feedback(servo.Temporal):
    def __init__(self, *args, selector, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = selector
        self.priors.on_success = self.earned

    def begin(self):
        super().begin()
        self.approach = Approach()
        self.decision = None
        self.history_enabled = True
        self.decisions = []

    def step(self, tick, eye, x, q, reference, base):
        context = self.approach.observe(x, q)
        if context is not None:
            chance = self.selector.chance(context)
            self.history_enabled = chance[1] + 0.012 >= chance[0]
            self.decision = dict(
                tick=tick,
                context=context.tolist(),
                arm=int(self.history_enabled),
                probabilities=chance.tolist(),
            )
            self.decisions.append(self.decision)
        if self.history_enabled:
            return super().step(tick, eye, x, q, reference, base)
        self.samples.clear()
        return DIRECT.step(self, tick, eye, x, q, reference, base)

    def earned(self, reward):
        if self.decision is not None:
            self.decision["landed"] = bool(reward)
            self.selector.examples.append(self.decision.copy())
            self.decision = None

    def missed(self, tape):
        self.earned(False)


class Memory(course.Carried):
    def keep(self, archive, leg):
        n = super().keep(archive, leg)
        self.on_success(True)
        return n


def play(a):
    data = _read(PRACTICE)
    if not data["complete"]:
        raise ValueError("Incomplete practice bank")
    cards = _read(fresh.PLAN)[a.deck][: a.n]
    if {r["release"] for r in data["rows"]} & {c["id"] for c in cards}:
        raise ValueError("Camera practice overlaps assessment")
    selector = Selector(data["rows"])
    native_play = course.mpc.play

    def tracked(*args, **kwargs):
        result = native_play(*args, **kwargs)
        result["selected_feedback"] = args[10].decisions
        result["context_memory_rows"] = len(selector.examples)
        return result

    with (
        patch.object(course, "PLAN", fresh.PLAN),
        patch.object(course, "Carried", Memory),
        patch.object(
            course.mpc,
            "MPC",
            lambda *args, **kw: Feedback(*args, selector=selector, **kw),
        ),
        patch.object(course.mpc, "play", tracked),
    ):
        course.run(a)
    result = _read(a.out)
    result.update(
        selector_source=_hash(PRACTICE), selector_checkpoint=selector.examples
    )
    _write(a.out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("collect", "check", "play"))
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="assessment", mode="live", cap=240, carry=fresh.CARRY, resume=False
    )
    a = p.parse_args()
    if a.phase == "collect":
        collect(a.out, a.n)
    elif a.phase == "check":
        fit_report(a.out)
    else:
        play(a)

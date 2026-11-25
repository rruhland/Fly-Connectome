"""Attribute real carried-MPC CPU time and compare an exact-reuse physical run."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried
from .tt45_rollout_reuse import ReuseMPC

OUT = Path("artifacts/tt45_efficiency")
CARRY = course.ROOT / "learning-24.json.gz"
STOCK = Path("artifacts/tt43_motor_modes/held-stock.json.gz")
FEEDS = Path("/tmp/tt25-rallies.json")


class Profile:
    def __init__(self):
        self.seconds = Counter()
        self.counts = Counter()
        self.shapes = Counter()

    def timed(self, name, fn, *, native=False):
        def call(*a, **kw):
            t = process_time()
            ans = fn(*a, **kw)
            self.seconds[name] += process_time() - t
            self.counts[name] += 1
            if native:
                count, frames = a[3:5]
                self.shapes[(count, frames)] += 1
                self.counts["native_rows"] += count
            return ans
        return call

    def wrap(self, table, renderer, model, raw):
        entries = (
            (table, "exchange", "actual_exchange"), (renderer, "capture", "stereo_render"),
            (mpc, "observed", "optical_fit"), (model.engine, "propagate", "body_propagate"),
            (model.engine, "sync", "sync"), (model.priors, "priors", "retrieve"),
            (raw, "get", "direct_field"), (model.engine.predict, "trial", "native"),
        )
        stack = ExitStack()
        for obj, field, label in entries:
            stack.enter_context(patch.object(obj, field, self.timed(label, getattr(obj, field), native=label=="native")))
        return stack

    def result(self):
        return {
            "cpu": self.seconds, "calls": self.counts,
            "native_batch_histogram": [
                dict(count=n, frames=f, calls=k) for (n, f), k in sorted(self.shapes.items())
            ],
        }


def parity(old, new):
    for key in ("actions", "events", "placements", "score", "searches", "memory", "motor_audit"):
        if digest_json(old[key]) != digest_json(new[key]):
            raise ValueError(f"Changed physical output: {key}")


def run(a):
    deck = _read(course.PLAN)["assessment"]
    cards = [deck[i] for i in a.indices]
    archived = _read(a.stock)
    if (archived["cap"] != 240 or archived["mode"] != "frozen"
            or archived["source_sha256"] != _hash(mpc.BANK)
            or archived["carry_sha256"] != _hash(a.carry)
            or archived["optical_sha256"] != _hash(course.OPTIC)
            or archived["variant"] != "elastic"):
        raise ValueError("Archived stock control settings differ")
    ref = {c["id"]: (c, row) for c, row in zip(archived["cards"], archived["rows"])}
    for c in cards:
        if c != ref[c["id"]][0]:
            raise ValueError("Assessment card changed")
    carry = _read(a.carry)
    checkpoint = a.carry.with_suffix("").with_suffix(".npz")
    feeds = mpc.measured_launches(a.feeds)
    ready = archived["ready"]
    with np.load(mpc.BANK) as f:
        raw = mpc.DeepField(f, "local")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "cards": cards, "cap": 240, "mode": "frozen", "repeats": a.repeats,
        "stock_sha256": _hash(a.stock), "carry_sha256": _hash(a.carry),
        "checkpoint_sha256": _hash(checkpoint), "source_sha256": {
            p.name: _hash(p) for p in (Path(__file__), Path(mpc.__file__),
                                      Path(__file__).with_name("tt45_rollout_reuse.py"),
                                      Path(__file__).with_name("tt29_rally_rollout.cpp"))
        },
        "fixture": ready, "profile_clock": "process CPU seconds with single native worker",
        "complete": False, "rows": [],
    }
    with mpc.centered(ready):
        table = mpc.arena()
        image, rig, stride, lo, hi = mpc.resources(table)
        with patch.object(mpc, "Fragments", lambda path, engine: Carried(
                path, engine, practice=[course.PRACTICE], held={c["id"] for c in cards})):
            model = ReuseMPC(mpc.BANK, raw, variant="elastic")
        try:
            course.resume(model, checkpoint, carry)
            for j in range(a.repeats):
                for i, c in enumerate(cards):
                    expected = ref[c["id"]][1]
                    order = ("stock", "reuse") if (i + j) % 2 == 0 else ("reuse", "stock")
                    for mode in order:
                        model.reuse, model.audit_prefix = mode == "reuse", mode == "stock"
                        profile = Profile()
                        with profile.wrap(table, image, model, raw):
                            wall = perf_counter()
                            first_launch = table.launches
                            row = mpc.play(table, image, rig, stride, lo, hi, ready, c, feeds[c["id"]],
                                           raw, model, course.OPTIC, "frozen", doc["cap"])
                            wall = perf_counter() - wall
                        if table.launches != first_launch + 1:
                            raise ValueError("Unexpected physical launch count")
                        parity(expected, row)
                        if any(list(draw(c, k, "random", None)[0]) != move["target"] for k, move in enumerate(row["placements"]["requested"])):
                            raise ValueError("Changed broad opponent draw")
                        doc["rows"].append(dict(id=c["id"], mode=mode, repeat=j, ratio_order=list(order),
                                                scores=row["score"], runtime_cpu_s=row["runtime_cpu_s"],
                                                runtime_wall_s=wall, profile=profile.result(),
                                                planner=row["planner"],
                                                actions_sha256=digest_json(row["actions"]),
                                                events_sha256=digest_json(row["events"]),
                                                physically_matches_archive=True))
                        _write(a.output, doc)
                        print(c["id"], mode, row["score"]["learner"], "CPU", round(row["runtime_cpu_s"], 3),
                              "native", round(profile.seconds["native"], 3), "rows", profile.counts["native_rows"],
                              "saved", row["planner"].get("reused_native_candidates", 0), flush=True)
            doc["complete"] = True
            _write(a.output, doc)
        finally:
            model.close()
            image.close()
            table.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--indices", type=int, nargs="+", default=[2, 6])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--carry", type=Path, default=CARRY)
    p.add_argument("--stock", type=Path, default=STOCK)
    p.add_argument("--feeds", type=Path, default=FEEDS)
    run(p.parse_args())

"""Measure the camera, controller and simulator costs of one verified regulation point."""

import argparse
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED
from .tt26_grip_course import Grip, ReferenceReceiver, load_match
from .tt27_flight_feedback import OUT
from .tt27_full_points import one
from .tt27_reply import DEV_DONORS, Flight, PoseMap, Recall, examples
from .tt27_reply_recorded import transported


def timed(fn, samples):
    def measure(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        samples.append((perf_counter() - start) * 1000)
        return result

    return measure


def stats(rows):
    return dict(
        n=len(rows),
        mean_ms=float(np.mean(rows)),
        median_ms=float(np.median(rows)),
        p95_ms=float(np.percentile(rows, 95)),
        max_ms=float(np.max(rows)),
        above_33ms=int(np.sum(np.array(rows) > 1000 / 30)),
    )


def run(path):
    shots, first = load_match(path)
    receipt = _read(OUT / "full-points.json.gz")
    row = max(receipt["rows"], key=lambda r: (r["by"]["repeat"]["near"], r["by"]["repeat"]["far"]))
    target = row["by"]["repeat"]
    arena = ReferenceReceiver(stationary_opponent=False)
    arena.max_steps = 450
    install_rally_opponent(arena, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
    tape = FarTape(arena)
    poses, flight = PoseMap(arena), Flight()
    src = [e for e in examples(_read(OUT / "reply-traces.json.gz")["traces"]) if e["id"] in DEV_DONORS]
    memory = transported(src, Grip(arena.model), poses, 0.0)
    camera, control, physics = [], [], []
    try:
        with (
            patch.object(arena, "render_cameras", timed(arena.render_cameras, camera)),
            patch.object(arena, "step", timed(arena.step, physics)),
            patch.object(Recall, "act", timed(Recall.act, control)),
        ):
            start = perf_counter()
            result = one(arena, tape, shots[row["id"]], first, memory, poses, flight, None, "repeat")
            seconds = perf_counter() - start
        if result["actions"] != target["actions"] or digest_json(result["events"]) != digest_json(
            target["events"]
        ):
            raise ValueError("Timed control loop changed the recorded point")
        active = [v for t, v in enumerate(control) if t >= result["triggers"][0]]
        info = dict(
            id=row["id"],
            frames=result["frames"],
            simulation_s=result["frames"] / 30,
            wall_s=seconds,
            camera=stats(camera),
            reply_compute=stats(control),
            after_reply_visible=stats(active),
            physics_and_opponent=stats(physics),
        )
        _write(OUT / "control-clock.json", info)
        print(info, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    run(parser.parse_args().feeds)

"""Learn a short visual extrapolation gain in real-feed match play, then bounce."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_pong_temporal_wrist import run as wrist_run, starts, variant
from .tt24_rally_policy import orange_center
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def fit(path):
    source = OUTPUT / "recorded-live-development.json.gz"
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    arena = ReferenceReceiver()
    arena.max_steps = 450
    rng = np.random.default_rng(5719039)
    design, outcomes, shuffled = [], [], []
    counts = []
    try:
        for row in _read(source)["cases"]:
            arena.launch(shots[row["id"]])
            sequence = [[], []]
            previous = [None, None]
            for near, far in zip(row["actions"], row["rival_actions"], strict=True):
                frames = arena.render_cameras()
                for j, name in enumerate(("left", "right")):
                    p = orange_center(frames[name], previous[j])
                    sequence[j].append([np.nan, np.nan] if p is None else p)
                    if p is not None:
                        previous[j] = p
                arena._aim = arena._command_to = lambda *args, far=far: np.asarray(far)
                arena.step(near, render=False)
            if digest_json(arena.events) != digest_json(row["events"]):
                raise ValueError("Source optical stream did not reproduce its point")
            total = 0
            for series in map(np.array, sequence):
                for gap in (2, 3, 4):
                    velocity = (series[2:-gap] - series[: -gap - 2]) / 2
                    displacement = series[gap + 2 :] - series[2:-gap]
                    valid = np.isfinite(velocity).all(axis=1) & np.isfinite(displacement).all(axis=1)
                    expected = gap * velocity[valid]
                    observed = displacement[valid]
                    design.extend(expected)
                    outcomes.extend(observed)
                    shuffled.extend(observed[rng.permutation(len(observed))])
                    total += len(expected)
            counts.append(dict(source_id=row["id"], frames=len(row["actions"]), examples=total))
    finally:
        arena.close()
    x = np.asarray(design)
    # One dimensionless time gain, fitted without wrist exposure.
    gains = {
        name: float(np.clip(np.sum(x * np.array(y)) / np.sum(x * x), 0.0, 2.0))
        for name, y in [("tennis", outcomes), ("shuffled", shuffled)]
    }
    receipt = dict(
        source_sha256=_hash(source), counts=counts, gains=gains, examples=len(x), target_practice=0
    )
    _write(OUTPUT / "recorded-delay-visual-fit.json", receipt)
    return gains


def run(path):
    gains = fit(path)
    controls = dict(
        tennis=variant(gains["tennis"]),
        shuffled=variant(gains["shuffled"]),
        cold=variant(0.0),
        unit=variant(1.0),
    )
    wrist_run(
        OUTPUT / "recorded-delayed-wrist.json.gz",
        starts(28461173, 12),
        controls,
        dict(source_sha256=_hash(OUTPUT / "recorded-delay-visual-fit.json"), target_gain_practice=0),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)

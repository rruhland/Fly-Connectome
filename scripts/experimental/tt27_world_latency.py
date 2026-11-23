"""Check a previously learned visual time gain on truly delayed full-size receiving."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_rally_policy import EYES, ViewHistory, orange_center
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_receiving_speed import draw
from .tt27_world_camera import CameraMounts, FEEDS, POSES, hit, watch


GAIN = Path("artifacts/tt25/recorded-delay-visual-fit.json")
OUT = Path("artifacts/tt27/world-latency.json.gz")
SEED = 83037667
SHOTS_PER_DELAY = 24
LAGS = (1, 2)


def centers(views):
    pixels = []
    previous = dict.fromkeys(EYES)
    for frames, _ in views:
        current = []
        for eye in EYES:
            point = orange_center(frames[eye], previous[eye])
            if point is not None:
                previous[eye] = point
            current.append(point)
        pixels.append(current)
    return pixels


def forecast(delivered, lag, gain):
    """Use only the captured observations delivered by the action decision."""
    tail = []
    last = len(delivered) - 1
    for offset in range(1, lag + 1):
        frame = []
        for eye in range(len(EYES)):
            sightings = [(t, p[eye]) for t, p in enumerate(delivered) if p[eye] is not None]
            point = None
            if len(sightings) >= 2:
                older_tick, older = sightings[-2]
                tick, recent = sightings[-1]
                velocity = (recent - older) / (tick - older_tick)
                point = recent + gain * (last + offset - tick) * velocity
            frame.append(point)
        tail.append(frame)
    return delivered + tail


def key(views, pixels):
    history = ViewHistory()
    for (_, joints), observed in zip(views, pixels, strict=True):
        state = history.observe_pixels(observed, joints, joints)
    return state[:14]


def run(path):
    with np.load(POSES, allow_pickle=False) as data:
        keys, goals, scale = (data[k] for k in ("keys", "goals", "scale"))
    sensors = dict(prompt=None, stale=None, unit=1., source=_read(GAIN)["gains"]["tennis"])
    donors = {r["shot"]["id"]: r["shot"] for r in _read(FEEDS)["cases"] if r["angle"] < 0}
    rng = np.random.default_rng(SEED)
    arena = ReferenceReceiver()
    mounts = CameraMounts(arena)
    report = dict(schema="tt27-world-latency-1", pose_source_sha256=_hash(POSES),
                  feed_source_sha256=_hash(FEEDS), source_gain_sha256=_hash(GAIN),
                  gains=sensors, seed=SEED, shots_per_delay=SHOTS_PER_DELAY, cases=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for lag in LAGS:
            rows = []
            for index in range(SHOTS_PER_DELAY):
                shot = draw(donors, sorted(donors), rng, index)
                views = watch(arena, shot, mounts, ((0, 0, 0),) * 2)
                prompt = centers(views)
                # No camera packet is available until the bounded delay has elapsed.
                delivered = prompt[:max(0, len(prompt) - lag)]
                inputs = dict(prompt=prompt, stale=[[None] * len(EYES)] * min(len(prompt), lag) + delivered)
                for name in ("unit", "source"):
                    inputs[name] = forecast(delivered, lag, sensors[name]) if delivered else inputs["stale"]
                results, physical = {}, {}
                for name, observations in inputs.items():
                    observed = key(views, observations)
                    choice = int(np.argmin(np.linalg.norm((keys - observed) / scale, axis=1)))
                    if choice not in physical:
                        trial = hit(arena, shot, goals[choice])
                        if hit(arena, shot, goals[choice], trial["actions"]) != trial:
                            raise ValueError(f"Physical replay changed for lag {lag}, trial {index}")
                        physical[choice] = trial
                    results[name] = dict(action=choice, **physical[choice])
                rows.append(dict(shot=shot, results=results, verified_choices=len(physical)))
            summary = {}
            for name in inputs:
                legal = [r["results"][name]["legal"] for r in rows]
                delta = np.asarray([v - r["results"]["stale"]["legal"] for v, r in zip(legal, rows, strict=True)])
                summary[name] = dict(legal=sum(legal),
                                     contacts=sum(r["results"][name]["contact"] for r in rows),
                                     gains_vs_stale=int((delta > 0).sum()), losses_vs_stale=int((delta < 0).sum()))
            report["cases"].append(dict(lag_ticks=lag, rows=rows, summary=summary))
            _write(path, report)
            print(lag, summary, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    run(parser.parse_args().output)

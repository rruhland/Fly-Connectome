"""Fit a camera-only pre-impact prediction of the next crossing from the opponent's silhouette."""

import argparse
import functools
import operator
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_rally_physics import measured_launches
from .tt29_vision import Rig
from .tt31_curriculum import card
from .tt31_curriculum_opponent import replay_arena


def stereo(frames, cameras):
    projections = []
    for name, frame in frames.items():
        sight = native_ball_pixels(frame)
        if sight is None:
            return None
        camera = cameras[name]
        ray = camera.ray(sight.center)
        ray /= np.linalg.norm(ray)
        projector = np.eye(3) - np.outer(ray, ray)
        projections.append((projector, projector @ camera.eye))
    return np.linalg.lstsq(
        np.sum([p for p, _ in projections], axis=0),
        np.sum([b for _, b in projections], axis=0),
        rcond=None,
    )[0]


def silhouette(image, camera):
    # Color and far-player region are calibrated perceptual scaffolding.
    u, v = camera.project([0.0, 1.45, 1.04])[:2].astype(int)
    patch = image[v - 27 : v + 27, u - 54 : u + 54].astype(np.int16)
    r, g, b = np.moveaxis(patch, -1, 0)
    mask = (
        (r > 20)
        & (r < 125)
        & (b - r >= 3)
        & (b - r <= 15)
        & (g - r >= 0)
        & (g - r <= 6)
        & (b - g >= 1)
    )
    y, x = np.nonzero(mask)
    if len(x) < 7:
        return None
    xx, yy = x - x.mean(), y - y.mean()
    return [
        float(x.mean()),
        float(y.mean()),
        float(len(x)),
        float(np.mean(xx * xx)),
        float(np.mean(xx * yy)),
        float(np.mean(yy * yy)),
    ]


def collect(feeds, output, count):
    shoots = measured_launches(feeds)
    train = [12330, 12451, 13140, 11909]
    test = [10898, 3855, 14183, 13159]
    out = {"feed_sha256": _hash(feeds), "rows": []}
    arena = replay_arena([])
    renderer = CachedStereoRenderer(arena)
    cameras = Rig.from_arena(arena).fixed
    try:
        for i in range(count):
            phase = "train" if i % 3 != 2 else "test"
            idx = (i - i // 3) if phase == "train" else i // 3
            sid = (train if phase == "train" else test)[idx % 4]
            c = card(sid, "full", i, 320500 + i)
            arena.launch(shoots[sid], card=c)
            arena.partner.reset(seed=c["seed"])
            captures = {}
            prev = None
            label = None
            for k in range(138):
                t = k * arena.control_dt
                if k % 2 == 0:
                    frames = renderer.capture()
                    p = stereo(frames, cameras)
                    if p is not None and prev is not None:
                        then, old = prev
                        for key, depth in [("early", 0.70), ("late", 1.10)]:
                            if key not in captures and old[1] < depth <= p[1]:
                                shape = [
                                    silhouette(img, cameras[name])
                                    for name, img in frames.items()
                                ]
                                if all(v is not None for v in shape):
                                    captures[key] = {
                                        "t": t,
                                        "ball": np.r_[
                                            p, (p - old) / (t - then)
                                        ].tolist(),
                                        "bat": functools.reduce(
                                            operator.iadd, shape, []
                                        ),
                                    }
                        if old[1] > -1.10 >= p[1] and t - then <= 0.07:
                            label = float(
                                old[0]
                                + (p[0] - old[0]) * (-1.10 - old[1]) / (p[1] - old[1])
                            )
                            break
                    if p is not None:
                        prev = t, p
                _, _, end, cap, _ = arena.exchange(np.zeros(6), np.zeros(5))
                if end or cap:
                    break
            hit = next(
                (
                    e["time"]
                    for e in arena.events
                    if e["kind"] == "contact" and e["side"] == -1
                ),
                None,
            )
            physical = arena.crossings.get(1)
            out["rows"].append(
                {
                    "id": sid,
                    "seed": c["seed"],
                    "phase": phase,
                    "captures": captures,
                    "visual_x": label,
                    "actual_x": physical["xyz"][0] if physical else None,
                    "hit_s": hit,
                }
            )
            if (i + 1) % 8 == 0:
                _write(output, out)
                print(i + 1, flush=True)
    finally:
        renderer.close()
        arena.close()
    _write(output, out)


def scores(data):
    summary = {}
    for gate in ("early", "late"):
        # The event time below is used only to audit that every RGB input predates impact.
        rows = [
            r
            for r in data["rows"]
            if gate in r["captures"]
            and r["hit_s"] is not None
            and r["actual_x"] is not None
            and r["visual_x"] is not None
            and r["captures"][gate]["t"] < r["hit_s"]
        ]
        training = [r for r in rows if r["phase"] == "train"]
        tests = [r for r in rows if r["phase"] == "test"]
        if not training or not tests:
            raise ValueError(f"Insufficient camera episodes at {gate}")
        y = np.array([r["visual_x"] for r in training])
        truth = np.array([r["actual_x"] for r in tests])
        result = {
            "train": len(y),
            "test": len(truth),
            "test_openings": sorted({r["id"] for r in tests}),
            "median_lead_s": float(
                np.median([r["hit_s"] - r["captures"][gate]["t"] for r in tests])
            ),
            "constant_mae_m": float(np.mean(abs(truth - np.mean(y)))),
        }
        for use_bat in (False, True):

            def features(r, phase=gate, include=use_bat):
                c = r["captures"][phase]
                return c["ball"] + (c["bat"] if include else [])

            x = np.array([features(r) for r in training])
            v = np.array([features(r) for r in tests])
            scale = np.maximum(x.std(axis=0), 1e-5)
            z = (x - x.mean(axis=0)) / scale
            beta = np.linalg.solve(
                z.T @ z + 5.0 * np.eye(z.shape[1]), z.T @ (y - y.mean())
            )
            prediction = y.mean() + (v - x.mean(axis=0)) / scale @ beta
            key = "with_bat" if use_bat else "ball_only"
            result[key] = {
                "mae_m": float(np.mean(abs(prediction - truth))),
                "predictions": prediction.tolist(),
            }
        summary[gate] = result
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "score"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=96)
    a = parser.parse_args()
    if a.mode == "collect":
        collect(a.source, a.output, a.count)
    else:
        _write(a.output, scores(_read(a.source)))

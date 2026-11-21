"""Fit free-flight coefficients to real SpinDOE shots, then score disjoint shot IDs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import least_squares


SOURCE_COMMIT = "ebc8a9eac32e42cdbe2978767ef0134d67dfc123"
GRAVITY = np.array([0.0, 0.0, -9.81])


def read_shots(path):
    rows = []
    with zipfile.ZipFile(path) as data:
        for row in csv.DictReader(io.StringIO(data.read("index.csv").decode())):
            shot = int(row["traj_file"])
            points = np.loadtxt(io.BytesIO(data.read(f"{shot:03d}.csv")), delimiter=";")
            times = points[:, 0] - points[0, 0]
            prefix = times <= 0.050
            endpoint = np.searchsorted(times, 0.150)
            # Stay in free flight, with enough frames to initialize velocity.
            if endpoint >= len(times) or times[endpoint] > 0.162 or prefix.sum() < 5:
                continue
            if points[:endpoint + 1, 3].min() <= 0.05 or np.any(np.diff(times[:endpoint + 1]) <= 0):
                continue
            obs = points[prefix, 1:]
            centered = times[prefix] - times[prefix].mean()
            mean = obs.mean(axis=0)
            slope = (centered @ obs) / (centered @ centered)
            # Least-squares intercept and slope corrections for initial curvature.
            c0 = (centered**2).mean() * 0.5
            c1 = centered.dot(centered**2) / (2 * centered.dot(centered))
            rows.append((shot, mean, slope, c0, c1, times[endpoint] - times[prefix].mean(),
                         [float(row[k]) for k in ("x_spin", "y_spin", "z_spin")], points[endpoint, 1:]))
    return tuple(np.asarray(col) for col in zip(*rows, strict=True))


def measured_bounces(path):
    """Estimate the normal restitution of spins/impacts observed on the table."""
    rebounds = []
    with zipfile.ZipFile(path) as archive:
        for row in csv.DictReader(io.StringIO(archive.read("index.csv").decode())):
            shot = int(row["traj_file"])
            pts = np.loadtxt(io.BytesIO(archive.read(f"{shot:03d}.csv")), delimiter=";")
            times, height = pts[:, 0], pts[:, 3]
            for i in range(5, len(pts) - 5):
                if height[i] > 0.055 or height[i] != min(height[i-4:i+5]):
                    continue
                if abs(pts[i, 1]) > 0.75 or abs(pts[i, 2]) > 1.36:
                    continue

                def vertical_speed(window):
                    dt = times[window] - times[i]
                    return np.polyfit(dt, height[window] + 0.5 * 9.81 * dt**2, 1)[0]

                before = vertical_speed(slice(i-5, i))
                after = vertical_speed(slice(i+1, i+6))
                if before < -0.65 and after > 0.5:
                    rebounds.append((before, after, -after / before))
                    break
    ratios = np.asarray(rebounds)[:, 2]
    return {"count": len(ratios), "median_normal_ratio": float(np.median(ratios)),
            "normal_ratio_p10_p90": np.quantile(ratios, [0.1, 0.9]).tolist()}


def initialize(data, coeff):
    _, mean, slope, c0, c1, horizon, spin, endpoint = data

    def accel(v):
        return GRAVITY - coeff[0] * np.linalg.norm(v, axis=1)[:, None] * v + coeff[1] * np.cross(spin, v)

    v = slope.copy()
    for _ in range(2):
        v = slope - c1[:, None] * accel(v)
    pos = mean - c0[:, None] * accel(v)
    return pos, v, accel


def predict(data, coeff):
    horizon, endpoint = data[5], data[7]
    pos, v, accel = initialize(data, coeff)
    dt = horizon[:, None] / 64
    for _ in range(64):
        vm = v + 0.5 * dt * accel(v)
        pos += dt * vm
        v += dt * accel(vm)
    return pos - endpoint


def engine_predict(data, coeff):
    import mujoco
    from .tt4_arena import TT4Arena

    pos, vel, _ = initialize(data, coeff)
    arena = TT4Arena()
    err = []
    # Translate the whole free-flight segment upward to isolate it from furniture.
    offset = np.array([0., 0., 10.])
    for start, velocity, spin, horizon, endpoint in zip(pos, vel, data[6], data[5], data[7], strict=True):
        arena.reset(render=False)
        arena.ball[:] = start + offset
        arena.velocity[:] = velocity
        q = arena.model.jnt_dofadr[arena.model.joint("ball_free").id]
        arena.data.qvel[q+3:q+6] = spin
        mujoco.mj_forward(arena.model, arena.data)
        while arena.data.time < horizon:
            previous = arena.ball.copy()
            arena._aerodynamics()
            mujoco.mj_step(arena.model, arena.data)
        fraction = (arena.data.time - horizon) / arena.model.opt.timestep
        prediction = arena.ball + (previous - arena.ball) * fraction
        err.append(prediction - offset - endpoint)
    arena.close()
    return np.asarray(err)


def audit(path, engine=False):
    data = read_shots(path)
    shot = data[0]
    subset = np.array([int.from_bytes(hashlib.sha256(str(i).encode()).digest()[:4], "big") % 3 for i in shot])
    train = tuple(x[subset != 0] for x in data)
    test = tuple(x[subset == 0] for x in data)

    def fit(has_spin):
        def residual(x):
            return predict(train, [x[0], x[1] if has_spin else 0.0]).ravel()
        init = [0.10, 0.001] if has_spin else [0.10]
        return least_squares(residual, init, bounds=(0.0, np.inf), loss="soft_l1", f_scale=0.01).x

    drag = fit(False)
    both = fit(True)
    modes = {"gravity_only": [0.0, 0.0], "drag_only": [float(drag[0]), 0.0], "drag_and_spin": both.tolist()}
    scores = {}
    for mode, coeff in modes.items():
        err = 1000 * np.linalg.norm(predict(test, coeff), axis=1)
        scores[mode] = {"coefficients": coeff, "median_mm": float(np.median(err)),
                        "mean_mm": float(err.mean()), "p90_mm": float(np.quantile(err, 0.90))}
    if engine:
        err = 1000 * np.linalg.norm(engine_predict(test, both), axis=1)
        scores["mujoco_free_flight"] = {"median_mm": float(np.median(err)),
                                        "mean_mm": float(err.mean()), "p90_mm": float(np.quantile(err, .9))}
    return {"dataset": "cogsys-tuebingen/spindoe", "commit": SOURCE_COMMIT,
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "fit_shots": train[0].tolist(), "test_shots": test[0].tolist(),
            "test_initial_speed_mps_range": [float(f(np.linalg.norm(test[2], axis=1))) for f in (min, max)],
            "scores": scores, "bounce_reference": measured_bounces(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", action="store_true", help="score the actual TT4 MuJoCo steps as well")
    args = parser.parse_args()
    result = audit(args.zip, args.engine)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["scores"], indent=2))

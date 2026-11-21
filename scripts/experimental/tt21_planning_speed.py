"""Measure receiver shortlist speed and verify identical candidates on real optical windows."""

from time import process_time
import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_catalog import candidate_costs
from .tt19_flight import optical_start
from .tt19_model_rollout import HORIZON, COURT, reachable_tapes
from .tt20_camera_carry import OUTPUT as TT20, Station
from .tt20_curriculum import configuration

OUTPUT = TT20.parent / "tt21"


def observations(game, law):
    windows = []
    for row in game["points"]:
        history = []
        for tick, xyz in row["optical"]:
            history.append((tick, xyz))
            start = optical_start(history, law.gravity)
            if start is not None and -0.81 < xyz[1] < -0.24 and start[1][1] < -0.031:
                windows.append(start)
    return windows


def surface_rank(poses, law, response, start, filtered):
    points, speeds = law.trajectory(*start, HORIZON, bounds=COURT)
    indices = reachable_tapes(poses, points) if filtered else np.arange(len(poses))
    if not indices.size:
        return np.empty((0, 4)), 0
    rows = [
        candidate_costs(poses[batch], law, response, points, speeds, False)
        for batch in np.array_split(indices, (len(indices) + 255) // 256)
    ]
    costs, eta, hit, _, _, _, _, ball = map(np.concatenate, zip(*rows, strict=True))
    ok = hit & (ball[:, 1] < -0.78) & (ball[:, 2] > 0.03)
    return np.column_stack([indices[ok], costs[ok], eta[ok], ball[ok, 1]]), len(indices)


def run():
    source = "camera-carry-left-game-40582930-1.json.gz"
    game = _read(TT20 / source)
    _, catalogue, law, response, _ = configuration(TT20)
    poses = catalogue["poses"]
    windows = observations(game, law)
    windows = windows[:: max(1, len(windows) // 36)][:36]
    totals = [0.0, 0.0]
    examined = [0, 0]
    legal_candidates = 0
    for i, start in enumerate(windows):
        results = {}
        for flag in (True, False) if i % 2 else (False, True):
            t = process_time()
            result, count = surface_rank(poses, law, response, start, flag)
            totals[flag] += process_time() - t
            examined[flag] += count
            results[flag] = result
        if not np.array_equal(results[True], results[False]):
            raise ValueError("reachable tape IDs, timings, positions or costs changed")
        legal_candidates += len(results[True])
    print("geometry", len(windows), totals, examined, flush=True)
    station = Station("left")
    frames = 0
    try:
        for row in game["points"]:
            restored = station.point(row["seed"], row["role"], game["candidate"])
            for key in ("rgb_sha256", "actions", "outcome", "events"):
                import json

                if json.dumps(row[key]) != json.dumps(restored[key]):
                    raise ValueError(f"{key} diverged at {row['seed']}")
            frames += len(row["actions"])
    finally:
        station.close()
    _write(
        OUTPUT / "planning-speed.json",
        dict(
            source=source,
            windows=len(windows),
            cpu_s=totals,
            candidates_scored=examined,
            exact_feasible=legal_candidates,
            verified_frames=frames,
            native_points=len(game["points"]),
            speedup=totals[0] / totals[1],
        ),
    )


if __name__ == "__main__":
    run()

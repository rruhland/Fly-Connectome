"""Check flight, pace and spin against independent Sony Ace match recordings.

Usage: python -m scripts.experimental.tt25_recorded_flight /path/to/match_data.csv
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .tt4_arena import AIR_DRAG, BALL_RADIUS, GRAVITY, MAGNUS_COEFFICIENT, SPIN_DECAY

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "tt25"
SOURCE = "57ac16a42c25c47bde24cd2a592804f4b800085b"
SHA256 = "e9ee2fc299b6b2358f4b43ca770cc7201210443f0135934122a5234b7f8860d5"
HELD = {"Pro 1", "Pro 2"}


@dataclass
class Flight:
    player: str
    key: str
    start: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray
    duration: float
    end: np.ndarray


def vector(row, field):
    value = np.fromstring(row[field].strip("[]"), sep=" ")
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {field}")
    return value


def records(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Expected the pinned Sony Ace match_data.csv")
    rallies = defaultdict(list)
    with source.open(newline="") as file:
        for row in csv.DictReader(file):
            key, number = row["rally_id"].rsplit("/event", 1)
            rallies[row["player_id"], key].append((int(number), row))
    flights, speeds, spins, intervals = [], [], [], []
    for (player, key), entries in rallies.items():
        entries.sort()
        strokes = [(number, row) for number, row in entries if row["type"].startswith("shot_")]
        # Exclude the opening service. Keep failed and successful rally shots for pace.
        if player in HELD:
            speeds.extend(np.linalg.norm(vector(r, "ball_vel_out")) for _, r in strokes[1:])
            spins.extend(np.linalg.norm(vector(r, "ball_spin_out")) for _, r in strokes[1:])
            intervals.extend(
                float(b["timestamp"]) - float(a["timestamp"])
                for (_, a), (_, b) in zip(strokes[1:], strokes[2:])
                if a["type"] != b["type"]
            )
        if not strokes:
            continue
        for (a, hit), (b, land) in zip(entries, entries[1:]):
            if (
                a <= strokes[0][0]
                or not hit["type"].startswith("shot_")
                or not land["type"].startswith("bounce_")
                or b != a + 1
            ):
                continue
            dt = float(land["timestamp"]) - float(hit["timestamp"])
            if dt <= 0:
                raise ValueError("Nonpositive flight time")
            flights.append(
                Flight(
                    player,
                    f"{key}/event{a}",
                    vector(hit, "ball_pos"),
                    vector(hit, "ball_vel_out"),
                    vector(hit, "ball_spin_out"),
                    dt,
                    vector(land, "ball_pos"),
                )
            )
    return flights, dict(speed=speeds, spin=spins, interval=intervals)


def propagate(rows, coeff, *, predict_landing=False):
    """Midpoint free-flight integration in the dataset's right-handed SI frame."""
    p = np.stack([r.start for r in rows])
    v = np.stack([r.velocity for r in rows])
    spin = np.stack([r.spin for r in rows])
    duration = np.array([r.duration for r in rows])
    count = 500 if predict_landing else 64
    dt = np.full((len(rows), 1), 0.003) if predict_landing else duration[:, None] / count
    landing = np.full((len(rows), 2), np.nan)
    landing_time = np.full(len(rows), np.nan)
    decay = np.exp(-SPIN_DECAY * dt / 2)

    def accel(velocity, omega):
        air = -coeff[0] * np.linalg.norm(velocity, axis=1, keepdims=True) * velocity + coeff[1] * np.cross(
            omega, velocity
        )
        air[:, 2] -= GRAVITY
        return air

    for tick in range(count):
        mid = v + dt / 2 * accel(v, spin)
        end = p + dt * mid
        v = v + dt * accel(mid, spin * decay)
        spin *= decay**2
        if predict_landing:
            cross = np.isnan(landing_time) & (p[:, 2] >= BALL_RADIUS) & (end[:, 2] < BALL_RADIUS)
            fraction = (p[cross, 2] - BALL_RADIUS) / (p[cross, 2] - end[cross, 2])
            landing[cross] = p[cross, :2] + fraction[:, None] * (end[cross, :2] - p[cross, :2])
            landing_time[cross] = (tick + fraction) * dt[cross, 0]
            if np.isfinite(landing_time).all():
                break
        p = end
    return (landing, landing_time) if predict_landing else p


def quantiles(values):
    a = np.asarray(values)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Empty or nonfinite metric")
    return dict(
        n=len(a), p10=float(np.quantile(a, 0.1)), median=float(np.median(a)), p90=float(np.quantile(a, 0.9))
    )


def score(rows, coeff):
    endpoints = np.stack([r.end for r in rows])
    pred = propagate(rows, coeff)
    predicted, times = propagate(rows, coeff, predict_landing=True)
    ok = np.isfinite(times)
    t = np.array([r.duration for r in rows])
    return dict(
        coefficients=list(coeff),
        shots=len(rows),
        no_downward_crossing=int((~ok).sum()),
        endpoint_at_recorded_time_cm=quantiles(100 * np.linalg.norm(pred - endpoints, axis=1)),
        landing_error_cm=quantiles(100 * np.linalg.norm(predicted[ok] - endpoints[ok, :2], axis=1)),
        landing_time_error_ms=quantiles(1000 * np.abs(times[ok] - t[ok])),
    )


def compact_court():
    speed, spin, interval = [], [], []
    games = []
    for i in range(2):
        path = ROOT / "artifacts" / "tt24" / f"rolling-match-inspected-{i}.json.gz"
        data = json.loads(gzip.decompress(path.read_bytes()))
        for case in data["cases"]:
            result = case["results"]["calibrated"]
            games.append(result["seed"])
            previous = None
            for event in result["events"]:
                side = event.get("collider")
                if event["kind"] != "separation" or side not in ("learner_racket", "rival_racket"):
                    continue
                speed.append(np.linalg.norm(event["velocity"]))
                spin.append(np.linalg.norm(event["spin"]))
                if previous and side != previous["collider"]:
                    interval.append(event["time"] - previous["time"])
                previous = event
    return dict(speed=speed, spin=spin, interval=interval), games


def run(source):
    rows, reference = records(source)
    train = [r for r in rows if r.player not in HELD]
    held = [r for r in rows if r.player in HELD]
    target = np.stack([r.end for r in train])
    current = [AIR_DRAG, MAGNUS_COEFFICIENT]
    fit = least_squares(
        lambda c: (propagate(train, c) - target).ravel(),
        current,
        bounds=([0.0, 0.0], [0.6, 0.01]),
        loss="soft_l1",
        f_scale=0.02,
        x_scale=current,
    )
    if not fit.success:
        raise ValueError(fit.message)
    models = dict(gravity=[0.0, 0.0], current=current, match_fit=fit.x.tolist())
    scores = {name: score(held, coeff) for name, coeff in models.items()}
    by_player = {
        player: {
            name: score([r for r in held if r.player == player], coeff) for name, coeff in models.items()
        }
        for player in sorted(HELD)
    }
    sim, starts = compact_court()
    report = dict(
        source_commit=SOURCE,
        source_sha256=SHA256,
        counts={p: sum(r.player == p for r in rows) for p in sorted({r.player for r in rows})},
        training_flights=len(train),
        held_flights=len(held),
        held_players=sorted(HELD),
        held_rallies=len({(r.player, r.key.rsplit("/", 1)[0]) for r in held}),
        held_scores=scores,
        by_player=by_player,
        compact_seeds=starts,
        measured_reference={k: quantiles(v) for k, v in reference.items()},
        compact_match={k: quantiles(v) for k, v in sim.items()},
    )
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "recorded-flight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    result = run(parser.parse_args().csv)
    print(result["held_scores"])
    print(result["measured_reference"], result["compact_match"])

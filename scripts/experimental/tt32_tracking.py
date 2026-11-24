"""Time camera-only incoming placement forecasts on intact varied rally action tapes."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_partner_fast import NativeBallFlight
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt31_curriculum_opponent import CENTER_DEPTH, replay_arena
from .tt31_ready import centered


def projection(eye, flight_model, t):
    f = eye.latest
    if (
        f is None
        or t - f.time >= 0.105
        or f.velocity[1] >= -0.90
        or f.rms >= 2.4
        or f.depth_sigma >= 0.16
    ):
        return None
    p, v = f.advance(t)
    path = flight_model.path(p, v, np.zeros(3), 1, bool(f.bounced and p[1] < 0.1))
    if len(path) < 2 or not path[0, 2] >= CENTER_DEPTH >= path[-1, 2]:
        return None
    return float(np.interp(CENTER_DEPTH, path[::-1, 2], path[::-1, 1]))


def play(source, output):
    original = _read(source)
    rows = []
    with centered(original["ready"]):
        arena = replay_arena(original["cards"])
        stereo = CachedStereoRenderer(arena)
        rig = Rig.from_arena(arena)
        nominal = NativeBallFlight(arena)
        try:
            for recorded in original["rows"]:
                arena.launch(recorded["shot"])
                arena.partner.reset(seed=recorded["opponent_seed"])
                rig.begin()
                eye = OpticalFlight(
                    rig, FlightPractice(), learn=True, detector=native_ball_pixels
                )
                forecasts, crossings = [], {}
                for step, action in enumerate(recorded["actions"]):
                    t = step * arena.control_dt
                    if step % 2 == 0:
                        eye.observe(t, stereo.capture(), arena.joints)
                        x = projection(eye, nominal, t)
                        if x is not None:
                            forecasts.append((t, x))
                    arena.exchange(action, np.zeros(5))
                    for i in arena.crossings:
                        crossings.setdefault(i, float(arena.data.time))
                if digest_json(arena.events) != digest_json(recorded["events"]):
                    raise ValueError("Action replay diverged")
                for i, leg in enumerate(recorded["placements"]["delivered"], 1):
                    if i not in crossings:
                        continue
                    contact = leg["far_contact"]["time"]
                    crossing = crossings[i]
                    truth = leg["receive_line"]["xyz"][0]
                    eligible = [(t, x) for t, x in forecasts if contact < t < crossing]
                    windows = {}
                    for lead in [0.4, 0.3, 0.2, 0.1]:
                        available = [
                            (t, x) for t, x in eligible if t <= crossing - lead
                        ]
                        if available:
                            t, x = available[-1]
                            windows[str(lead)] = {
                                "time": t,
                                "prediction": x,
                                "error": x - truth,
                            }
                    rows.append(
                        {
                            "shot": recorded["shot"]["id"],
                            "leg": i,
                            "landed": leg["near_landed"],
                            "far_contact_s": contact,
                            "crossing_by_s": crossing,
                            "true_x": truth,
                            "first_forecast": eligible[0] if eligible else None,
                            "windows": windows,
                        }
                    )
        finally:
            stereo.close()
            arena.close()
    summary = {}
    for lead in ["0.4", "0.3", "0.2", "0.1"]:
        records = [r for r in rows if lead in r["windows"]]
        errors = [abs(r["windows"][lead]["error"]) for r in records]
        summary[lead] = {
            "count": len(records),
            "median_abs_error_m": float(np.median(errors)) if errors else None,
            "center_only_m": float(np.median([abs(r["true_x"]) for r in records]))
            if errors
            else None,
        }
    _write(output, {"source_sha256": _hash(source), "rows": rows, "summary": summary})
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    play(args.source, args.output)

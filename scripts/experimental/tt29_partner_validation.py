"""Qualify the far carriage against held, rotated regulation match launches."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt19_view_full_game import digest_json
from .tt29_partner import ANGULAR_SPEED, DEEP_LANES, LANES, LINEAR_ACCELERATION, LINEAR_SPEED, MASS, PartnerArena
from .tt29_rally_physics import OUTPUT, measured_launches


SEED = 65819417
PROBE_IDS = {2704, 2867, 3737, 6811, 6995, 7025, 8693, 10001, 13820, 14217, 15921}


def schedule(launches, count):
    """Use a typical draw and enrich independent draws for unusual incoming conditions."""
    typical = [(i, s) for i, s in sorted(launches.items()) if i not in PROBE_IDS and s["vel_y"] < -1]
    rng = np.random.default_rng(SEED)
    selected = set(PROBE_IDS)
    for name, feature, quantile in (
        ("ordinary", lambda s: abs(s["vel_y"]), (.20, .80)),
        ("fast", lambda s: abs(s["vel_y"]), (.90, 1.0)),
        ("oblique", lambda s: abs(s["vel_x"] / s["vel_y"]), (.87, 1.0)),
        ("strong_spin", lambda s: np.linalg.norm([s[f"w_vel_{a}"] for a in "xyz"]), (.88, 1.0)),
        ("high", lambda s: s["pos_z"], (.90, 1.0)),
        ("slow", lambda s: abs(s["vel_y"]), (.0, .12)),
    ):
        values = np.array([feature(s) for _, s in typical])
        low, high = np.quantile(values, quantile)
        band = [i for (i, _), v in zip(typical, values) if low <= v <= high and i not in selected]
        for i in rng.choice(band, count, replace=False).tolist():
            selected.add(i)
            yield name, launches[i]


def _rotated_launch(arena, shot):
    """Use the recorded state at t=0, rotated 180 degrees around the tabletop's normal."""
    mirrored = shot.copy()
    for key in ("pos_x", "pos_y", "vel_x", "vel_y", "w_vel_x", "w_vel_y"):
        mirrored[key] *= -1
    arena.launch(mirrored)
    arena._last_hitter = 1
    arena.server = 1
    return mirrored


def _state(arena):
    return np.r_[arena.data.time, arena.ball, arena.velocity, arena.spin]


def play(arena, shot, *, deep=False):
    launch = _rotated_launch(arena, shot)
    action = np.zeros(arena.action_size)
    previous = _state(arena)
    at_depth = None
    for _ in range(arena.max_steps):
        _, _, done, capped, _ = arena.step(action, render=False)
        current = _state(arena)
        if arena.rival_returns:
            target_y = arena.partner.target[1] if arena.partner.reports else -1.06
            if previous[2] >= target_y > current[2]:
                fraction = (previous[2] - target_y) / (previous[2] - current[2])
                at_depth = previous + fraction * (current - previous)
                break
        previous = current
        if done or capped:
            break
    events = arena.events
    contact = next((e for e in events if e["kind"] == "contact" and e.get("side") == -1), None)
    departure = next((e for e in events if e["kind"] == "separation" and e.get("collider") == "rival_racket"), None)
    landing = next((e for e in events if e["kind"] == "legal_return" and e.get("side") == -1), None)
    lane = arena.partner.target if arena.partner.reports else None
    # A practical corridor covers the intended SO-101 intercept height and depth,
    # including variation on each side. This geometric check is independent of joint reach.
    low, high, slow, fast = (1.015, 1.19, 3.0, 4.85) if deep else (1.095, 1.29, 3.35, 5.5)
    useful = (at_depth is not None and abs(at_depth[1]) <= .13 and low <= at_depth[3] <= high
              and slow <= -at_depth[5] <= fast)
    return dict(
        id=shot["id"], launch=launch, result=arena.point_result(), events=events.copy(), event_sha256=digest_json(events),
        target=None if lane is None else lane.tolist(), receiving_state=None if at_depth is None else at_depth.tolist(),
        corridor=bool(useful), plans=arena.partner.reports.copy(), received_legal=bool(arena.legal_returns),
        strike_velocity=None if contact is None else contact["incoming"],
        depart_velocity=None if departure is None else departure["velocity"],
        depart_spin=None if departure is None else departure["spin"],
        landing=None if landing is None else landing["location"],
        flight_s=None if departure is None or landing is None else landing["time"] - departure["time"],
        net_contact=any(e["kind"] == "net" for e in events[events.index(contact):]) if contact else False,
        max_carriage_speed=arena.partner.max_speed, max_carriage_angular_speed=arena.partner.max_angular_speed,
        max_carriage_force=arena.partner.max_force,
    )


def _limits(values):
    return None if not values else dict(zip(("min", "median", "max"), np.quantile(values, [0, .5, 1]).tolist()))


def summarize(rows):
    legal = [r for r in rows if r["landing"] is not None]
    crossing = [r["receiving_state"] for r in legal if r["receiving_state"] is not None]
    return dict(
        launches=len(rows), valid_incoming=sum(r["received_legal"] for r in rows),
        far_contacts=sum(r["result"]["rival_contacts"] for r in rows), far_legal=len(legal),
        below_net_clearance=sum(r["net_contact"] for r in legal), reaches_target_depth=len(crossing),
        in_corridor=sum(r["corridor"] for r in rows), terminal_reasons=dict(Counter(r["result"]["reason"] or "audit_stopped" for r in rows)),
        departure_speed_mps=_limits([np.linalg.norm(r["depart_velocity"]) for r in legal]),
        departure_spin_rad_s=_limits([r["depart_spin"][0] for r in legal]),
        outgoing_strike_to_bounce_s=_limits([r["flight_s"] for r in legal]),
        landing_depth_m=_limits([-r["landing"][1] for r in legal]),
        at_target_depth_x_m=_limits([r[1] for r in crossing]),
        at_target_depth_z_m=_limits([r[3] for r in crossing]),
        at_target_depth_vy_mps=_limits([-r[5] for r in crossing]),
        at_target_depth_vertical_velocity_mps=_limits([r[6] for r in crossing]),
        max_carriage_speed_mps=max(r["max_carriage_speed"] for r in rows),
        max_carriage_angular_speed_rad_s=max(r["max_carriage_angular_speed"] for r in rows),
        max_carriage_force_n=max(r["max_carriage_force"] for r in rows),
    )


def run(path, count, output, *, deep=False):
    a = PartnerArena(max_steps=145, targets=DEEP_LANES if deep else LANES)
    measured = measured_launches(path)
    results = []
    try:
        for group, shot in schedule(measured, count):
            before = perf_counter()
            row = play(a, shot, deep=deep)
            replay = play(a, shot, deep=deep)
            if row != replay:
                raise RuntimeError(f"Changed native replay on measured launch {shot['id']}")
            results.append(dict(group=group, **row))
            print(group, shot["id"], row["received_legal"], bool(row["landing"]), row["corridor"],
                  round(perf_counter() - before, 3), flush=True)
    finally:
        a.close()
    tally = {group: summarize([r for r in results if r["group"] == group]) for group in dict.fromkeys(r["group"] for r in results)}
    corridor = "x ±.13 m, z 1.015–1.19 m, -vy 3.0–4.85 m/s at target y" if deep else (
        "x ±.13 m, z 1.095–1.29 m, -vy 3.35–5.5 m/s at target y")
    metadata = dict(measured_launch_sha256=_hash(path), seed=SEED, fixture=a.fixture,
                    angle_rotation_deg=180, boundary=corridor,
                    carriage_mass_kg=MASS, configured_speed_mps=LINEAR_SPEED,
                    configured_angular_speed_rad_s=ANGULAR_SPEED,
                    configured_acceleration_mps2=LINEAR_ACCELERATION, replayed=len(results))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema="tt29-partner-collision-1", metadata=metadata, all=summarize(results), groups=tally, rows=results))
    print(summarize(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launches", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path)
    settings = parser.parse_args()
    default = "partner-deep-validation.json.gz" if settings.deep else "partner-validation.json.gz"
    run(settings.launches, settings.count, settings.output or OUTPUT / default, deep=settings.deep)

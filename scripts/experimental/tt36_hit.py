"""Learn short-range strike corrections from stereo observation of outgoing flight."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import brentq

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_optical_outcomes import NET_HEIGHT, OutgoingFlight, stereo_position
from .tt29_rally_physics import measured_launches
from .tt29_vision import BOUNCE_Z, VERTICAL, flight_terms
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt36"
BASE = ARTIFACTS / "tt33" / "joint-raw.json.gz"
PRACTICE_CARDS = {
    "recovery-full": (3855, 14183, 13159, 13621),
    "confirm-recovery": (7935, 15343),
    "recovery-periodic": (12330,),
}
OFFSETS = np.array([
    [0., 0., 0., 0., 0.],
    [0., 0., .04, .10, 0.],
    [0., 0., -.04, -.10, 0.],
    [0., .075, -.075, 0., .09],
    [0., -.075, .075, 0., -.09],
])
SIDES = ("left", "right")
ZERO = np.zeros(5)


def smooth(value):
    u = float(np.clip(value, 0., 1.))
    return u * u * (3. - 2. * u)


def activation(optical):
    """Use present proximity to blend the strike change into the base joint policy."""
    return smooth((-optical[1] - .76) / .44)


def lateral(optical):
    return float(optical[0] + .18 * optical[3])


class HitAdjustment:
    """Return a five-angle correction that composes with the unmodified raw policy."""

    def __init__(self, checkpoint):
        self.left, self.right = (np.asarray(checkpoint["bias_rad"][s], dtype=float) for s in SIDES)
        if any(bias.shape != (5,) or not np.isfinite(bias).all() for bias in (self.left, self.right)):
            raise ValueError("Each hit correction must contain five finite joint angles")

    def bias(self, optical, measured_q=None):
        if optical is None:
            return ZERO.copy()
        blend = smooth((lateral(optical) + .055) / .11)
        return activation(optical) * ((1. - blend) * self.left + blend * self.right)


def command_with_hit_bias(policy, optical, measured_q, hit, other_bias=None):
    """Add another five-angle correction before issuing the one bounded command."""
    correction = hit.bias(optical, measured_q)
    if other_bias is not None:
        correction = correction + other_bias
    if not np.any(correction):
        return policy.command(optical, measured_q)
    return policy.command(optical, measured_q, offset=correction)


def roster():
    paths = {name: ARTIFACTS / "tt31" / f"{name}.json.gz" for name in PRACTICE_CARDS}
    cards = [entry for name, path in paths.items() for entry in _read(path)["cards"]
             if entry["id"] in PRACTICE_CARDS[name]]
    return cards, {name: _hash(path) for name, path in paths.items()}


def predicted_outcome(flight):
    time, position, speed = flight.fit
    first = flight.segment[0][0]
    if flight.position(first)[2] < BOUNCE_Z or flight.position(first + 1.)[2] >= BOUNCE_Z:
        return None
    landing_at = brentq(lambda t: flight.position(t)[2] - BOUNCE_Z, first, first + 1.)
    landing = flight.position(landing_at)
    clearance = -.35
    if flight.position(first)[1] < 0. < flight.position(first + 1.)[1]:
        net_at = brentq(lambda t: flight.position(t)[1], first, first + 1.)
        clearance = float(flight.position(net_at)[2] - NET_HEIGHT)
    travel, _ = flight_terms(first - time, flight.drag)
    velocity = np.exp(-flight.drag * (first - time)) * speed + travel * VERTICAL
    return dict(velocity=velocity.tolist(), speed=float(np.linalg.norm(velocity)),
                landing_xy=landing[:2].tolist(), clearance=clearance, time=first,
                last_observed_s=flight.segment[-1][0], frames=len(flight.segment))


def optical_outcomes(observed):
    """Detect near-side y reversals from image points and estimate outgoing velocity."""
    optical = []
    last_contact = -np.inf
    for i in range(2, len(observed) - 3):
        tick, p, _ = observed[i]
        if tick - last_contact < .45 or not -1.68 < p[1] < -.82:
            continue
        prev_t, prev, _ = observed[i - 2]
        new_t, new, _ = observed[i + 2]
        if not .03 < tick - prev_t < .14 or not .03 < new_t - tick < .14:
            continue
        incoming = (p - prev) / (tick - prev_t)
        outgoing = (new - p) / (new_t - tick)
        if incoming[1] > -.8 or outgoing[1] < .75:
            continue
        before = [(t, state) for t, _, state in observed[max(0, i - 7):i + 1]
                  if state is not None and t < tick]
        if not before:
            continue
        tracker = OutgoingFlight(tick, np.linalg.norm(outgoing), censored=True)
        for time, position, _ in observed[i:]:
            tracker.observe(time, position)
            if tracker.done:
                break
        if tracker.fit is None or len(tracker.segment) < 3:
            continue
        result = predicted_outcome(tracker)
        if result is None:
            continue
        last_contact = tick
        incoming_time, state = before[-1]
        result.update(reversal_time=tick, lateral=float(np.median([lateral(state) for _, state in before])),
                      incoming=state.tolist(), incoming_time=incoming_time)
        optical.append(result)
    return optical


def quality(outcome):
    """Use stereo-predicted net margin and a tabletop bounce to score a strike."""
    if outcome is None:
        return -.80
    x, y = outcome["landing_xy"]
    depth = min(y - .04, 1.35 - y) / .55
    width = (.75 - abs(x)) / .40
    net = (outcome["clearance"] - .018) / .18
    return float(np.clip(min(depth, width, net), -1., 1.))


@contextmanager
def station(feeds, model):
    shots = measured_launches(feeds)
    with centered(model["ready"]):
        arena = replay_arena([])
        try:
            renderer, rig, stride, lower, upper = resources(arena)
            try:
                def play(entry, hit):
                    shot = shots[entry["id"]]
                    arena.launch(shot, card=entry)
                    arena.partner.reset(seed=entry["seed"])
                    vision = Vision(rig, arena.control_dt, raw=True)
                    policy = JointPolicy(model, stride, lower, upper)
                    actions, observed = [], []
                    approaching = []
                    starting = process_time()
                    for tick in range(arena.max_steps):
                        q = arena.joints
                        frames = renderer.capture() if tick % 2 == 0 else None
                        x = vision.update(tick, frames, q)
                        if frames is not None:
                            p = stereo_position(vision.eye)
                            if p is not None:
                                observed.append((tick * arena.control_dt, p, x))
                            if x is not None and -.93 < x[1] < -.60:
                                approaching.append((tick * arena.control_dt, lateral(x)))
                        u = command_with_hit_bias(policy, x, q, hit)
                        _, _, done, capped, _ = arena.exchange(u, FAR_COMMAND)
                        actions.append(u.tolist())
                        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= 16:
                            break
                    results = optical_outcomes(observed)
                    near = [e for e in arena.events if e["kind"] == "contact" and e["side"] == 1]
                    receipt = [min((abs(row["reversal_time"] - e["time"]) for e in near), default=100.)
                               for row in results]
                    first = next(iter(approaching), None)
                    measured = next((row for row in results
                                     if first is not None and first[0] < row["reversal_time"] < first[0] + .60), None)
                    return dict(
                        shot=shot, opponent_seed=entry["seed"], actions=actions, events=arena.events,
                        placements=arena.point_placements(), score=audit(arena.events, opening_side=1),
                        optical_outgoing=results, optical_first_turn=measured, first_optical_context=first,
                        optical_timing_audit_s=receipt, motor_audit=arena.motor_audit(),
                        runtime_cpu_s=process_time() - starting,
                    )

                yield play
            finally:
                renderer.close()
        finally:
            arena.close()


def collect(feeds, output):
    model = _read(BASE)
    cards, hashes = roster()
    result = dict(cards=cards, ready=model["ready"], opening_side=1, base_sha256=_hash(BASE),
                  offsets_rad=OFFSETS.tolist(), sources=hashes, rounds=[])
    order = np.random.default_rng(360254)
    with station(feeds, model) as play:
        for entry in cards:
            rows = [None] * len(OFFSETS)
            for i in order.permutation(len(OFFSETS)):
                hit = HitAdjustment(dict(bias_rad={s: OFFSETS[i] for s in SIDES}))
                rows[i] = play(entry, hit)
                row = rows[i]
                print(entry["id"], i, row["score"]["learner"],
                      [(round(y["speed"], 2), round(quality(y), 2)) for y in row["optical_outgoing"]], flush=True)
            result["rounds"].append(rows)
            _write(output, result)


def fit(source, output):
    doc = _read(source)
    if len(doc["rounds"]) != len(doc["cards"]) or doc["base_sha256"] != _hash(BASE):
        raise ValueError("Practice observations are incomplete or the policy changed")
    lane = {side: [] for side in SIDES}
    for rows in doc["rounds"]:
        context = rows[0]["first_optical_context"]
        if context is not None:
            lane["left" if context[1] < 0. else "right"].append(rows)
    chosen, estimates, biases = {}, {}, {}
    for side, cases in lane.items():
        utilities = []
        estimates[side] = []
        for i in range(len(OFFSETS)):
            observations = [rows[i]["optical_first_turn"] for rows in cases]
            velocities = [item["velocity"] for item in observations if item is not None]
            scores = [quality(item) for item in observations]
            utility = float(np.mean(scores)) if scores else -.80
            utilities.append(utility)
            speeds = [np.linalg.norm(v) for v in velocities]
            estimates[side].append(dict(
                chances=len(cases), visible=len(velocities), optical_utility=utility,
                predicted_outgoing_vxyz=np.mean(velocities, axis=0).tolist() if velocities else None,
                predicted_outgoing_speed_mps=float(np.mean(speeds)) if speeds else None,
            ))
        best = int(np.argmax(utilities))
        chosen[side] = best if len(cases) >= 3 and utilities[best] > utilities[0] + .035 else 0
        biases[side] = OFFSETS[chosen[side]].tolist()
    checkpoint = dict(
        schema="tt36-hit-1", base_sha256=_hash(BASE), practice_sha256=_hash(source),
        training_releases=sorted({c["id"] for c in doc["cards"]}),
        chosen=chosen, bias_rad=biases, effect_estimates=estimates,
        cpu_s=sum(r["runtime_cpu_s"] for rows in doc["rounds"] for r in rows),
    )
    _write(output, checkpoint)
    print(chosen, estimates, flush=True)


def trial(feeds, plan_path, checkpoint_path, output):
    model = _read(BASE)
    plan = _read(plan_path)
    checkpoint = _read(checkpoint_path) if checkpoint_path is not None else dict(bias_rad={s: ZERO for s in SIDES})
    if checkpoint_path is not None:
        if checkpoint["base_sha256"] != _hash(BASE):
            raise ValueError("Hit adjustment and raw policy differ")
        if {c["id"] for c in plan["cards"]}.intersection(checkpoint["training_releases"]):
            raise ValueError("Held roster overlaps the training releases")
    hit = HitAdjustment(checkpoint)
    report = dict(cards=plan["cards"], ready=model["ready"], opening_side=1, rows=[],
                  plan_sha256=_hash(plan_path), checkpoint_sha256=_hash(checkpoint_path) if checkpoint_path else None,
                  base_sha256=_hash(BASE))
    with station(feeds, model) as play:
        for entry in report["cards"]:
            row = play(entry, hit)
            report["rows"].append(row)
            _write(output, report)
            print(entry["id"], row["score"]["learner"], row["score"]["returns"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    phases = p.add_subparsers(dest="phase", required=True)
    practice = phases.add_parser("practice")
    practice.add_argument("feeds", type=Path)
    practice.add_argument("output", type=Path)
    train = phases.add_parser("fit")
    train.add_argument("practice", type=Path)
    train.add_argument("output", type=Path)
    games = phases.add_parser("trial")
    games.add_argument("feeds", type=Path)
    games.add_argument("plan", type=Path)
    games.add_argument("output", type=Path)
    games.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.phase == "practice":
        collect(a.feeds, a.output)
    elif a.phase == "fit":
        fit(a.practice, a.output)
    else:
        trial(a.feeds, a.plan, a.checkpoint, a.output)


if __name__ == "__main__":
    main()

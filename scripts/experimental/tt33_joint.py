"""Distill rewarded table-tennis movements into instantaneous visual joint feedback."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import brentq

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_rally_physics import measured_launches
from .tt29_vision import BOUNCE_Z, Flight, FlightPractice, OpticalFlight, Rig
from .tt31_contacts import record as replay_contacts
from .tt31_curriculum import OUTPUT as TT31
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered


OUTPUT = TT31.parent / "tt33"
SOURCES = (
    "ready-practice-staged", "confirm-recovery", "transition-6810-reference",
    "transition-practice",
)
RECEIVE_Y = -1.29
FAR_COMMAND = np.zeros(5)


def flight_at_plane(flight, time):
    """Describe the arriving ball at one fixed observation plane, using RGB alone."""
    if flight is None or time - flight.time >= .105 or flight.velocity[1] >= -.90:
        return None
    if flight.rms >= 2.4 or flight.depth_sigma >= .16:
        return None
    position, velocity = flight.advance(time)
    travel = (RECEIVE_Y - position[1]) / velocity[1]
    damp = flight.damping
    if not -.13 < travel < .74 or 1. - damp * travel <= 0.2:
        return None
    eta = -np.log1p(-damp * travel) / damp
    arrival, arriving = flight.advance(time + eta)
    if eta > .02 and not flight.bounced and position[2] >= BOUNCE_Z > arrival[2]:
        rebound_time = brentq(lambda t: flight.advance(time + t)[0][2] - BOUNCE_Z, 0., eta)
        rebound_pos, rebound_velocity = flight.advance(time + rebound_time)
        if abs(rebound_pos[0]) < .76 and -1.37 < rebound_pos[1] < 0.:
            rebound_velocity *= (.95, .95, -.92)
            bounce = Flight(rebound_pos, rebound_velocity, time + rebound_time, damp, 0., 0., 0, True)
            travel = (RECEIVE_Y - rebound_pos[1]) / rebound_velocity[1]
            eta = rebound_time - np.log1p(-damp * travel) / damp
            arrival, arriving = bounce.advance(time + eta)
    return np.r_[eta, arrival[0], arrival[2], arriving, position[1], flight.rms]


class Vision:
    """Retain an RGB-derived state estimate through brief ball occlusion."""

    def __init__(self, rig, dt, *, raw=False):
        self.eye = None
        if rig is not None:
            rig.begin()
            self.eye = OpticalFlight(rig, FlightPractice(), learn=True, detector=native_ball_pixels)
        self.dt = dt
        self.raw = raw
        self.last = None
        self.last_fit = None
        self.samples = deque(maxlen=5)
        self.context = None

    def update(self, tick, frames, joints):
        now = tick * self.dt
        if frames is not None:
            self.eye.observe(now, frames, joints)
        return self.from_flight(now, self.eye.latest)

    def from_flight(self, now, f):
        if self.raw:
            if (f is not None and now - f.time < .105 and f.velocity[1] < -.90
                    and f.rms < 2.4 and f.depth_sigma < .16):
                return np.r_[*f.advance(now)]
            return None
        if f is not None and f is not self.last_fit:
            self.last_fit = f
            current = flight_at_plane(f, now)
            if current is not None:
                if self.last is None or (self.last[0] < .0 and current[0] > .30):
                    self.samples.clear()
                self.last = current
                self.last_time = now
                if current[0] > .12:
                    self.samples.append(current[1:6])
                    self.context = np.median(list(self.samples), axis=0)
            elif f.velocity[1] > 1.05 and f.rms < 2.4 and f.depth_sigma < .16:
                self.last = self.context = None
                self.samples.clear()
        if self.last is None:
            return None
        feature = self.last.copy()
        feature[0] -= now - self.last_time
        if feature[0] < -.22 or now - self.last_time > .19:
            self.last = self.context = None
            self.samples.clear()
            return None
        return np.r_[feature, self.context if self.context is not None else feature[1:6]]


def resources(arena):
    arena.streams = ("left", "right")
    axes = arena._arm_ids[1][2]
    stride = arena._joint_increments[axes] * (arena.control_dt / arena.model.opt.timestep)
    lower, upper = arena._joint_low[axes], arena._joint_high[axes]
    return CachedStereoRenderer(arena), Rig.from_arena(arena), stride, lower, upper


def observe(arena, stereo, rig, row, *, raw):
    """Replay one training point and retain only independently rewarded strike samples."""
    arena.launch(row["shot"])
    arena.partner.reset(seed=row["opponent_seed"])
    eye = Vision(rig, arena.control_dt, raw=raw)
    axes = arena._arm_ids[1][2]
    previous = arena.joints
    xs, qs, dqs, targets, actions, ticks = [], [], [], [], [], []
    legs = [p for p in row["placements"]["delivered"] if p["near_landed"]]
    handoff_tick = max(row.get("pupil_ticks", []), default=-1)
    for tick, action in enumerate(row["actions"]):
        now = tick * arena.control_dt
        q = arena.joints
        dq = (q - previous) / arena.control_dt
        x = eye.update(tick, stereo.capture() if tick % 2 == 0 else None, q)
        leg = next((i for i, p in enumerate(legs)
                    if p["far_contact"]["time"] <= now <= p["near_contact"]["time"] + .056), None)
        arena.exchange(action, FAR_COMMAND)
        if x is not None and leg is not None and tick > handoff_tick:
            xs.append(x)
            qs.append(q)
            dqs.append(dq)
            targets.append(arena.data.ctrl[axes].copy())
            actions.append(action)
            ticks.append((tick, leg, legs[leg]["near_contact"]["time"] - now))
        previous = q
    if digest_json(arena.events) != digest_json(row["events"]):
        raise ValueError("Demonstration action replay diverged")
    return xs, qs, dqs, targets, actions, ticks


def capture(sources, output, *, raw=False):
    columns = [[] for _ in range(6)]
    receipts = []
    seen = set()
    for source in sources:
        doc = _read(source)
        before = len(columns[0])
        with centered(doc["ready"]):
            arena = replay_arena([])
            stereo, rig, *_ = resources(arena)
            try:
                for entry, row in zip(doc["cards"], doc["rows"], strict=True):
                    if not row["score"]["learner"]:
                        continue
                    key = entry["id"], entry["mode"], entry["seed"]
                    if key in seen:
                        continue
                    seen.add(key)
                    arena.cards = iter([entry])
                    samples = observe(arena, stereo, rig, row, raw=raw)
                    n = len(samples[0])
                    receipts.append(dict(source=source.name, card=entry,
                                         strikes=row["score"]["learner"], samples=n))
                    for destination, values in zip(columns, samples, strict=True):
                        destination.extend(values)
            finally:
                stereo.close()
                arena.close()
        print(source.name, len(columns[0]) - before, flush=True)
    np.savez_compressed(output, **dict(zip(("optical", "joints", "velocities", "targets", "actions", "ticks"),
                                          map(np.asarray, columns), strict=True)))
    source_hashes = {path.name: _hash(path) for path in sources}
    _write(output.with_suffix(".json"), dict(raw=raw, sources=source_hashes, receipts=receipts))



def features(optical, option, q=None, home=None):
    """Dimensionless inputs favor lateral placement and optical time to arrive."""
    if option == "raw":
        return (optical - (0., -.65, 1.10, 0., -3.6, 0.)) / (.10, .25, .12, .7, 1.4, 1.8)
    if option == "raw-proprioception":
        ball = features(optical, "raw")
        return np.concatenate((ball, (q[..., :5] - home[:5]) / (1.25, 1.25, 1.25, 1.25, 1.25)), axis=-1)
    if option == "lateral":
        return optical[..., :2] / (.085, .085)
    if option == "height":
        return (optical[..., :3] - (0., 0., 1.06)) / (.085, .085, .11)
    if option == "context":
        return (optical[..., [0, 8, 9, 10, 11]] - (0., 0., 1.06, 0., -3.3)) / (.085, .085, .13, 1., 1.2)
    if option == "current":
        return (optical[..., :6] - (0., 0., 1.06, 0., -3.3, 0.)) / (.085, .085, .13, 1., 1.2, 1.8)
    raise ValueError(f"Unknown feature set {option}")


def activation(inputs, centers, width):
    distance = np.sum(np.square(inputs[..., None, :] - centers), axis=-1)
    activity = np.exp(-.5 * distance / width**2)
    # A normalized smooth basis can extrapolate a small amount just outside the observations.
    return activity / np.maximum(np.sum(activity, axis=-1, keepdims=True), .02)


class JointPolicy:
    """Emit a single slew-limited joint command from a smooth visual feedback surface."""

    def __init__(self, model, stride, lower, upper):
        self.centers = np.asarray(model["centers"])
        self.weights = np.asarray(model["weights"])
        self.home = np.asarray(model["ready"])
        self.options = model["options"]
        self.stride = stride
        self.lower = lower
        self.upper = upper
        self.reference = self.home.copy()
        self.raw = self.options["features"].startswith("raw")

    def command(self, optical, q, *, offset=None):
        desired = self.home
        bound = self.options["recovery"]
        if optical is not None and (self.raw or -.09 < optical[0] < .69):
            value = features(optical, self.options["features"], q, self.home)
            weights = activation(value, self.centers, self.options["width"])
            desired = self.home + weights @ self.weights
            # Blend smoothly to readiness outside the observed approach phase.
            if not self.raw:
                blend = np.clip((optical[0] + .09) / .045, 0., 1.) * np.clip((.69 - optical[0]) / .13, 0., 1.)
                desired = self.home + blend * (desired - self.home)
            if offset is not None:
                desired[:5] += offset
            bound = self.options["limit"]
        return self.drive(desired, q, bound)

    def drive(self, desired, q, bound):
        """Convert a desired posture to bounded one-tick joint commands."""
        feedback = self.options["feedback"] * (self.reference - q)
        desired = np.clip(desired + feedback, self.lower, self.upper)
        command = np.clip((desired - self.reference) / self.stride, -bound, bound)
        self.reference = np.clip(self.reference + self.stride * command, self.lower, self.upper)
        return command


def fit(training, output, options):
    paths = [training, *options.pop("additional", ())]
    datasets = []
    receipts = []
    for path in paths:
        with np.load(path) as archive:
            datasets.append(dict(archive))
        receipts.append(_read(path.with_suffix(".json")))
    data = {key: np.concatenate([archive[key] for archive in datasets]) for key in datasets[0]}
    sources = {name: digest for receipt in receipts for name, digest in receipt["sources"].items()}
    records = [row for receipt in receipts for row in receipt["receipts"]]
    ready = np.asarray(_read(TT31 / "left-ready.json")["ready"])
    optical = data["optical"]
    target = data["targets"]
    raw = options["features"].startswith("raw")
    if raw != (optical.shape[1] == 6):
        raise ValueError("Training observation format differs from the requested input")
    x = features(optical, options["features"], data["joints"], ready)
    # A fixed coarse mesh spans local optical state. Rounded cell midpoints are
    # aggregated before least squares, without retaining commands or episode identities.
    bins = np.round(x / options["spacing"])
    _, index, count = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    centers = np.zeros((len(count), x.shape[1]))
    np.add.at(centers, index, x)
    centers /= count[:, None]
    design = activation(x, centers, options["width"])
    time_to_hit = data["ticks"][:, 2]
    weights = 1. + options["impact_weight"] * np.exp(-.5 * np.square(time_to_hit / .09))
    matrix = design.T @ (weights[:, None] * design)
    matrix.flat[::len(matrix) + 1] += options["ridge"]
    coefficients = np.linalg.solve(matrix, design.T @ (weights[:, None] * (target - ready)))
    rms = np.sqrt(np.mean(np.square(design @ coefficients - target + ready), axis=0))
    model = dict(schema="tt33-joint-1", sources=sources,
                 training_release_ids=sorted({r["card"]["id"] for r in records}),
                 strikes=sum(r["strikes"] for r in records), samples=len(x),
                 ready=ready.tolist(), centers=centers.tolist(), weights=coefficients.tolist(), options=options,
                 training_rms_rad=rms.tolist())
    _write(output, model)
    print("fit", len(x), len(centers), rms, flush=True)


def point(arena, stereo, rig, model, strain, lo, hi, shot, entry, *, cap, trace=False):
    arena.launch(shot)
    arena.partner.reset(seed=entry["seed"])
    policy = JointPolicy(model, strain, lo, hi) if model is not None else None
    eye = Vision(rig, arena.control_dt, raw=policy.raw if policy else False)
    actions, trajectory, states = [], [], []
    start = process_time()
    while True:
        tick = len(actions)
        encoders = arena.joints
        optical = eye.update(tick, stereo.capture() if tick % 2 == 0 else None, encoders)
        command = policy.command(optical, encoders) if policy is not None else np.zeros_like(encoders)
        _, _, done, truncated, _ = arena.exchange(command, FAR_COMMAND)
        actions.append(command.tolist())
        states.append(encoders)
        if trace:
            trajectory.append(dict(optical=None if optical is None else optical.tolist(), q=encoders.tolist()))
        if done or truncated or arena.legal_returns + arena.rival_returns - 1 >= cap:
            break
    movement = np.asarray(actions)
    analysis = audit(arena.events, opening_side=1)
    change = np.diff(movement, axis=0)
    difference = np.diff(states, axis=0)
    metrics = dict(active_steps=int(np.sum(np.max(abs(movement), axis=1) > .01)),
                   active_by_axis=np.sum(abs(movement) > .01, axis=0).tolist(),
                   max_slew=np.max(abs(movement), axis=0).tolist(),
                   median_slew_change=np.median(abs(change), axis=0).tolist(),
                   p95_slew_change=np.quantile(abs(change), .95, axis=0).tolist(),
                   total_joint_travel_rad=np.sum(abs(difference), axis=0).tolist())
    return dict(shot=shot, opponent_seed=entry["seed"], score=analysis,
                placements=arena.point_placements(), events=arena.events, actions=actions,
                trace=trajectory if trace else None, activity=metrics,
                motor_audit=arena.motor_audit(), runtime_cpu_s=process_time() - start)


def evaluate(feeds, plan, checkpoint, output, *, cap, limit, trace, allow_seen=False):
    doc = _read(plan)
    model = _read(checkpoint) if checkpoint is not None else None
    ready = doc.get("ready") or _read(TT31 / "left-ready.json")["ready"]
    known = set(model["training_release_ids"]) if model is not None and not allow_seen else set()
    roster = [(i, c) for i, c in enumerate(doc["cards"]) if c["id"] not in known]
    if limit:
        roster = roster[:limit]
    shots = measured_launches(feeds)
    out = dict(ready=ready, cards=[c for _, c in roster], rows=[], opening_side=1,
               cap=cap, plan_sha256=_hash(plan), feed_sha256=_hash(feeds),
               checkpoint_sha256=_hash(checkpoint) if checkpoint else None, excluded_seen=sorted(known))
    with centered(ready):
        arena = replay_arena([])
        stereo, rig, stride, lo, hi = resources(arena)
        try:
            for i, entry in roster:
                arena.cards = iter([entry])
                row = point(arena, stereo, rig, model, stride, lo, hi,
                            shots[entry["id"]], entry, cap=cap, trace=trace)
                if doc.get("rows"):
                    row["reference"] = doc["rows"][i]["score"]["learner"]
                out["rows"].append(row)
                _write(output, out)
                print(output.name, entry["id"], row["score"]["learner"], row["score"]["returns"],
                      row.get("reference"), [(item["observed_lane"], item["near_contact"] is not None)
                                             for item in row["placements"]["delivered"]], flush=True)
        finally:
            stereo.close()
            arena.close()


def verify(source, output, *, cad=False):
    """Reproduce every saved referee event without opening cameras or loading a policy."""
    doc = _read(source)
    reports = []
    with centered(doc["ready"]):
        arena = replay_arena([])
        try:
            if cad:
                from .tt29_arm_clearance import ArmClearance
                from .tt29_rally_physics import mujoco

                full_body = replay_arena([], detail="cad")
                guard = ArmClearance(full_body.model, tolerance=0.)
                shadow = mujoco.MjData(full_body.model)
            for entry, row in zip(doc["cards"], doc["rows"], strict=True):
                arena.launch(row["shot"], card=entry)
                arena.partner.reset(seed=row["opponent_seed"])
                if cad:
                    arena.sample_clearance(True)
                    arena._cad_reader = guard, shadow
                contacts = replay_contacts(arena, row, opening_side=1)
                score = audit(arena.events, opening_side=1)
                if (digest_json(score) != digest_json(row["score"])
                        or digest_json(arena.point_placements()) != digest_json(row["placements"])):
                    raise ValueError(f"Replayed score or placements diverged for {entry['key']}")
                clearance = arena.clearance_result() if cad else None
                if cad and clearance["samples"] != round(arena.data.time / arena.model.opt.timestep):
                    raise ValueError("Clearance count differs from solver update count")
                reports.append(dict(id=entry["id"], events_sha256=digest_json(arena.events),
                                    learner=score["learner"], returns=score["returns"],
                                    contacts=contacts, clearance=clearance))
        finally:
            arena.close()
            if cad:
                full_body.close()
    _write(output, dict(source_sha256=_hash(source), points=reports))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    collect = phases.add_parser("capture")
    collect.add_argument("--output", type=Path, default=OUTPUT / "joint-training.npz")
    collect.add_argument("--raw", action="store_true")
    collect.add_argument("--sources", type=Path, nargs="+",
                         default=[TT31 / f"{source}.json.gz" for source in SOURCES])
    train = phases.add_parser("fit")
    train.add_argument("training", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--additional", type=Path, nargs="+", default=[])
    train.add_argument("--features", choices=("lateral", "height", "context", "current", "raw", "raw-proprioception"),
                       default="height")
    for key, default in dict(width=1., spacing=.55, ridge=.05, impact_weight=3.,
                             limit=.8, recovery=.65, feedback=.0).items():
        train.add_argument("--" + key.replace("_", "-"), type=float, default=default)
    assess = phases.add_parser("evaluate")
    assess.add_argument("feeds", type=Path)
    assess.add_argument("plan", type=Path)
    assess.add_argument("output", type=Path)
    assess.add_argument("--checkpoint", type=Path)
    assess.add_argument("--cap", type=int, default=16)
    assess.add_argument("--limit", type=int)
    assess.add_argument("--trace", action="store_true")
    assess.add_argument("--allow-seen", action="store_true")
    replay = phases.add_parser("verify")
    replay.add_argument("source", type=Path)
    replay.add_argument("output", type=Path)
    replay.add_argument("--cad", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "capture":
        capture(args.sources, args.output, raw=args.raw)
    elif args.phase == "fit":
        options = vars(args).copy()
        for key in ("phase", "training", "output"):
            del options[key]
        fit(args.training, args.output, options)
    elif args.phase == "evaluate":
        evaluate(args.feeds, args.plan, args.checkpoint, args.output, cap=args.cap, limit=args.limit,
                 trace=args.trace, allow_seen=args.allow_seen)
    else:
        verify(args.source, args.output, cad=args.cad)


if __name__ == "__main__":
    main()

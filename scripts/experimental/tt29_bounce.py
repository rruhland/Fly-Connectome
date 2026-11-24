"""Warm the existing wrist policy on the edge-clamped robot using actual RGB drops."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import WRIST_CAMERA
from .tt19_view_full_game import digest_json
from .tt24_bounce_ident import _ellipse, _unit_camera_ray
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy, commands
from .tt24_metric_stereo import triangulate
from .tt24_public_body import (
    ConsensusWrist,
    FOCAL,
    PublicDecoder,
    PublicMotion,
    blade_profile,
    disk_pose,
    stationary,
    stock,
)
from .tt26_bounce_optics import impact_candidates, rotation_between, separation, unit
from .tt28_dose import FIRST_FRAMES, NOMINAL, SCALES, SoloLearner, mkplan, solo_summary

OUTPUT = Path("artifacts/tt29")
CHECKPOINT = OUTPUT / "tt29_bounce_checkpoint.json"
COURSE = OUTPUT / "tt29_bounce_course.json.gz"
SEED = 290678113
HELD_SEED = 292972867
DOSES = (0, 4, 12, 24)
PRACTICE_CAP = 160
HELD_CAP = 300
# The published clamp dimensions give this approximate installation shift.
BASE_SHIFT = np.array((0., .366, .2544))


def cold_snapshot():
    policy = CarriedWristPolicy()
    p = policy.calibration[0]
    return dict(
        schema="tt29-wrist-warmup-1",
        drops=0,
        wrist_sha256=_hash(BUNDLE),
        world_offset_m=(p["offset"][:3] + BASE_SHIFT).tolist(),
        blade_offset_m=p["offset"][3:].tolist(),
        blade_normal=p["normal"].tolist(),
        registration=p["registration"].tolist(),
        focal_px=float(FOCAL),
        rotation=np.eye(3).tolist(),
        position_scale_m=float(SCALES[0]),
        image_height=144,
    )


def load_checkpoint(path=CHECKPOINT):
    checkpoint = _read(path)
    if checkpoint["schema"] != "tt29-wrist-warmup-1" or checkpoint["wrist_sha256"] != _hash(BUNDLE):
        raise ValueError("Unrecognized wrist warm-up checkpoint")
    return checkpoint


class Decoder(PublicDecoder):
    """A carried rigid camera projection with a separately estimated base registration."""

    def __init__(self, checkpoint, default):
        offset = np.r_[checkpoint["world_offset_m"], checkpoint["blade_offset_m"]]
        super().__init__(offset, np.array(checkpoint["blade_normal"]),
                         np.array(checkpoint["registration"]), default)
        self.focal = checkpoint["focal_px"]

    def ball(self, ellipse, body):
        return None if ellipse is None else body[:3] + body[3:12].reshape(3, 3) @ (
            _unit_camera_ray(ellipse, self.focal) * self.radius)


def transport_goal(goal, nominal, fitted, bounds, *, scale=.025, include_world=False):
    """Preserve a previous blade pose with a new measured blade and frame."""
    p0 = nominal.body(goal)[12:17]
    if not include_world:
        p0[:3] += fitted.offset[:3] - nominal.offset[:3]

    def residual(q):
        error = (fitted.body(q)[12:17] - p0) / np.array([scale] * 3 + [.045] * 2)
        return np.r_[error, (q - goal) / .70]

    lo, hi = bounds
    start = np.clip(goal, lo, hi)
    fit = least_squares(residual, start, bounds=(np.maximum(lo, start - .70),
                                                np.minimum(hi, start + .70)), max_nfev=40)
    return fit.x


class WarmWrist:
    """The legacy motor readout with an optical warm-up calibration and fresh drop state."""

    def __init__(self, checkpoint, bounds):
        self.checkpoint = checkpoint
        self.bounds = bounds
        self.carried = CarriedWristPolicy()
        p = self.carried.calibration[0]
        self.nominal = Decoder(cold_snapshot(), p["default"])
        self.rig = Decoder(checkpoint, p["default"])
        self.motion = PublicMotion(p["motion_mean"], p["motion_scale"], p["motion_weights"])

    def begin_release(self, obs):
        q = obs["joints"]
        self.reference = q + self.carried.reuse * (self.carried.equilibrium - q)
        self.adjustment = transport_goal(self.reference, self.nominal, self.rig, self.bounds,
                                         scale=self.checkpoint["position_scale_m"]) - self.reference
        self.applied = np.zeros(5)
        self.issued = (q - self.reference).astype(float)
        self.sensor = ConsensusWrist(self.rig, self.motion)
        self.sensor.home = self.rig.body(self.reference + self.adjustment)[12:15]
        self.opening = []
        self.tick = 0

    def act(self, obs, ellipse=None):
        # Preserve the earlier release alignment to the first photographed ball.
        if self.tick < 10 and len(self.opening) < 3:
            body = self.rig.body(obs["joints"])
            point = self.rig.ball(ellipse, body)
            if point is not None and np.linalg.norm(point - self.sensor.home - self.rig.default) < .15:
                if self.opening and np.linalg.norm(point - self.opening[-1]) > .05:
                    self.opening.clear()
                self.opening.append(point)
                if len(self.opening) == 3:
                    query = np.median(self.opening, axis=0)
                    move = query[:2] - self.sensor.home[:2] - self.rig.default[:2]
                    move *= min(1., .0557 / max(1e-9, np.linalg.norm(move)))
                    p0 = self.rig.body(self.reference + self.adjustment)[12:17].copy()
                    p0[:2] += move
                    base = self.reference + self.adjustment
                    lo, hi = self.bounds
                    fit = least_squares(
                        lambda q: np.r_[(self.rig.body(q)[12:17] - p0) / [.03, .03, .03, .15, .15],
                                         .075 * (q - base) / .24],
                        np.clip(base, lo, hi),
                        bounds=(np.maximum(lo, base - .24), np.minimum(hi, base + .24)), max_nfev=35)
                    self.adjustment = fit.x - self.reference
                    self.sensor.home = self.rig.body(fit.x)[12:15]
        self.tick += 1
        state = self.sensor.read(ellipse, obs["joints"])
        base = commands(state, self.carried.weights, self.issued)
        physical = np.clip(base + (self.adjustment - self.applied) / (5.2 / 60), -1, 1)
        self.applied += (physical - base) * (5.2 / 60)
        return physical


def serial(array):
    value = np.asarray(array, dtype=float).astype(object)
    value[~np.isfinite(value.astype(float))] = None
    return value.tolist()


def run_drop(arena, seed, regime, split, checkpoint=None, *, collect=False, tape=None):
    obs = arena.reset(seed=seed, split=split, regime=regime, render=tape is None)
    controller = None
    if tape is None:
        controller = WarmWrist(checkpoint, (arena._control_low[:5], arena._control_high[:5]))
        controller.begin_release(obs)
    actions, enc, wrist, stereo, blade = [], [], [], [], []
    reward = 0.
    for tick in range(arena.max_steps):
        ellipse = _ellipse(obs["rgb"]) if tape is None else None
        if collect and tick < FIRST_FRAMES:
            images = arena.fixed_images()
            enc.append(obs["joints"].tolist())
            wrist.append(np.full(3, np.nan) if ellipse is None else ellipse)
            found = [_ellipse(im) for im in images]
            stereo.append([np.full(2, np.nan) if val is None else val[:2] for val in found])
            if tick % 5 == 0:
                blade.append(dict(tick=tick, profiles=serial([blade_profile(im) for im in images])))
        action = controller.act(obs, ellipse) if tape is None else tape[tick]
        actions.append(np.asarray(action).tolist())
        obs, got, done = arena.step(action, render=tape is None)
        reward += got
        if done:
            break
    metrics = arena.judge_metrics()
    result = dict(actions=actions, reward=reward, steps=metrics["steps"], rebounds=metrics["rebounds"],
                  streak=metrics["longest_consecutive_rebounds"], referee=digest_json(metrics))
    if collect:
        result["optics"] = dict(enc=enc, wrist=serial(wrist), stereo=serial(stereo), blade=blade)
    return result


def blade_measurements(rows, eyes):
    result = []
    for row in rows:
        optics = row["optics"]
        for sample in optics["blade"]:
            if sample["tick"] > 15:
                continue
            measured = disk_pose(eyes, np.array(sample["profiles"], dtype=float), fixed_size=(.076, .006))
            if measured is not None:
                params, residual = measured
                result.append((optics["enc"][sample["tick"]], params[:3], unit(np.r_[params[3:5], 1.]), residual))
    return result


def fit_wrist(rows, eyes, rig, nominal_registration, focal):
    joints = np.concatenate([r["optics"]["enc"] for r in rows])
    wrist = np.concatenate([np.array(r["optics"]["wrist"], dtype=float) for r in rows])
    stereo = np.concatenate([np.array(r["optics"]["stereo"], dtype=float) for r in rows])
    ball, gap, _ = triangulate(eyes, stereo, tolerance=.018)
    valid = np.isfinite(wrist).all(1) & np.isfinite(ball).all(1) & (wrist[:, 2] > 2.5)
    if valid.sum() < 24:
        return nominal_registration, dict(pairs=int(valid.sum()), median_px=None, median_m=None)
    p, r = rig.links.read(joints[valid])
    local = np.einsum("nij,ni->nj", r, ball[valid] - p - rig.offset[:3])
    sight = wrist[valid]
    rays = _unit_camera_ray(sight, focal)
    radius = float(nominal_registration[-1])
    rotated_source = radius * (rays - rays.mean(axis=0))
    shifted_target = local - local.mean(axis=0)
    u, _, vt = np.linalg.svd(rotated_source.T @ shifted_target)
    attitude = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    translation = local.mean(axis=0) - attitude @ (radius * rays.mean(axis=0))

    def camera(params):
        return Rotation.from_rotvec(params[:3]).inv().apply(local - params[3:])

    def residual(params):
        points = camera(params)
        depth = np.maximum(.025, -points[:, 2])
        pixels = focal * points[:, :2] / depth[:, None]
        depth_from_size = radius * -rays[:, 2]
        strength = np.sqrt(1 / (1 + (gap[valid] / .008) ** 2))
        return (np.column_stack((pixels - sight[:, :2], (depth - depth_from_size) * 65))
                * strength[:, None]).ravel()

    start = np.r_[Rotation.from_matrix(attitude).as_rotvec(), translation]
    fit = least_squares(residual, start, loss="soft_l1", f_scale=2, max_nfev=80)
    distance = np.linalg.norm(camera(fit.x) - rays * radius, axis=1)
    error = residual(fit.x).reshape(-1, 3)
    return np.r_[fit.x, radius].tolist(), dict(pairs=int(valid.sum()),
        median_px=float(np.median(np.linalg.norm(error[:, :2], axis=1))), median_m=float(np.median(distance)))


def restore_snapshot(rows, drops, eyes, focal, nominal, solo):
    """Fit only observations collected in the completed prefix of actual drops."""
    checkpoint = nominal | dict(drops=drops)
    rig = Decoder(nominal, np.zeros(3))
    measured = blade_measurements(rows, eyes)
    if measured:
        joints, points, normals, residuals = zip(*measured, strict=True)
        p, rotations = rig.links.read(np.asarray(joints))
        # The new fixture retains the same links and grasp. A conservative prior
        # separates the absolute clamp offset from the rotated tool offset.
        target = np.array(points) - p - rig.offset[:3] - rotations @ rig.offset[3:]
        weights = np.sqrt(1 / (1 + np.square(residuals)))
        blocks = np.concatenate((np.broadcast_to(np.eye(3), rotations.shape), rotations), axis=2)
        design = np.concatenate(((blocks * weights[:, None, None]).reshape(-1, 6), np.eye(6) * .75))
        answer = np.r_[(target * weights[:, None]).ravel(), np.zeros(6)]
        delta = np.linalg.lstsq(design, answer, rcond=None)[0]
        predicted = (blocks @ delta).reshape(-1, 3)
        local = np.einsum("nij,ni->nj", rotations, np.asarray(normals))
        local *= np.where(local @ rig.normal >= 0, 1, -1)[:, None]
        outline_normal = unit(np.median(local, axis=0))
        cone = separation(outline_normal, rig.normal)
        if cone < 20:
            direction = unit(.75 * rig.normal + .25 * outline_normal)
            checkpoint["blade_normal"] = direction.tolist()
        checkpoint.update(world_offset_m=(rig.offset[:3] + delta[:3]).tolist(),
                          blade_offset_m=(rig.offset[3:] + delta[3:]).tolist(),
                          blade_samples=len(measured),
                          blade_median_m=float(np.median(np.linalg.norm(predicted - target, axis=1))),
                          outline_error_deg=cone)
    result = solo.snapshot()
    if result["samples"] >= 3:
        # Each confirmed reversal anchors the normal via the legacy relative-impulse fit.
        # The outline regularizes residual bias from ball spin and friction.
        impulse = np.array(result["rotation"]) @ rig.normal
        direction = unit(.70 * impulse + .30 * np.array(checkpoint["blade_normal"]))
        checkpoint["blade_normal"] = direction.tolist()
    fitted = Decoder(checkpoint, np.zeros(3))
    registration, optics = fit_wrist(rows, eyes, fitted, nominal["registration"], focal)
    checkpoint.update(registration=registration, focal_px=float(focal) if optics["median_px"] is not None else nominal["focal_px"],
                      rotation=rotation_between(rig.normal, fitted.normal).tolist(), optical=optics,
                      position_scale_m=float(SCALES[result["selected"]]), tolerance=result)
    return checkpoint


def held_score(checkpoints, *, held_count=8, seed=HELD_SEED, cap=HELD_CAP):
    """Replay a fixed, disjoint wrist-only held deck with each frozen checkpoint."""
    arena = stock(fixture="table-edge")
    arena.max_steps = cap
    rows = []
    try:
        for seed, regime in mkplan(seed, held_count):
            cases = {}
            for label, snapshot in checkpoints.items():
                result = run_drop(arena, seed, regime, "held", snapshot)
                if run_drop(arena, seed, regime, "held", tape=result["actions"]) != result:
                    raise ValueError(f"Held wrist replay diverged for {seed}/{label}")
                cases[label] = result
            rows.append(dict(seed=seed, regime=regime, results=cases))
    finally:
        arena.close()
    return rows


def finish_course(receipt, path, held_count):
    if held_count:
        receipt["held"] = held_score(receipt["snapshots"], held_count=held_count, seed=receipt["held_seed"])
        receipt["verified_held"] = len(receipt["snapshots"]) * len(receipt["held"])
        receipt["held_summary"] = solo_summary(receipt["held"])
        print("held bounce", receipt["held_summary"], flush=True)
    _write(path, receipt)


def course(output, drops=DOSES[-1], held_count=8):
    arena = stock(fixture="table-edge")
    arena.max_steps = PRACTICE_CAP
    # The lens FOV is a fixed hardware specification. Only the training camera
    # calibrator reads it. The deployed observer consumes the saved focal value.
    fov = float(arena._model.camera(WRIST_CAMERA).fovy[0])
    focal = arena.height / (2 * np.tan(np.deg2rad(fov) / 2))
    nominal = cold_snapshot()
    source = _read(NOMINAL)
    reference = np.asarray(source["mounts"]["nominal"]["normal_link"])
    acceleration = np.array(source["free_flight_acceleration_m_s2"])
    solo = SoloLearner(reference)
    deck = mkplan(SEED, drops)
    rng = np.random.default_rng(SEED ^ 1761209)
    snapshots = {"0": nominal}
    checkpoint = nominal
    rows = []
    begin = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema="tt29-camera-solo-course-1", practice_seed=SEED, held_seed=HELD_SEED,
                   practice_cap_ticks=PRACTICE_CAP, held_cap_ticks=HELD_CAP, practice=rows,
                   snapshots=snapshots, verified_practice=0, verified_held=0,
                   controller_inputs="wrist RGB and own joint encoders at 60 Hz",
                   learning_inputs="paired fixed RGB, wrist RGB, encoders and scalar physical rebound reward",
                   inherited_nominal_optics_sha256=_hash(NOMINAL), inherited_policy_sha256=_hash(BUNDLE))
    path = output / COURSE.name
    deployed = output / CHECKPOINT.name
    try:
        arena.reset(seed=deck[0][0], render=False)
        eyes = stationary(arena)
        for index, (seed, regime) in enumerate(deck, 1):
            action = solo.select(index - 1, rng)
            current = checkpoint | dict(position_scale_m=float(SCALES[action]))
            row = run_drop(arena, seed, regime, "practice", current, collect=True)
            replay = run_drop(arena, seed, regime, "practice", tape=row["actions"])
            if replay != {key: val for key, val in row.items() if key != "optics"}:
                raise ValueError(f"Practice wrist replay diverged for {seed}")
            ray, gap, _ = triangulate(eyes, np.array(row["optics"]["stereo"], dtype=float), tolerance=.018)
            observation = dict(ball=ray, joint_angles=np.array(row["optics"]["enc"]), ray_gap=gap)
            candidates = impact_candidates(observation, acceleration, Decoder(current, np.zeros(3)))
            impact = candidates[0] if candidates else None
            solo.update(action, row["reward"], None if impact is None else impact["raw_normal_link"])
            rows.append(dict(seed=seed, regime=regime, first_impact=impact, tolerance_action=action, **row))
            receipt["verified_practice"] += 1
            # Fit every completed small packet to allow the measured projection to
            # improve the next phase while keeping deployment checkpoints simple.
            if index in {2, *DOSES, drops}:
                checkpoint = restore_snapshot(rows, index, eyes, focal, nominal, solo)
                if index in {*DOSES, drops}:
                    snapshots[str(index)] = checkpoint
                    _write(deployed, checkpoint)
                    receipt["published_checkpoint_sha256"] = _hash(deployed)
                receipt["practice_wall_seconds"] = perf_counter() - begin
                _write(path, receipt)
                print("bounce", index, "recent streaks", [r["streak"] for r in rows[-4:]],
                      "calibration", checkpoint["optical"], "seconds", round(perf_counter() - begin, 1), flush=True)
    finally:
        arena.close()
    finish_course(receipt, path, held_count)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--drops", type=int, choices=range(1, 193), default=DOSES[-1])
    parser.add_argument("--held", type=int, default=8)
    parser.add_argument("--held-only", action="store_true", help="score a verified completed practice prefix")
    args = parser.parse_args()
    if args.held_only:
        path = args.output / COURSE.name
        source = _read(path)
        checkpoint = args.output / CHECKPOINT.name
        if (source["published_checkpoint_sha256"] != _hash(checkpoint)
                or source["verified_practice"] != len(source["practice"])):
            raise ValueError("Practice prefix and published checkpoint disagree")
        finish_course(source, path, args.held)
    else:
        course(args.output, args.drops, args.held)

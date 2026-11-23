"""Register physically rotated stereo views from ordinary RGB before returning a feed."""

import argparse
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import correlate
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt24_rally_policy import EYES, ViewHistory, orange_center
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_grip_course import load_match
from .tt26_receiving_speed import draw


OUT = Path("artifacts/tt27/world-camera.json.gz")
POSES = Path("artifacts/tt26/match-broader.npz")
FEEDS = Path("artifacts/tt26/live-points.json.gz")
RECORDED = Path("artifacts/tt26/fresh-recordings.json.gz")
SOURCE_ROTATIONS = (
    ((-4, 3, 3), (4, 3, -3)),
    ((3, -5, -3), (-3, 4, 4)),
    ((-5, 4, -4), (5, -3, 3)),
)
HELD_ROTATIONS = (
    ((5, 7, 5), (-3, -7, -3)),
    ((-7, -7, 4), (7, 5, 4)),
    ((-6, 7, -4), (8, -6, -3)),
    ((7, -6, -4), (-6, 6, 5)),
)
SEED = 28271191
STRESS_SEED = 60843753
STRESS_CHANGES = 12
PER_SETTING = 32
RADIUS = 6
SEARCH = 32
BUDGETS = (5, 8)
PREFIT_BUDGET = 30


class CameraMounts:
    """Evaluator-only camera motor. Its orientation never enters the receiver."""

    def __init__(self, arena):
        self.cameras = {eye: arena.model.camera(CAMERAS[eye][0]) for eye in EYES}
        self.original = {eye: cam.quat.copy() for eye, cam in self.cameras.items()}

    def move(self, angles):
        for eye, turn in zip(EYES, angles, strict=True):
            change = np.roll(Rotation.from_euler("xyz", turn, degrees=True).as_quat(), 1)
            mujoco.mju_mulQuat(self.cameras[eye].quat, self.original[eye], change)


def gray(rgb):
    return np.asarray(rgb, dtype=np.float64) @ (0.299, 0.587, 0.114)


def templates(rgb):
    """Give each part of the scene an opportunity to supply the clearest texture."""
    image = gray(rgb)
    height, width = image.shape
    templates = []
    for y in range(46, height - 21, 19):
        for x in range(29, width - 25, 24):
            choices = []
            for dy in (-4, 0, 4):
                for dx in (-4, 0, 4):
                    u, v = x + dx, y + dy
                    patch = image[v - RADIUS : v + RADIUS + 1, u - RADIUS : u + RADIUS + 1]
                    gy, gx = np.gradient(patch)
                    quality = float(np.linalg.eigvalsh(np.cov([gx.ravel(), gy.ravel()]))[0])
                    choices.append((quality, (u, v), patch - patch.mean()))
            quality, location, patch = max(choices, key=lambda row: row[0])
            if quality > 2:
                templates.append((quality, location, patch))
    return templates


def match(tile, image):
    _, (x, y), template = tile
    top = max(0, y - SEARCH - RADIUS)
    left = max(0, x - SEARCH - RADIUS)
    clip = image[top : y + SEARCH + RADIUS + 1, left : x + SEARCH + RADIUS + 1]
    box = np.ones_like(template)
    windows = template.size
    sums = correlate(clip, box, mode="valid", method="fft")
    squared = correlate(clip**2, box, mode="valid", method="fft")
    energy = np.maximum(1e-7, squared - sums**2 / windows)
    scores = correlate(clip, template, mode="valid", method="fft") / np.sqrt(energy * np.sum(template**2))
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    best = float(scores[i, j])
    # Use the curvature of the peak to resolve motion between pixel centers.
    offset = np.zeros(2)
    for axis, k in enumerate((j, i)):
        line = scores[i] if axis == 0 else scores[:, j]
        if 0 < k < len(line) - 1:
            curvature = line[k - 1] - 2 * best + line[k + 1]
            if curvature < -1e-6:
                offset[axis] = np.clip((line[k - 1] - line[k + 1]) / (2 * curvature), -.5, .5)
    return np.array((left + j + RADIUS, top + i + RADIUS)) + offset, best


class Registration:
    """Match template tiles and find the camera rotation with the most agreement."""

    def __init__(self, rgb, fovy):
        self.height, self.width = rgb.shape[:2]
        self.focal = self.height / (2 * np.tan(np.deg2rad(fovy) / 2))
        self.center = np.array(((self.width - 1) / 2, (self.height - 1) / 2))
        self.tiles = templates(rgb)
        self.all = np.arange(len(self.tiles))
        self.nominal = np.array([tile[1] for tile in self.tiles])
        self.quality = np.sqrt([tile[0] for tile in self.tiles])
        self.good = np.ones(len(self.all))
        self.seen = np.ones(len(self.all))

    def select(self, budget, learned):
        """Spread the matches across the image using confidence from prior changes."""
        gain = self.quality * (self.good / self.seen if learned else 1)
        diversity = np.ones(len(self.tiles))
        selected = []
        for _ in range(min(budget, len(self.tiles))):
            best = int(np.argmax(gain * diversity))
            selected.append(best)
            distance = np.linalg.norm(self.nominal - self.nominal[best], axis=1)
            diversity *= np.minimum(1, distance / 63)
        return np.asarray(selected)

    def rays(self, points):
        return np.column_stack(((points - self.center) / self.focal, np.ones(len(points))))

    def pixels(self, rays):
        return rays[:, :2] * (self.focal / rays[:, 2:]) + self.center

    @staticmethod
    def rotation(source_rays, new_rays, weights):
        source_rays = source_rays / np.linalg.norm(source_rays, axis=1, keepdims=True)
        new_rays = new_rays / np.linalg.norm(new_rays, axis=1, keepdims=True)
        left, _, right = np.linalg.svd((source_rays * weights[:, None]).T @ new_rays)
        left[:, -1] *= np.linalg.det(left @ right)
        return left @ right

    def locate(self, rgb, choices):
        image = gray(rgb)
        candidates = [match(self.tiles[i], image) for i in choices]
        return np.asarray([x for x, _ in candidates]), np.asarray([quality for _, quality in candidates])

    def estimate(self, choices, points, scores):
        source = self.rays(self.nominal[choices])
        observed = self.rays(points)
        # Distinct landmarks supply two independent rays. Test all pairs at the
        # scored budget, then re-fit from the agreeing points.
        i, j = np.triu_indices(len(choices), 1)
        eligible = (np.linalg.norm(self.nominal[choices[i]] - self.nominal[choices[j]], axis=1) >= 35)
        eligible &= (scores[i] > .6) & (scores[j] > .6)
        weight = np.maximum(0, (scores - .5))**2
        best = -1.0
        agreement = None
        for a, b in zip(i[eligible], j[eligible], strict=True):
            pair = [a, b]
            rotation = self.rotation(source[pair], observed[pair], weight[pair])
            if Rotation.from_matrix(rotation).magnitude() > np.deg2rad(18):
                continue
            error = np.linalg.norm(self.pixels(source @ rotation) - points, axis=1)
            support = error < 2.4
            vote = np.dot(weight, np.maximum(0, 1 - error**2 / 2.4**2))
            if support.sum() >= 3 and vote > best:
                best, agreement = vote, support
        if agreement is None:
            return np.eye(3), 0
        rotation = self.rotation(source[agreement], observed[agreement], weight[agreement])
        return rotation, int(agreement.sum())

    def practice(self, rgb):
        reference = self.select(PREFIT_BUDGET, False)
        points, scores = self.locate(rgb, self.all)
        rotation, support = self.estimate(reference, points[reference], scores[reference])
        if not support:
            raise ValueError("Could not register a calibration-only practice view")
        residual = np.linalg.norm(self.pixels(self.rays(self.nominal) @ rotation) - points, axis=1)
        self.good += (residual < 2.4) & (scores > .6)
        self.seen += 1
        return rotation, support

    def calibrate(self, rgb, learned, budget):
        choices = self.select(budget, learned)
        points, scores = self.locate(rgb, choices)
        rotation, support = self.estimate(choices, points, scores)
        return rotation, int(support), choices.tolist()

    def warp_point(self, point, rotation):
        if point is None:
            return None
        raster = (point + .5) / (480 / self.width) - .5
        centered = self.pixels(self.rays(raster[None]) @ rotation.T)[0]
        return (centered + .5) * (480 / self.width) - .5


def watch(arena, shot, mounts, angles):
    """Collect the same four-frame unacted stereo prefix for all paired policies."""
    mounts.move(angles)
    arena.launch(shot)
    views = []
    for tick in range(OBSERVE + 1):
        views.append((arena.render_cameras(), arena.joints.copy()))
        if tick < OBSERVE:
            _, _, done, cap, _ = arena.step(np.zeros(arena.action_size), render=False)
            if done or cap:
                break
    return views


def visual_key(views, guides, calibration):
    vision = ViewHistory()
    previous = dict.fromkeys(EYES)
    for frames, joints in views:
        pixels = []
        for eye in EYES:
            raw = orange_center(frames[eye], previous[eye])
            if raw is not None:
                previous[eye] = raw
            pixels.append(guides[eye].warp_point(raw, calibration[eye]))
        key = vision.observe_pixels(pixels, joints, joints)
    return key[:14]


def hit(arena, shot, goal, tape=None):
    arena.launch(shot)
    body = PublicBody(arena)
    actions = []
    for tick in range(arena.max_steps):
        if tape is None:
            action = np.zeros(arena.action_size)
            if tick >= OBSERVE:
                action[:5] = np.clip((goal - body.target[:5]) / STEP_SIZE[:5], -1, 1)
        else:
            action = tape[tick]
        _, _, done, cap, _ = arena.step(action, render=False)
        body.issued(action)
        actions.append(np.asarray(action).tolist())
        if done or cap or arena.legal_returns:
            break
    return dict(actions=actions, legal=arena.legal_returns, contact=arena.contacts,
                incoming=arena.rival_returns, events=digest_json(arena.events))


def summary(rows, names):
    nominal = [row["results"]["nominal"] for row in rows]
    result = {}
    for name in names:
        current = [row["results"][name] for row in rows]
        delta = np.asarray([a["legal"] - b["results"]["factory"]["legal"]
                            for a, b in zip(current, rows, strict=True)])
        result[name] = dict(legal=sum(a["legal"] for a in current),
                            contacts=sum(a["contact"] for a in current),
                            different_nominal_action=sum(a["action"] != b["action"]
                                                         for a, b in zip(current, nominal, strict=True)),
                            paired_gains_vs_factory=int((delta > 0).sum()),
                            paired_losses_vs_factory=int((delta < 0).sum()))
    return result


def paired_returns(rows, active, baseline):
    delta = np.asarray([r["results"][active]["legal"] - r["results"][baseline]["legal"] for r in rows])
    gains, losses = int((delta > 0).sum()), int((delta < 0).sum())
    return dict(net_legal=int(delta.sum()), gains=gains, losses=losses,
                ties=int((delta == 0).sum()),
                exact_two_sided=float(binomtest(gains, gains + losses).pvalue) if gains + losses else 1.)


def true_direction(angles):
    sign = np.diag((1, -1, -1))
    return sign @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix() @ sign


def schedule(count, changes, feeds, seed):
    rng = np.random.default_rng(seed)
    if feeds is not None:
        shots, _ = load_match(feeds)
        ids = [r["id"] for r in _read(RECORDED)["records"][0]["rows"]]
        roster = rng.permutation(ids)
        if count * changes > len(roster):
            raise ValueError("The held recorded roster is smaller than the requested course")
        return [[shots[int(k)] for k in roster[i * count : (i + 1) * count]] for i in range(changes)]
    donors = {r["shot"]["id"]: r["shot"] for r in _read(FEEDS)["cases"] if r["angle"] < 0}
    return [[draw(donors, sorted(donors), rng, i * count + j) for j in range(count)]
            for i in range(changes)]


def stress_rotations(seed):
    rng = np.random.default_rng(seed)
    angles = rng.integers((-9, 6, -7), (10, 11, 8), size=(STRESS_CHANGES, 2, 3))
    angles[:, :, 1] *= rng.choice((-1, 1), size=(STRESS_CHANGES, 2))
    return angles.tolist()


def run(path=OUT, count=PER_SETTING, changes=len(HELD_ROTATIONS), feeds=None, each_feed=False, stress=False):
    with np.load(POSES, allow_pickle=False) as data:
        keys, goals, scale = (data[k] for k in ("keys", "goals", "scale"))
    turns = stress_rotations(STRESS_SEED) if stress else HELD_ROTATIONS[:changes]
    seed = SEED + 1 if stress else SEED
    course = schedule(count, len(turns), feeds, seed)
    arena = ReferenceReceiver()
    mounts = CameraMounts(arena)
    nominal = ((0, 0, 0),) * 2
    identity = {eye: np.eye(3) for eye in EYES}
    ablations = {f"{label}_{budget}": (carried, budget)
                 for budget in BUDGETS for label, carried in (("reset", False), ("carried", True))}
    names = ("factory", "frozen", *ablations, "nominal")
    report = dict(schema="tt27-world-camera-1", pose_source_sha256=_hash(POSES),
                  feed_source_sha256=_hash(FEEDS if feeds is None else feeds),
                  roster_source_sha256=None if feeds is None else _hash(RECORDED),
                  roster="fresh perturbations of stored recorded feeds" if feeds is None else "raw held recordings",
                  seed=seed, held_trials_per_change=count,
                  source_calibration=[], camera_changes=[], summary={},
                  calibration_budget=dict(stereo_frames=1, patch_matches_per_eye=BUDGETS),
                  controller_inputs=["ordinary fixed stereo RGB", "joint encoders", "issued joint targets"])
    if each_feed:
        report["calibrate_every_feed"] = True
    if stress:
        report["rotation_schedule_seed"] = STRESS_SEED
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        arena.reset(seed=109723, server=-1, render=False)
        original = arena.render_cameras()
        guides = {eye: Registration(original[eye], arena.lens_intrinsics()[eye]["fovy"]) for eye in EYES}
        frozen = {}
        for turn in SOURCE_ROTATIONS:
            mounts.move(turn)
            views = arena.render_cameras()
            fitted, support = {}, {}
            for eye in EYES:
                fitted[eye], support[eye] = guides[eye].practice(views[eye])
            frozen = fitted
            report["source_calibration"].append(dict(rotation_degrees=turn, support=support,
                rgb_sha256={eye: sha256(views[eye]).hexdigest() for eye in EYES}))
        for phase, (turn, shots) in enumerate(zip(turns, course, strict=True)):
            rows = []
            record = dict(rotation_degrees=turn, rows=rows, calibration={})
            report["camera_changes"].append(record)
            fits = {"factory": identity, "frozen": frozen, "nominal": identity}
            for index, shot in enumerate(shots):
                views = watch(arena, shot, mounts, turn)
                calibration = {}
                if index == 0 or each_feed:
                    for label, (prior, budget) in ablations.items():
                        fits[label] = {}
                        calibration[label] = {}
                        for eye, angles in zip(EYES, turn, strict=True):
                            correction, support, choices = guides[eye].calibrate(views[0][0][eye], prior, budget)
                            fits[label][eye] = correction
                            error = Rotation.from_matrix(correction @ true_direction(angles).T).magnitude()
                            calibration[label][eye] = dict(matches=support, selected_tiles=choices,
                                                            rotation_error_degrees=float(np.rad2deg(error)))
                    if index == 0:
                        record["calibration"] = calibration
                baseline = watch(arena, shot, mounts, nominal)
                results, physical = {}, {}
                for name in names:
                    cue = visual_key(baseline if name == "nominal" else views, guides, fits[name])
                    choice = int(np.argmin(np.linalg.norm((keys - cue) / scale, axis=1)))
                    if choice not in physical:
                        trial = hit(arena, shot, goals[choice])
                        replay = hit(arena, shot, goals[choice], trial["actions"])
                        if replay != trial:
                            raise ValueError(f"Physical replay changed on change {phase}, shot {index}")
                        physical[choice] = trial
                    results[name] = dict(action=choice, **physical[choice])
                row = dict(index=index, shot=shot, results=results, verified_actions=len(physical),
                           first_rgb_sha256={eye: sha256(views[0][0][eye]).hexdigest() for eye in EYES})
                if each_feed:
                    row["calibration"] = calibration
                rows.append(row)
            record["summary"] = summary(rows, names)
            print("camera change", phase, record["summary"], record["calibration"], flush=True)
            all_rows = [row for case in report["camera_changes"] for row in case["rows"]]
            report["summary"] = summary(all_rows, names)
            report["paired_legal"] = {label: {
                baseline: paired_returns(all_rows, label, baseline)
                for baseline in ("factory", "frozen", label.replace("carried", "reset"))
            } for label in ablations if label.startswith("carried")}
            _write(path, report)
    finally:
        arena.close()
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--per-change", type=int, default=PER_SETTING)
    cli.add_argument("--changes", type=int, choices=range(1, len(HELD_ROTATIONS) + 1), default=len(HELD_ROTATIONS))
    cli.add_argument("--feeds", type=Path, help="Use the raw held DeepMind shot roster")
    cli.add_argument("--each-feed", action="store_true", help="Recalibrate on each first frame without a change signal")
    cli.add_argument("--stress", action="store_true", help="Use the twelve randomized larger physical camera turns")
    args = cli.parse_args()
    if args.per_change <= 0:
        cli.error("--per-change must be positive")
    run(args.output, args.per_change, args.changes, args.feeds, args.each_feed, args.stress)

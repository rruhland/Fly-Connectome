"""Learn a direct joint correction from optical gaps during zero-score play."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt34_visual as visual
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_source_rally import blade_sighting
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_rally_physics import measured_launches
from .tt29_vision import Rig
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries, held_difference
from .tt31_ready import centered
from .tt33_local import ROOT, Local


OUTPUT = ROOT / "tt35"
ARMS = (0, 1, 2, 3, 4)
MAX_CORRECTION = .16


def triangulate(pixels, rig):
    cameras = [rig.fixed[name] for name in ("left", "right")]
    directions = []
    for xy, camera in zip(pixels, cameras, strict=True):
        ray = np.r_[(xy - camera.center) / camera.focal, 1.] @ camera.rotation
        directions.append(ray / np.linalg.norm(ray))
    a, b = directions
    delta = cameras[1].eye - cameras[0].eye
    cross = float(a @ b)
    if abs(cross) > .985:
        return None
    ta = (a @ delta - cross * (b @ delta)) / (1 - cross**2)
    tb = cross * ta - b @ delta
    p, q = cameras[0].eye + ta * a, cameras[1].eye + tb * b
    if min(ta, tb) <= 0 or np.linalg.norm(p - q) > .035:
        return None
    return .5 * (p + q)


class Observations:
    """Record simultaneous raw stereo ball and blade centers, separate from control."""

    def __init__(self, stereo, rig):
        self.stereo, self.rig = stereo, rig
        self.begin()

    def begin(self):
        self.tick = 0
        self.rows = []

    def capture(self):
        images = self.stereo.capture()
        tick = self.tick
        self.tick += 2
        ball = [native_ball_pixels(images[k]) for k in ("left", "right")]
        if not all(b is not None and not b.clipped for b in ball):
            return images
        point = triangulate([b.center for b in ball], self.rig)
        if point is None or not (-1.62 < point[1] < -.70 and .63 < point[2] < 1.40 and abs(point[0]) < .66):
            return images
        blade = [blade_sighting(images[k]) for k in ("left", "right")]
        if any(b is None for b in blade):
            return images
        center = triangulate([v[0] * images[k].shape[0] / 360 for k, v in zip(("left", "right"), blade, strict=True)], self.rig)
        if center is None or not (-1.67 < center[1] < -.82 and .75 < center[2] < 1.42 and abs(center[0]) < .62):
            return images
        self.rows.append((tick, point, center))
        return images


def passage(rows, onset):
    """Read a passage after play, interpolating only between two actual frames."""
    samples = [(i, ball, blade, ball - blade) for i, ball, blade in rows if onset <= i <= onset + 70]
    for first, second in zip(samples, samples[1:]):
        i, p, q, e = first
        j, r, s, f = second
        if j - i > 4 or r[1] >= p[1] or e[1] <= 0 or f[1] > 0:
            continue
        mix = float(e[1] / (e[1] - f[1]))
        at = i + mix * (j - i)
        difference = e + mix * (f - e)
        return dict(tick=at, gap_xz_m=difference[[0, 2]].tolist(), ball_m=(p + mix * (r - p)).tolist(),
                    blade_m=(q + mix * (s - q)).tolist(), frames=[i, j])
    return None


def record_practice(output):
    paths = [visual.OUTPUT / f"visual-practice-{i}.json.gz" for i in ARMS]
    documents = [_read(p) for p in paths]
    cards = documents[0]["cards"]
    ready = documents[0]["ready"]
    if any(d["cards"] != cards or len(d["rows"]) != len(cards) or d["ready"] != ready for d in documents):
        raise ValueError("Practice tapes differ")
    records = []
    with centered(ready):
        arena = replay_arena([])
        stereo = CachedStereoRenderer(arena)
        camera = Observations(stereo, Rig.from_arena(arena))
        try:
            for entry, trials in zip(cards, zip(*(d["rows"] for d in documents), strict=True), strict=True):
                if any(r["score"]["learner"] or not r["choices"] for r in trials):
                    continue
                if any(len(r["placements"]["delivered"]) != 1 for r in trials):
                    continue
                contexts = [r["choices"][0]["features"] for r in trials]
                if any(c != contexts[0] for c in contexts[1:]):
                    raise ValueError("The observed first decision changed across practice tapes")
                measurements = []
                for row in trials:
                    arena.launch(row["shot"], card=entry)
                    arena.partner.reset(seed=entry["seed"])
                    camera.begin()
                    for i, commands in enumerate(row["actions"]):
                        if i % 2 == 0:
                            camera.capture()
                        arena.exchange(commands, visual.FAR)
                    if digest_json(arena.events) != digest_json(row["events"]):
                        raise ValueError("A practice miss did not replay")
                    measurements.append(passage(camera.rows, row["choices"][0]["tick"]))
                print(entry["id"], entry["seed"], [m and m["gap_xz_m"] for m in measurements], flush=True)
                records.append(dict(card=entry, features=contexts[0], optical_passages=measurements))
        finally:
            stereo.close()
            arena.close()
    _write(output / "error-practice.json.gz", dict(zero_legal_training=True, ready=ready, arms=ARMS,
           source_sha256=[_hash(p) for p in paths], records=records))


def fit(output):
    path = output / "error-practice.json.gz"
    source = _read(path)
    labels = []
    for row in source["records"]:
        observations = row["optical_passages"]
        base = observations[0]
        if base is None:
            continue
        jacobian = np.zeros((2, 2))
        support = []
        for axis, (plus, minus) in enumerate(((1, 2), (3, 4))):
            hi, lo = observations[plus], observations[minus]
            if hi is None or lo is None or max(abs(m["tick"] - base["tick"]) for m in (hi, lo)) > 9:
                continue
            strength = visual.OFFSETS[plus, axis] - visual.OFFSETS[minus, axis]
            jacobian[:, axis] = (np.array(hi["gap_xz_m"]) - lo["gap_xz_m"]) / strength
            support.append(axis)
        if not support:
            continue
        error = np.asarray(base["gap_xz_m"])
        desired = -.75 * jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .016 * np.eye(2), error)
        delta = np.clip(desired, -MAX_CORRECTION, MAX_CORRECTION)
        labels.append(dict(card=row["card"], observation=row["features"], delta_rad=delta.tolist(),
                           gap_xz_m=error.tolist(), camera_effect_m_per_rad=jacobian.tolist(), axes=support))
    if len(labels) < 3:
        raise ValueError(f"Only {len(labels)} optical passage pairs support a correction")
    _write(output / "error-checkpoint.json", dict(only_zero_score=True, camera_receipt_sha256=_hash(path),
           raw_policy_sha256=_hash(visual.MODEL), labels=labels, sensory_scale=visual.SCALE.tolist(),
           max_correction_rad=MAX_CORRECTION))
    print("fitted", len(labels), "representatives", [v["delta_rad"] for v in labels], flush=True)


class Continuous:
    def __init__(self, choose):
        self.choose = choose
        self.current = None
        self.choices = []

    def delta(self, tick, observed):
        if observed[4] > .9:
            self.current = None
        if observed[4] >= -.85:
            return None
        distance = np.linalg.norm((observed[:3] - [0., -1.30, 1.00]) / [.49, .67, .46])
        weight = np.clip((1 - distance) / .38, 0., 1.)
        if weight == 0:
            return None
        if self.current is None:
            self.current = np.r_[self.choose(observed[:11]), np.zeros(3)]
            self.choices.append(dict(tick=tick, delta_rad=self.current.tolist()))
        return weight * self.current if np.any(self.current) else None


def run(feeds, output, mode):
    previous = _read(visual.OUTPUT / "visual-plan.json.gz")
    fixed_source = visual.OUTPUT / "visual-held-fixed.json.gz"
    fixed = _read(fixed_source)
    state_path = output / "error-checkpoint.json"
    state = _read(state_path)
    if _hash(visual.MODEL) != state["raw_policy_sha256"] or _hash(feeds) != previous["feed_sha256"]:
        raise ValueError("Pupil or recorded feed changed")
    cards = fixed["cards"]
    if {c["id"] for c in cards} & {item["card"]["id"] for item in state["labels"]}:
        raise ValueError("A held measured release was present in training")
    contexts = np.asarray([v["observation"] for v in state["labels"]])
    offsets = np.asarray([v["delta_rad"] for v in state["labels"]])
    scale = np.asarray(state["sensory_scale"])
    sign = {"fixed": 0., "learned": 1., "reversed": -1.}[mode]

    def choose(observed):
        weights = np.exp(-.5 * np.sum(np.square((contexts - observed) / scale), axis=1))
        return sign * (weights @ offsets) / (.30 + weights.sum())

    shots = measured_launches(feeds)
    with np.load(visual.MODEL) as data:
        model = Local(dict(data))
    report = dict(ready=fixed["ready"], opening_side=1, cards=cards, rows=[], mode=mode, cap=16,
                  checkpoint_sha256=_hash(state_path), source_sha256=_hash(fixed_source))
    with centered(report["ready"]), patch.object(visual, "Correction", Continuous):
        arena = replay_arena([])
        stereo = CachedStereoRenderer(arena)
        rig = Rig.from_arena(arena)
        camera = Observations(stereo, rig)
        try:
            for i, entry in enumerate(cards):
                camera.begin()
                row = visual.play(arena, camera, rig, model, shots[entry["id"]], entry, choose, 16)
                if mode == "fixed" and any(digest_json(row[k]) != digest_json(fixed["rows"][i][k])
                                           for k in ("actions", "events", "placements")):
                    raise ValueError("Measuring the scene changed the uncorrected game")
                row["optical_passage"] = passage(camera.rows, row["choices"][0]["tick"]) if row["choices"] else None
                report["rows"].append(row)
                _write(output / f"error-{mode}.json.gz", report)
                print(mode, i+1, entry["id"], row["score"]["learner"], row["optical_passage"] and row["optical_passage"]["gap_xz_m"], flush=True)
        finally:
            stereo.close()
            arena.close()


def summary(output):
    sections = _read(visual.OUTPUT / "visual-plan.json.gz")["held"]
    paths = {k: output / f"error-{k}.json.gz" for k in ("fixed", "learned", "reversed")}
    docs = {k: _read(p) for k, p in paths.items()}
    expected = sum(sections.values(), [])
    for doc in docs.values():
        if doc["cards"] != expected or len(doc["rows"]) != len(expected):
            raise ValueError("Held deck incomplete or changed")
    report = {}
    i = 0
    for name, cards in sections.items():
        n = len(cards)
        rows = {k: d["rows"][i:i+n] for k,d in docs.items()}
        i += n
        common = [j for j in range(n) if all(r[j]["optical_passage"] is not None for r in rows.values())]
        result = {}
        for k, tape in rows.items():
            passed = [np.linalg.norm(r["optical_passage"]["gap_xz_m"]) for r in tape if r["optical_passage"]]
            gaps = [np.linalg.norm(tape[j]["optical_passage"]["gap_xz_m"]) for j in common]
            scores = [r["score"]["learner"] for r in tape]
            stats = dict(arm=sum(scores), per_point=scores, both=sum(r["score"]["returns"] for r in tape),
                         cpu_s=sum(r["process_cpu_s"] for r in tape),
                         applied=sum(any(c["delta_rad"]) for r in tape for c in r["choices"]))
            result[k] = dict(stats=stats, paired=held_difference(tape, rows["fixed"]),
                first_offers=compare_delivery(first_deliveries(tape), first_deliveries(rows["fixed"])),
                observed_passes=len(passed), mean_common_gap_m=float(np.mean(gaps)) if gaps else None,
                common_gap_m=[float(g) for g in gaps])
        report[name] = dict(common_cards=[cards[j] for j in common], results=result)
    _write(output / "error-summary.json", dict(checkpoint_sha256=_hash(output / "error-checkpoint.json"),
           held_sha256={k:_hash(p) for k,p in paths.items()}, cohorts=report))
    print({k: {m: (d["stats"]["arm"], d["mean_common_gap_m"]) for m,d in g["results"].items()} for k,g in report.items()}, flush=True)


def native_passages(path):
    """Audit a completed game tape with native poses, outside learning and play."""
    import mujoco

    archive = _read(path)
    receipts = []
    with centered(archive["ready"]):
        arena = replay_arena([])
        shadow = mujoco.MjData(arena.model)
        blade = arena.model.geom("learner_racket").id
        try:
            for entry, row in zip(archive["cards"], archive["rows"], strict=True):
                observed = row["optical_passage"]
                if observed is None:
                    continue
                frames = observed["frames"]
                arena.launch(row["shot"], card=entry)
                arena.partner.reset(seed=entry["seed"])
                samples = {}
                for tick, action in enumerate(row["actions"]):
                    if frames[0] - 9 <= tick <= frames[1] + 9:
                        shadow.qpos[:] = arena.data.qpos
                        mujoco.mj_kinematics(arena.model, shadow)
                        samples[tick] = arena.ball.copy(), shadow.geom_xpos[blade].copy()
                    arena.exchange(action, visual.FAR)
                if any(tick not in samples for tick in frames) or digest_json(arena.events) != digest_json(row["events"]):
                    raise ValueError("Native pose audit changed the game or sample support")
                mix = (observed["tick"] - frames[0]) / (frames[1] - frames[0])
                before, after = (samples[tick][0] - samples[tick][1] for tick in frames)
                physical = before + mix * (after - before)
                error = np.asarray(observed["gap_xz_m"]) - physical[[0, 2]]
                native_crossing = passage([(i, *pair) for i, pair in samples.items()], frames[0] - 9)
                receipts.append(dict(id=entry["id"], native_gap_xyz_m=physical.tolist(),
                                     observed_gap_xz_m=observed["gap_xz_m"], error_xz_m=error.tolist(),
                                     native_crossing=native_crossing))
        finally:
            arena.close()
    errors = np.asarray([r["error_xz_m"] for r in receipts])
    return dict(source_sha256=_hash(path), rows=receipts,
           median_absolute_error_xz_m=np.median(abs(errors), axis=0).tolist(),
           median_abs_native_depth_m=float(np.median([abs(r["native_gap_xyz_m"][1]) for r in receipts])))


def audit_optics(output):
    checked = {mode: native_passages(output / f"error-{mode}.json.gz") for mode in ("fixed", "learned", "reversed")}
    gaps = {mode: {r["id"]: float(np.linalg.norm(r["native_crossing"]["gap_xz_m"]))
                   for r in d["rows"] if r["native_crossing"] is not None} for mode, d in checked.items()}
    common = sorted(set.intersection(*(set(items) for items in gaps.values())))
    comparison = {mode: [items[i] for i in common] for mode, items in gaps.items()}
    _write(output / "error-optical-audit.json", dict(modes=checked, common_ids=common, native_gap_m=comparison,
           native_mean_m={k: float(np.mean(v)) if v else None for k,v in comparison.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("phase", choices=("practice", "fit", "fixed", "learned", "reversed", "summary", "audit"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "practice":
        record_practice(a.output)
    elif a.phase == "fit":
        fit(a.output)
    elif a.phase == "summary":
        summary(a.output)
    elif a.phase == "audit":
        audit_optics(a.output)
    else:
        run(a.feeds, a.output, a.phase)


if __name__ == "__main__":
    main()

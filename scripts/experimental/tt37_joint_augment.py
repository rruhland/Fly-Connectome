"""Prepare a training-only sensor correction for native mentor candidate frames."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_joint_capture import OUTPUT
from .tt37_joint_deep import SCALES, run
from .tt37_joint_deep_report import totals
from .tt37_joint_report import paired, read_complete, replay_case
from .tt37_opponent import arena


SCALE = np.array([.20, .18, .17, 1., .90, 1., 2., 2., 2., 2., 2.])
K = 24
STRENGTH = .85
VARIANTS = ("average", "coherent")


def matched_native(folder, receipt, optical, marks):
    """Replay only training tapes. Their simulator state never enters live play."""
    sources = {}
    for name, digest in receipt["sources"].items():
        path = folder / name
        if _hash(path) != digest:
            raise ValueError(f"Mentor archive has changed: {name}")
        sources[name] = _read(path)
    snapshots = []
    releases = []
    cursor = 0
    with centered(receipt["ready"]):
        environment = arena(protocol="random")
        try:
            for sample in receipt["practice"]:
                card = sample["card"]
                index = sample["source_row"]
                original = sources[sample["source"]]
                if original["cards"][index] != card:
                    raise ValueError("Mentor row no longer matches the capture receipt")
                row = original["rows"][index]
                stop = cursor + sample["frames"]
                tick_ids = marks[cursor:stop, 0].astype(np.int64)
                cursor = stop
                if len(np.unique(tick_ids)) != len(tick_ids):
                    raise ValueError("Capture contains duplicate camera ticks within a point")
                take = set(tick_ids)
                environment.launch(row["shot"], card=card)
                environment.partner.reset(seed=card["seed"])
                for tick, command in enumerate(row["actions"]):
                    if tick in take:
                        snapshots.append(np.r_[environment.ball.copy(), environment.velocity.copy(),
                                               environment.joints[:5].copy()])
                        releases.append(card["id"])
                    environment.exchange(command, FAR_COMMAND)
                if digest_json(environment.events) != digest_json(row["events"]):
                    raise ValueError("Native alignment diverged from the mentor's physical event stream")
        finally:
            environment.close()
    native = np.array(snapshots)
    if native.shape != optical.shape or not np.allclose(native[:, 6:], optical[:, 6:], atol=1e-8):
        raise ValueError("Training tick alignment failed its measured encoder check")
    return native, np.asarray(releases)


def weighted_median(residual, weight):
    order = np.argsort(residual, axis=1)
    sorted_residual = np.take_along_axis(residual, order, axis=1)
    weights = np.broadcast_to(weight[:, :, None], residual.shape)
    weights = np.take_along_axis(weights, order, axis=1)
    first_half = np.argmax(np.cumsum(weights, axis=1) >= .5, axis=1)
    return np.take_along_axis(sorted_residual, first_half[:, None], axis=1)[:, 0]


class NativeCorrection:
    """Robust local native-to-camera shift, with no effect outside calibration support."""

    def __init__(self, native, optical, use):
        self.native = native
        self.residual = optical[:, :6] - native[:, :6]
        self.use = np.asarray(use)
        self.trees = {}
        for sign in (-1, 1):
            indices = np.flatnonzero(sign * native[:, 4] > .9)
            if len(indices) >= K:
                self.trees[sign] = cKDTree(native[indices] / SCALE), indices

    def predict(self, native):
        native = np.asanyarray(native)
        shift = np.zeros((len(native), 6))
        for sign, (tree, indices) in self.trees.items():
            active = np.flatnonzero(sign * native[:, 4] > .9)
            if not len(active):
                continue
            distance, neighbors = tree.query(native[active] / SCALE, K)
            weight = np.exp(-.5 * (distance / np.maximum(.23, distance[:, [K // 2]]))**2)
            weight /= np.sum(weight, axis=1, keepdims=True)
            residual = weighted_median(self.residual[indices[neighbors]], weight)
            support = np.clip((1.65 - distance[:, 2]) / .65, 0., 1.)
            shift[active] = residual * support[:, None] * self.use
        predicted = native.copy()
        predicted[:, :6] += shift
        return predicted


class MixedField:
    """Offer simulated steering where RGB training gives weak local support."""

    def __init__(self, rgb, candidate):
        self.rgb = rgb
        self.candidate = candidate
        self.begin_point()

    def begin_point(self):
        self.diagnostics = dict(candidate_ticks=0, candidate_only_ticks=0, candidate_weight_sum=0.)

    def get(self, optical, q, *, reference=None):
        real, a = self.rgb.get(optical, q)
        candidate, b = self.candidate.get(optical, q, reference=reference)
        native_weight = STRENGTH * b * (1. - .65 * a)
        rgb_weight = a * (1. - native_weight)
        weight = rgb_weight + native_weight
        if native_weight:
            self.diagnostics["candidate_ticks"] += 1
            self.diagnostics["candidate_only_ticks"] += int(real is None)
            self.diagnostics["candidate_weight_sum"] += native_weight
        if weight == 0.:
            return None, 0.
        if candidate is None:
            return real, float(weight)
        if real is None:
            return candidate, float(weight)
        return (real * rgb_weight + candidate * native_weight) / weight, float(weight)


class CandidateField:
    """Predict one joint vector from native, camera-aligned incoming labels."""

    def __init__(self, bank, mode, variant):
        self.scale = SCALES[mode]
        self.variant = variant
        use = bank["states"][:, 4] < -.9
        self.x = bank["states"][use] / self.scale
        self.label = bank["targets_now" if mode == "local" else "targets_lead"][use]
        _, self.groups = np.unique(bank["group"][use], axis=0, return_inverse=True)
        if len(self.x) < 64:
            raise ValueError("Not enough successful native incoming samples")
        self.tree = cKDTree(self.x)

    def get(self, optical, q, *, reference=None):
        if optical[4] > -.9:
            return None, 0.
        center = np.r_[optical, q[:5]] / self.scale
        dist, idx = self.tree.query(center, 64)
        support = float(np.clip((1.8 - dist[2]) / .8, 0., 1.))
        if not support:
            return None, 0.
        if self.variant == "coherent":
            reference = q if reference is None else reference
            continuity = np.linalg.norm(self.label[idx, :5] - reference[:5], axis=1)
            anchor = np.argmin(dist + .58 * continuity)
            group = self.groups[idx[anchor]]
            neighborhood = np.flatnonzero(self.groups[idx] == group)[:8]
            # The group can change every tick and there is no retained index or clock.
            support *= float(np.clip((1.8 - dist[anchor]) / .8, 0., 1.))
        else:
            neighborhood = np.arange(12)
        dist, idx = dist[neighborhood], idx[neighborhood]
        weight = np.exp(-.5 * (dist / max(.14, np.median(dist)))**2)
        weight /= weight.sum()
        label = self.label[idx]
        x = self.x[idx] - center
        mean, ref = weight @ x, weight @ label
        d = x - mean
        fit = np.linalg.solve(d.T @ (weight[:, None] * d) + .045 * np.eye(len(center)),
                              d.T @ (weight[:, None] * (label - ref)))
        return ref - np.clip(mean @ fit, -.23, .23), support


def fit(folder, output):
    source = folder / "joint-deep-observations.npz"
    receipt = _read(source.with_suffix(".json"))
    if receipt["bank_sha256"] != _hash(source):
        raise ValueError("The paired RGB training bank no longer matches its capture receipt")
    with np.load(source) as doc:
        optical, marks = doc["states"], doc["marks"]
    native, releases = matched_native(folder, receipt, optical, marks)
    out_of_fold = np.empty_like(native)
    for group in np.unique(releases):
        held = releases == group
        correction = NativeCorrection(native[~held], optical[~held], np.ones(6, dtype=bool))
        out_of_fold[held] = correction.predict(native[held])
    raw_mae = np.mean(np.abs(native[:, :6] - optical[:, :6]), axis=0)
    fit_mae = np.mean(np.abs(out_of_fold[:, :6] - optical[:, :6]), axis=0)
    use = fit_mae < .95 * raw_mae
    np.savez_compressed(output, native=native, optical=optical, releases=releases, use=use)
    report = dict(rows=len(native), releases=len(np.unique(releases)),
                  source_sha256=_hash(source), bank_sha256=_hash(output),
                  features=["x", "y", "z", "vx", "vy", "vz"], raw_mae=raw_mae.tolist(),
                  corrected_mae=fit_mae.tolist(), use=use.tolist(),
                  validation="leave one training release out; require >5% MAE reduction per component")
    _write(output.with_suffix(".json"), report)
    print(report, flush=True)


def prepare(candidates, sensor, output, opening_plan):
    """Apply the frozen training-only correction to positive native candidates."""
    with np.load(sensor) as s:
        correction = NativeCorrection(s["native"], s["optical"], s["use"])
    held = {card["id"] for card in _read(opening_plan)["held_openings"]}
    fits, receipts = [], []
    total = 0
    for candidate in candidates:
        with np.load(candidate) as c:
            x, target, lead, group = (c[k] for k in ("states", "targets_now", "targets_lead", "group"))
            if (x.ndim != 2 or x.shape[1] != 11 or target.shape != (len(x), 6) or lead.shape != (len(x), 6)
                    or group.shape != (len(x), 6) or len(x) < 1):
                raise ValueError("Native candidate must use the deep p/v/q and actuator-label schema")
            if not all(np.isfinite(a).all() for a in (x, target, lead)):
                raise ValueError("Candidate archive has nonfinite states or commands")
            ids = set(group[:, 0])
            matches = candidate.with_name(candidate.stem.replace("native-", "clinic-labels-", 1) + ".json.gz")
            mentor = _read(matches) if matches.is_file() else None
            cards = mentor["cards"] if mentor else []
            if mentor is not None:
                if len(cards) != len(mentor["rows"]):
                    raise ValueError("Candidate labels lack a complete physical receipt for every declared card")
                keys = {(card["id"], card["seed"], card["reply_seed"]) for card in cards}
                if any(tuple(key) not in keys for key in np.unique(group[:, :3], axis=0)):
                    raise ValueError("Native frame provenance is missing from physical practice receipts")
            if (ids | {card["id"] for card in cards}) & held:
                raise ValueError("A candidate training release intersects the independent opening deck")
            learned = np.concatenate([correction.predict(x[i:i + 4096]) for i in range(0, len(x), 4096)])
            mask = (learned[:, 4] < -.9) & (x[:, 4] < -.9)
            fits.append(tuple(a[mask] for a in (learned, target, lead, group)))
            total += len(mask)
            receipts.append(dict(name=candidate.name, sha256=_hash(candidate), positive=len(mask),
                                 failed=len(c["weak_states"]), retained=int(mask.sum()),
                                 release_ids=[int(i) for i in sorted(ids)], mentor_points=len(cards),
                                 mentor_sha256=_hash(matches) if mentor else None))
    x, target, lead, groups = (np.concatenate(a) for a in zip(*fits, strict=True))
    np.savez_compressed(output, states=x, targets_now=target, targets_lead=lead, group=groups)
    _write(output.with_suffix(".json"), dict(
        candidates=receipts, sensor_sha256=_hash(sensor), bank_sha256=_hash(output),
        native_rows=total, retained_rows=len(x), strength=STRENGTH,
        rule="at most 85% candidate confidence if RGB unsupported, at most 29.75% if RGB fully supported",
    ))


def assess(feeds, candidate, deck, variant, output, *, folder, cap):
    choice = _read(folder / "joint-deep-choice.json")
    source = folder / "joint-deep-observations.npz"
    if choice["bank_sha256"] != _hash(source):
        raise ValueError("RGB pupil calibration changed")
    report = _read(candidate.with_suffix(".json"))
    if report["bank_sha256"] != _hash(candidate) or report["strength"] != STRENGTH:
        raise ValueError("Prepared simulation candidate changed since calibration")
    with np.load(candidate) as bank:
        simulation = CandidateField(bank, choice["mode"], variant)
    report["variant"] = variant
    run(feeds, source, folder / "opponent-plan.json", deck, choice["mode"], output,
        cap=cap, field_hook=lambda rgb: MixedField(rgb, simulation), details=report,
        limit=12 if deck == "familiar" else None)


def select(folder, output):
    base = _read(folder / "joint-deep-choice.json")["mode"]
    baseline = read_complete(folder / f"joint-deep-familiar-{base}.json.gz")
    baseline = dict(baseline, cards=baseline["cards"][:12], rows=baseline["rows"][:12])
    stats, hashes = {"rgb": totals(baseline)}, {}
    shared_bank = None
    for variant in VARIANTS:
        path = folder / f"joint-native-familiar-{variant}.json.gz"
        doc = read_complete(path)
        details = doc["candidate_source"]
        if doc["cards"] != baseline["cards"] or details["variant"] != variant:
            raise ValueError("All candidate modes must use the same familiar roster")
        if shared_bank is None:
            shared_bank = details["bank_sha256"]
        if shared_bank != details["bank_sha256"]:
            raise ValueError("Alternative modes must share the same augmentation bank")
        stats[variant], hashes[variant] = totals(doc), _hash(path)
    chosen = max(VARIANTS, key=lambda v: (stats[v]["own"] + stats[v]["later"], stats[v]["own"],
                                         -VARIANTS.index(v)))
    _write(output, dict(variant=chosen, stats=stats, bank_sha256=shared_bank, sources=hashes,
                        rule="Choose a candidate on first 12 training contexts by own plus later, then own"))


def compare(folder, output):
    choice = _read(folder / "joint-native-choice.json")
    base = _read(folder / "joint-deep-choice.json")["mode"]
    held = {}
    for deck in ("held_openings", "fresh_replies"):
        before = read_complete(folder / f"joint-deep-{deck}-{base}.json.gz")
        path = folder / f"joint-native-{deck}-{choice['variant']}.json.gz"
        after = read_complete(path)
        if after["candidate_source"]["bank_sha256"] != choice["bank_sha256"]:
            raise ValueError("Native candidate changed after the familiar tradeoff")
        diagnostics = {key: sum(r["controller_diagnostics"][key] for r in after["rows"])
                       for key in ("candidate_ticks", "candidate_only_ticks", "candidate_weight_sum")}
        held[deck] = dict(rgb=totals(before), augmented=totals(after), paired=paired(before, after),
                          intervention=diagnostics, source_sha256=_hash(path))
    _write(output, dict(calibration=choice, held=held))


def audit(folder, output):
    picked = _read(folder / "joint-native-choice.json")["variant"]
    base = _read(folder / "joint-deep-choice.json")["mode"]
    evidence = []
    for deck in ("held_openings", "fresh_replies"):
        before = read_complete(folder / f"joint-deep-{deck}-{base}.json.gz")
        path = folder / f"joint-native-{deck}-{picked}.json.gz"
        after = read_complete(path)
        if before["cards"] != after["cards"]:
            raise ValueError("Audit points changed the paired roster")
        own = [r["score"]["learner"] for r in after["rows"]]
        old = [r["score"]["learner"] for r in before["rows"]]
        i = max(range(len(own)), key=lambda i: (own[i] - old[i], own[i]))
        evidence.append(dict(deck=deck, original_own=old[i], source_sha256=_hash(path),
                             **replay_case(after["ready"], "random", after["cards"][i], after["rows"][i])))
    _write(output, dict(rows=evidence))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    correction = commands.add_parser("calibrate")
    correction.add_argument("--folder", type=Path, default=OUTPUT)
    correction.add_argument("--output", type=Path, default=OUTPUT / "joint-native-sensor.npz")
    candidate = commands.add_parser("prepare")
    candidate.add_argument("output", type=Path)
    candidate.add_argument("candidates", type=Path, nargs="+")
    candidate.add_argument("--sensor", type=Path, default=OUTPUT / "joint-native-sensor.npz")
    candidate.add_argument("--opening-plan", type=Path, default=OUTPUT / "opponent-plan.json")
    evaluate = commands.add_parser("assess")
    evaluate.add_argument("feeds", type=Path)
    evaluate.add_argument("candidate", type=Path)
    evaluate.add_argument("deck", choices=("familiar", "held_openings", "fresh_replies", "held_replies"))
    evaluate.add_argument("variant", choices=VARIANTS)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--folder", type=Path, default=OUTPUT)
    evaluate.add_argument("--cap", type=int, default=200)
    calibrate_choice = commands.add_parser("select")
    calibrate_choice.add_argument("--folder", type=Path, default=OUTPUT)
    calibrate_choice.add_argument("--output", type=Path, default=OUTPUT / "joint-native-choice.json")
    analysis = commands.add_parser("compare")
    analysis.add_argument("--folder", type=Path, default=OUTPUT)
    analysis.add_argument("--output", type=Path, default=OUTPUT / "joint-native-results.json")
    replay = commands.add_parser("audit")
    replay.add_argument("--folder", type=Path, default=OUTPUT)
    replay.add_argument("--output", type=Path, default=OUTPUT / "joint-native-audit.json")
    args = p.parse_args()
    if args.command == "calibrate":
        fit(args.folder, args.output)
    elif args.command == "prepare":
        prepare(args.candidates, args.sensor, args.output, args.opening_plan)
    elif args.command == "assess":
        assess(args.feeds, args.candidate, args.deck, args.variant, args.output, folder=args.folder, cap=args.cap)
    elif args.command == "select":
        select(args.folder, args.output)
    elif args.command == "compare":
        compare(args.folder, args.output)
    else:
        audit(args.folder, args.output)

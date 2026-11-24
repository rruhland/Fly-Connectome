"""Correct current ball–blade alignment with a learned visual body Jacobian."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, features, resources, verify
from .tt34_latency import pixels
from .tt34_translate import LearnedBody, triangulate
from .tt35_replies import varied_opponent


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
OUTPUT = ARTIFACTS / "tt36"
PUPIL = ARTIFACTS / "tt33" / "joint-raw.json.gz"
BODY_BANK = ARTIFACTS / "tt34" / "lag-bank.npz"
BALL_SCALE = np.array([.33, .13, .14, 1.05, 1.15, 1.75])
Q_SCALE = np.full(5, 2.6)
WEIGHTED_AXES = np.array([1., .40, 1.])
AUTHORITY = (.38, .65)


def smooth(value):
    u = float(np.clip(value, 0., 1.))
    return u * u * (3. - 2. * u)


class Body(LearnedBody):
    """Differentiate a weighted local map trained on reward-free blade observations."""

    def local(self, q):
        distance, indexes = self.tree.query(q / self.scale, 48)
        weights = np.exp(-.5 * (distance / max(.12, distance[16])) ** 2)
        weights /= weights.sum()
        mean_q = weights @ self.q[indexes]
        mean_p = weights @ self.p[indexes]
        dx = self.q[indexes] - mean_q
        dy = self.p[indexes] - mean_p
        change = np.linalg.solve(dx.T @ (weights[:, None] * dx) + .0007 * np.eye(5),
                                 dx.T @ (weights[:, None] * dy)).T
        return mean_p + change @ (q - mean_q), change, distance[0]


def prepare(recording, output):
    """Store independently observed ball–blade displacements on rewarded teacher frames."""
    with np.load(BODY_BANK) as data:
        body = Body(data)
    with np.load(recording) as data:
        optical, q = data["optical"], data["joints"][:, :5]
        time_to_hit = data["ticks"][:, 2]
    use = (time_to_hit > .005) & (time_to_hit < .55) & (optical[:, 1] < -.24)
    optical, q = optical[use], q[use]
    pose = [body.local(row) for row in q]
    reliable = np.array([dist < .82 for _, _, dist in pose])
    displacement = optical[:, :3] - np.array([point for point, _, _ in pose])
    np.savez_compressed(output, optical=optical[reliable], encoders=q[reliable],
                        displacement=displacement[reliable])
    receipt = _read(recording.with_suffix(".json"))
    _write(output.with_suffix(".json"), dict(
        body_sha256=_hash(BODY_BANK), source_sha256=_hash(recording), sources=receipt["sources"],
        training_releases=sorted({r["card"]["id"] for r in receipt["receipts"]}),
        samples=int(np.count_nonzero(reliable)), pupil_sha256=_hash(PUPIL),
    ))
    print("recorded ball–blade relations", int(reliable.sum()), flush=True)


class Relation:
    """Interpolate the current relative position across individual observation samples."""

    def __init__(self, observations):
        optical = observations["optical"] / BALL_SCALE
        measured = observations["encoders"] / Q_SCALE
        self.displacement = observations["displacement"]
        self.neighbors = cKDTree(np.concatenate((optical, measured), axis=1))

    def target(self, optical, joints):
        key = np.r_[optical / BALL_SCALE, joints / Q_SCALE]
        dist, near = self.neighbors.query(key, 20)
        weight = np.exp(-.5 * (dist / max(.19, dist[9])) ** 2)
        weight /= weight.sum()
        return weight @ self.displacement[near], dist[0]


class Reach:
    """Issue bounded pose adjustments using only the present ball and arm state."""

    def __init__(self, body, relation, pupil, rig, authority, sign):
        self.body = body
        self.relation = relation
        self.pupil = pupil
        self.rig = rig
        self.authority = authority
        self.sign = sign
        self.support = cKDTree(pupil.centers)
        self.correction = np.zeros(3)
        self.last_glance = -10
        self.active = []

    def bias(self, tick, optical, joints, frames):
        """Apply the body Jacobian only when the direct pupil has weak local support."""
        if optical is None or not -1.53 < optical[1] < -.34:
            return None
        raw = features(optical, self.pupil.options["features"], joints, self.pupil.home)
        support, _ = self.support.query(raw)
        onset = smooth((-.34 - optical[1]) / .32)
        gate = smooth((support - .58) / .75) * onset
        if gate < .03:
            return None

        blade, jacobian, body_distance = self.body.local(joints[:5])
        if frames is not None:
            observed = pixels(frames)
            if observed is not None:
                seen, gap = triangulate(observed[None], self.rig)
                residual = seen[0] - blade
                if gap[0] < .035 and np.linalg.norm(residual) < .095:
                    self.correction = .50 * self.correction + .50 * residual
                    self.last_glance = tick
        visual = tick - self.last_glance <= 4
        if visual:
            blade += self.correction
        desired_relative, relation_support = self.relation.target(optical, joints[:5])
        weight = gate * (1. - smooth((body_distance - .64) / .85))
        weight *= 1. - smooth((relation_support - 1.15) / .95)
        if weight < .025:
            return None

        error = optical[:3] - blade - desired_relative
        error = np.clip(error, -.20, .20) * WEIGHTED_AXES
        jacobian = WEIGHTED_AXES[:, None] * jacobian
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .0025 * np.eye(3), error)
        correction = self.sign * weight * np.clip(2.0 * delta, -self.authority, self.authority)
        self.active.append(dict(tick=tick, support=support, body_distance=body_distance,
                                gate=weight, relative_error_m=error.tolist(), correction_rad=correction.tolist(),
                                live_blade=visual))
        return correction


def play(arena, renderer, rig, model, stride, lo, hi, shot, card, body, relation, authority, sign):
    arena.launch(shot, card=card)
    arena.partner.reset(seed=card["seed"])
    pupil = JointPolicy(model, stride, lo, hi)
    eye = Vision(rig, arena.control_dt, raw=True)
    reach = Reach(body, relation, pupil, rig, authority, sign)
    commands, encoders = [], []
    started = process_time()
    for tick in range(arena.max_steps):
        q = arena.joints
        frames = renderer.capture() if tick % 2 == 0 else None
        optical = eye.update(tick, frames, q)
        correction = reach.bias(tick, optical, q, frames)
        action = pupil.command(optical, q, offset=correction)
        _, _, ended, capped, _ = arena.exchange(action, FAR_COMMAND)
        commands.append(action.tolist())
        encoders.append(q.tolist())
        if ended or capped or arena.legal_returns + arena.rival_returns - 1 >= 24:
            break
    return dict(
        shot=shot, opponent_seed=card["seed"], actions=commands, events=arena.events,
        placements=arena.point_placements(), score=audit(arena.events, opening_side=1),
        encoder_trajectory=encoders, active_corrections=reach.active, motor_audit=arena.motor_audit(),
        runtime_cpu_s=process_time() - started,
    )


def run(feeds, observations, output_dir, cohort, modes):
    available = ("0", "1") if cohort == "practice" else ("aligned", "reversed")
    if any(mode not in available for mode in modes):
        raise ValueError("Practice and held modes have different configuration fields")
    output_dir.mkdir(exist_ok=True, parents=True)
    model = _read(PUPIL)
    source = OUTPUT / f"course-{cohort}.json.gz"
    plan = _read(source)
    receipt = _read(observations.with_suffix(".json"))
    if receipt["body_sha256"] != _hash(BODY_BANK) or receipt["pupil_sha256"] != _hash(PUPIL):
        raise ValueError("Body or raw policy changed since building the observation sample")
    if any(card["id"] in receipt["training_releases"] for card in plan["cards"]):
        raise ValueError("Acting course overlaps the rewarded observation examples")
    shots = measured_launches(feeds)
    with np.load(BODY_BANK) as data:
        body = Body(data)
    with np.load(observations) as data:
        relation = Relation(data)
    checkpoint = None
    if cohort != "practice":
        checkpoint = _read(output_dir / "reach-checkpoint.json")
        if checkpoint["observations_sha256"] != _hash(observations):
            raise ValueError("The body feedback checkpoint uses different observation samples")

    ready = model["ready"]
    with centered(ready):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for mode in modes:
                authority = AUTHORITY[int(mode)] if cohort == "practice" else checkpoint["authority_rad"]
                sign = -1 if mode == "reversed" else 1
                report = dict(
                    ready=ready, cards=plan["cards"], opening_side=1, rows=[], sign=sign,
                    mode=mode, authority_rad=authority, base_sha256=_hash(PUPIL),
                    observations_sha256=_hash(observations), course_sha256=_hash(source),
                )
                seeds = plan.get("reply_seeds", [None] * len(plan["cards"]))
                for card, seed in zip(plan["cards"], seeds, strict=True):
                    context = varied_opponent(card["index"], seed) if seed is not None else nullcontext()
                    with context:
                        row = play(arena, renderer, rig, model, stride, lo, hi,
                                   shots[card["id"]], card, body, relation, authority, sign)
                    report["rows"].append(row)
                    _write(output_dir / f"reach-{cohort}-{mode}.json.gz", report)
                    print(cohort, mode, card["id"], row["score"]["learner"], row["score"]["returns"],
                          len(row["active_corrections"]), flush=True)
        finally:
            renderer.close()
            arena.close()


def scores(rows):
    first = [row["placements"]["delivered"][0] for row in rows if row["placements"]["delivered"]]
    legs = [leg for row in rows for leg in row["placements"]["delivered"]]
    result = dict(
        own=sum(row["score"]["learner"] for row in rows),
        later=sum(max(0, row["score"]["learner"] - 1) for row in rows),
        alternating=sum(row["score"]["returns"] for row in rows),
        first_contacts=sum(leg["near_contact"] is not None for leg in first),
        contacts=sum(leg["near_contact"] is not None for leg in legs),
        cpu_s=sum(row["runtime_cpu_s"] for row in rows),
    )
    return result


def choose(output, observations):
    baseline = _read(OUTPUT / "paired-practice-fixed.json.gz")
    variants = [_read(output / f"reach-practice-{mode}.json.gz") for mode in ("0", "1")]
    if any(len(data["rows"]) != len(baseline["cards"]) or data["cards"] != baseline["cards"] for data in variants):
        raise ValueError("Practice comparison is incomplete")
    result = [scores(data["rows"]) for data in variants]
    i = max(range(len(result)), key=lambda k: (result[k]["own"], result[k]["first_contacts"], -k))
    _write(output / "reach-checkpoint.json", dict(
        authority_rad=AUTHORITY[i], selected=i, scores=result, baseline=scores(baseline["rows"]),
        observations_sha256=_hash(observations),
    ))
    print("selected", i, result, flush=True)



def compare(control, tested):
    if control["cards"] != tested["cards"] or len(tested["rows"]) != len(control["rows"]):
        raise ValueError("Mismatched paired course")
    gained = []
    lost = []
    lines = 0
    rollout = []
    for original, feedback in zip(control["rows"], tested["rows"], strict=True):
        a, b = (row["placements"]["delivered"][0] for row in (original, feedback))
        label = original["shot"]["id"]
        if any(digest_json(a[field]) != digest_json(b[field]) for field in ("far_contact", "near_bounce")):
            raise ValueError("The first physical offer changed")
        if digest_json(a["receive_line"]) == digest_json(b["receive_line"]):
            lines += 1
        old, new = a["near_contact"] is not None, b["near_contact"] is not None
        if new and not old:
            gained.append(label)
        if old and not new:
            lost.append(label)
        rollout.append(dict(id=label, first_contacts=[old, new],
                            near_legal=[original["score"]["learner"], feedback["score"]["learner"]]))
    active = [tick for row in tested["rows"] for tick in row["active_corrections"]]
    return dict(
        tested=scores(tested["rows"]), active_ticks=len(active),
        sight_anchored_ticks=sum(row["live_blade"] for row in active),
        peak_applied_offset_rad=max((max(abs(np.asarray(row["correction_rad"]))) for row in active), default=0.),
        max_measured_joint_speed_rad_s=max(max(row["motor_audit"]["max_measured_rate_rad_s"]) for row in tested["rows"]),
        exact_receiving_line=lines, gained_opening_contacts=gained, lost_opening_contacts=lost, per_point=rollout,
    )


def report(output, *, confirmation=False):
    cohorts = {"confirm": ("aligned",)} if confirmation else {"new": ("aligned", "reversed"), "replies": ("aligned",)}
    reports = {}
    for cohort, modes in cohorts.items():
        fixed_path = OUTPUT / f"paired-{cohort}-fixed.json.gz"
        fixed = _read(fixed_path)
        measured = {}
        for mode in modes:
            path = output / f"reach-{cohort}-{mode}.json.gz"
            measured[mode] = compare(fixed, _read(path)) | {"sha256": _hash(path)}
        reports[cohort] = dict(fixed=scores(fixed["rows"]), fixed_sha256=_hash(fixed_path), feedback=measured)
    name = "reach-confirm-results.json" if confirmation else "reach-results.json"
    _write(output / name, reports)
    for name, item in reports.items():
        print(name, item["fixed"], {mode: record["tested"] for mode, record in item["feedback"].items()}, flush=True)


def audit_contacts(output, mode, cohort):
    """Replay every scoring or newly contacting point with full arm clearance checks."""
    baseline = _read(OUTPUT / f"paired-{cohort}-fixed.json.gz")
    path = output / f"reach-{cohort}-{mode}.json.gz"
    source = _read(path)
    selected = []
    for card, current, original in zip(source["cards"], source["rows"], baseline["rows"], strict=True):
        original_legs = original["placements"]["delivered"]
        current_legs = current["placements"]["delivered"]
        first_gain = current_legs[0]["near_contact"] is not None and original_legs[0]["near_contact"] is None
        if current["score"]["learner"] or first_gain:
            selected.append((card, current))
    audit_path = output / f"reach-{cohort}-{mode}-audit.json"
    if not selected:
        _write(audit_path, dict(source_sha256=_hash(path), points=[]))
        return
    with TemporaryDirectory(prefix="tt36-reach-") as scratch:
        tape = Path(scratch) / "subset.json.gz"
        verified = Path(scratch) / "audit.json"
        cards, rows = map(list, zip(*selected, strict=True))
        _write(tape, dict(source, cards=cards, rows=rows))
        verify(tape, verified, cad=True)
        result = _read(verified)
    _write(audit_path, dict(result, source_sha256=_hash(path)))


def audit_later(output):
    """Verify later legal gains while reproducing each opponent continuation seed."""
    path = output / "reach-replies-aligned.json.gz"
    reached = _read(path)
    original = _read(OUTPUT / "paired-replies-fixed.json.gz")
    plan_path = OUTPUT / "course-replies.json.gz"
    plan = _read(plan_path)
    if (reached["cards"] != plan["cards"] or reached["cards"] != original["cards"]
            or reached["course_sha256"] != _hash(plan_path)):
        raise ValueError("Reply treatment changed")
    results = []
    with TemporaryDirectory(prefix="tt36-reach-reply-") as scratch:
        source = Path(scratch) / "selected.json.gz"
        response = Path(scratch) / "audit.json"
        for card, seed, current, baseline in zip(
            plan["cards"], plan["reply_seeds"], reached["rows"], original["rows"], strict=True
        ):
            if current["score"]["learner"] <= max(1, baseline["score"]["learner"]):
                continue
            _write(source, dict(reached, cards=[card], rows=[current]))
            with varied_opponent(card["index"], seed):
                verify(source, response, cad=True)
            result, = _read(response)["points"]
            results.append(dict(result, reply_seed=seed))
    _write(output / "reach-replies-aligned-audit.json", dict(source_sha256=_hash(path), points=results))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="phase", required=True)
    capture = actions.add_parser("prepare")
    capture.add_argument("recording", type=Path)
    capture.add_argument("output", type=Path)
    run_parser = actions.add_parser("run")
    run_parser.add_argument("feeds", type=Path)
    run_parser.add_argument("cohort", choices=("practice", "new", "replies", "confirm"))
    run_parser.add_argument("modes", nargs="+", choices=("0", "1", "aligned", "reversed"))
    for item in (run_parser, actions.add_parser("choose")):
        item.add_argument("--observations", type=Path, default=OUTPUT / "reach-observations.npz")
        item.add_argument("--output", type=Path, default=OUTPUT)
    summary = actions.add_parser("report")
    summary.add_argument("--output", type=Path, default=OUTPUT)
    summary.add_argument("--confirmation", action="store_true")
    replay = actions.add_parser("audit")
    replay.add_argument("mode", choices=("aligned", "reversed"))
    replay.add_argument("--output", type=Path, default=OUTPUT)
    replay.add_argument("--cohort", choices=("new", "confirm", "replies"), default="new")
    args = parser.parse_args()
    if args.phase == "prepare":
        args.output.parent.mkdir(exist_ok=True, parents=True)
        prepare(args.recording, args.output)
    elif args.phase == "run":
        run(args.feeds, args.observations, args.output, args.cohort, args.modes)
    elif args.phase == "choose":
        choose(args.output, args.observations)
    elif args.phase == "report":
        report(args.output, confirmation=args.confirmation)
    elif args.cohort == "replies":
        if args.mode != "aligned":
            parser.error("Only the aligned continuation treatment was tested")
        audit_later(args.output)
    else:
        audit_contacts(args.output, args.mode, args.cohort)


if __name__ == "__main__":
    main()

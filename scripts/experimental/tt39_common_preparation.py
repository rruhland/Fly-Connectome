"""Preload only the encoder movements shared by both wings after a legal shot."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import OUTPUT as TT37, observed
from .tt37_joint_deep import DeepField
from .tt37_joint_deep_report import totals
from .tt37_joint_report import paired, read_complete, replay_case
from .tt37_opponent import arena
from .tt38_body_relative import Body, groups
from .tt39_recovery_audit import OUT

BANK = TT37 / "self-combined.npz"
BODY = TT37.parent / "tt38-body-relative" / "body.npz"
BASE = TT37 / "self-combined-held_openings.json.gz"
STRIKE_TIME = -.28  # Training slice only. No impact time is available to the actor.


def onset_samples(bank, receipt):
    s, mark = bank["states"], bank["marks"]
    point, release = groups(receipt)
    picked = {}
    for i in np.flatnonzero((s[:, 4] < -.90) & (abs(mark[:, 1] - STRIKE_TIME) < .050)):
        # Marks share the actual index of the same archived legal contact.
        key = point[i], int(round(mark[i, 0] - 60 * mark[i, 1]))
        if key not in picked or abs(mark[i, 1] - STRIKE_TIME) < abs(mark[picked[key], 1] - STRIKE_TIME):
            picked[key] = int(i)
    ix = np.array(list(picked.values()), dtype=int)
    return s[ix, 6:11], release[ix]


def common_pose(q, home):
    """Pointwise strong agreement in both observed wings, not a mean stroke."""
    delta = q - home[:5]
    # Shoulder pan separates the two practiced motor wings. q[0] is not a target x label.
    groups_ = (delta[:, 0] < -.115, delta[:, 0] > .115, abs(delta[:, 0]) <= .115)
    if any(g.sum() < 8 for g in groups_):
        raise ValueError("Too few legal observations on a motor wing")
    sign = np.sign(np.median(delta, axis=0))
    aligned = delta * sign
    support = np.array([np.mean(aligned[g] > .035, axis=0) for g in groups_])
    keep = np.min(support, axis=0) >= .72
    # The lateral pan and paddle roll distinguish the branches, so do not commit either here.
    keep[[0, 4]] = False
    change = .72 * np.clip(np.median(delta, axis=0), -.45, .45) * keep
    out = home.copy()
    out[:5] += change
    return out, support


def calibrate(bank_path, path):
    receipt = _read(bank_path.with_suffix(".json"))
    if _hash(bank_path) != receipt["bank_sha256"]:
        raise ValueError("The practice receipt changed")
    home = np.asarray(receipt["ready"])
    with np.load(bank_path) as d:
        q, r = onset_samples(d, receipt)
    with np.load(BODY) as fitted:
        body = Body(fitted["weights"])
    pure = np.max(abs(q - home[:5]), axis=1)
    selected = np.empty_like(q)
    for release in np.unique(r):
        test = r == release
        selected[test] = common_pose(q[~test], home)[0][:5]
    warm = np.max(abs(q - selected), axis=1)
    goal, support = common_pose(q, home)
    result = dict(home=home.tolist(), goal=goal.tolist(), angle_change_rad=(goal-home).tolist(),
                  nominal_home_blade=body.position(home).tolist(), nominal_ready_blade=body.position(goal).tolist(),
                  strokes=len(q), releases=len(set(r)), training_s_before_strike=STRIKE_TIME,
                  wing_agreement=support.tolist(),
                  held_peak_joint_travel_rad={k: dict(median=float(np.median(y)), p90=float(np.quantile(y, .9)))
                                              for k, y in (("stock", pure), ("preload", warm))},
                  split="one full release out at a time", bank_sha256=_hash(bank_path), body_sha256=_hash(BODY),
                  validation="Travel is a joint-angle proxy, not an audited motor trajectory or a return score")
    _write(path, result)
    print(result, flush=True)


class Preload:
    """Only the public legal-return pulse opens the next preparation window."""

    def __init__(self, goal, enabled):
        self.goal = np.asarray(goal)
        self.enabled = enabled
        self.active = None
        self.trace = []

    def scored(self, tick, q):
        if self.enabled:
            self.active = dict(start_tick=tick+1, start_q=q.tolist(), issued_ticks=0)
            self.trace.append(self.active)

    def change(self, tick, optical, q):
        note = self.active
        if note is None:
            return None
        if (optical is not None and optical[4] < -.90) or tick-note["start_tick"] >= 45:
            note["handoff_tick"] = tick
            note["handoff_q"] = q.tolist()
            note["handoff_observation"] = None if optical is None else optical.tolist()
            self.active = None
            return None
        note["issued_ticks"] += 1
        return self.goal


def play(a, renderer, rig, stride, low, high, home, card, shot, field, goal, enabled, cap):
    a.launch(shot, card=card)
    a.partner.reset(seed=card["seed"])
    eye = Vision(rig, a.control_dt, raw=True)
    controller = Preload(goal, enabled)
    reference = home.copy()
    sensed_inbound = False
    command_tape = []
    field_ticks = 0
    score_count = 0
    start = process_time()
    for tick in range(a.max_steps):
        q = a.joints
        view = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q)
        if view is not None and view[4] < -.9 and view[1] < -.90:
            sensed_inbound = True
        desired = home
        if view is not None and (view[4] < -.9 or (sensed_inbound and view[1] < .50)):
            guess, confidence = field.get(view, q, reference=reference)
            if guess is not None:
                desired = home + confidence * (guess-home)
                field_ticks += 1
        prep = controller.change(tick, view, q)
        if prep is not None:
            desired = prep
        u = np.clip((np.clip(desired, low, high) - reference) / stride, -.8, .8)
        reference = np.clip(reference + u*stride, low, high)
        _, _, done, capped, score = a.exchange(u, FAR_COMMAND)
        command_tape.append(u.tolist())
        if any(e["kind"] == "legal_return" and e["side"] == 1 for e in score["events"]):
            own = audit(a.events, opening_side=1)["learner"]
            if own > score_count:
                controller.scored(tick, a.joints)
                score_count = own
        if done or capped or a.legal_returns + a.rival_returns - 1 >= cap:
            break
    return dict(actions=command_tape, score=audit(a.events, opening_side=1), shot=shot,
                events=a.events, placements=a.point_placements(), opponent_seed=card["seed"],
                motor_audit=a.motor_audit(), runtime_cpu_s=process_time()-start,
                field_ticks=field_ticks, preloads=controller.trace)


def run(bank, frozen, turn, feeds, mode, output, limit):
    calibration = _read(turn)
    receipt = _read(bank.with_suffix(".json"))
    previous = read_complete(frozen)
    if ({_hash(bank), calibration["bank_sha256"], receipt["bank_sha256"], previous["source_sha256"]}
            != {_hash(bank)} or calibration["home"] != receipt["ready"] or calibration["home"] != previous["ready"]):
        raise ValueError("Paired assay requires the identical arm and instruction bank")
    if calibration["held_peak_joint_travel_rad"]["preload"]["median"] >= calibration["held_peak_joint_travel_rad"]["stock"]["median"]:
        raise ValueError("Observed movement did not support a preload")
    keep = [i for i, r in enumerate(previous["rows"]) if r["score"]["learner"] > 0]
    keep = keep[:limit]
    cards = [previous["cards"][i] for i in keep]
    home, goal = np.asarray(calibration["home"]), np.asarray(calibration["goal"])
    launches = measured_launches(feeds)
    with np.load(bank) as data:
        field = DeepField(data, "local")
    doc = dict(cards=cards, rows=[], ready=home.tolist(), cap=previous["cap"], deck=previous["deck"],
               opening_side=1, source_sha256=_hash(bank), control_sha256=_hash(frozen), preload_sha256=_hash(turn),
               mode=mode, included_indices=keep, inclusion="Every returned opening in the frozen physical control. No pre-opening action changes.")
    with centered(home):
        a = arena(protocol="random")
        renderer, rig, stride, lower, upper = resources(a)
        try:
            for i, c in zip(keep, cards, strict=True):
                r = play(a, renderer, rig, stride, lower, upper, home, c, launches[c["id"]], field, goal,
                         mode == "preload", previous["cap"])
                baseline = previous["rows"][i]
                hit = next(s["hit"] for s in baseline["score"]["shots"] if s["side"] == 1)
                first_legal = next(e for e in baseline["events"] if e["kind"] == "legal_return" and e["side"] == 1 and e["time"] > hit)
                unmodified = int(first_legal["step"])
                if r["actions"][:unmodified] != baseline["actions"][:unmodified]:
                    raise ValueError("The preload did not reproduce every command up to the opening legal return")
                if mode == "neutral" and (r["actions"] != baseline["actions"] or digest_json(r["events"]) != digest_json(baseline["events"])):
                    raise ValueError("Calibration mode failed exact baseline reproduction")
                doc["rows"].append(r)
                _write(output, doc)
                print(c["id"], baseline["score"]["learner"], r["score"]["learner"], len(r["preloads"]), flush=True)
        finally:
            renderer.close()
            a.close()


def compare(old, changed, output, *, replay=False, checkpoint=OUT/"preload.json"):
    original, active = read_complete(old), read_complete(changed)
    if _hash(old) != active["control_sha256"]:
        raise ValueError("Frozen control was replaced")
    keep = active["included_indices"]
    paired_original = dict(original, cards=[original["cards"][i] for i in keep], rows=[original["rows"][i] for i in keep])
    control_stats, active_stats = totals(paired_original), totals(active)
    same = paired(paired_original, active)
    identical_first_reply = 0
    for a, b in zip(paired_original["rows"], active["rows"], strict=True):
        ad, bd = [r["placements"]["delivered"] for r in (a, b)]
        if len(ad) > 1 and len(bd) > 1:
            identical_first_reply += all(digest_json(ad[1][k]) == digest_json(bd[1][k])
                                         for k in ("target", "far_contact", "near_bounce", "receive_line"))
    out = dict(control=control_stats, preload=active_stats, paired=same,
               same_first_later_offer=identical_first_reply,
               control_sha256=_hash(old), preload_sha256=_hash(changed))
    if _hash(checkpoint) != active["preload_sha256"]:
        raise ValueError("Ready target differs from the physical rehearsal")
    goal = np.asarray(_read(checkpoint)["goal"])
    handoffs = [p for r in active["rows"] for p in r["preloads"] if "handoff_q" in p]
    if handoffs:
        travel = [float(np.max(abs(np.array(p["handoff_q"]) - goal))) for p in handoffs]
        seconds = [p["issued_ticks"] / 60. for p in handoffs]
        out["observed_handoffs"] = dict(n=len(handoffs), median_s=float(np.median(seconds)),
                                        median_remaining_rad=float(np.median(travel)),
                                        within_tolerance_count=int(np.count_nonzero(np.asarray(travel) < .055)))
    if replay:
        gain = np.array([r["near"][1]-r["near"][0] for r in same["rows"]])
        if len(gain) and gain.max() > 0:
            i = int(np.argmax(gain))
            out["replay"] = replay_case(active["ready"], "random", active["cards"][i], active["rows"][i])
    _write(output, out)
    print(control_stats["later"], active_stats["later"], same["wins"], same["losses"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    phases = p.add_subparsers(dest="phase", required=True)
    fit = phases.add_parser("fit")
    fit.add_argument("output", type=Path)
    fit.add_argument("--bank", type=Path, default=BANK)
    test = phases.add_parser("test")
    test.add_argument("output", type=Path)
    test.add_argument("--bank", type=Path, default=BANK)
    test.add_argument("--control", type=Path, default=BASE)
    test.add_argument("--checkpoint", type=Path, default=OUT/"preload.json")
    test.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    test.add_argument("--mode", choices=("neutral", "preload"), default="preload")
    test.add_argument("--limit", type=int)
    report = phases.add_parser("report")
    report.add_argument("output", type=Path)
    report.add_argument("changed", type=Path)
    report.add_argument("--control", type=Path, default=BASE)
    report.add_argument("--checkpoint", type=Path, default=OUT/"preload.json")
    report.add_argument("--replay", action="store_true")
    args = p.parse_args()
    if args.phase == "fit":
        calibrate(args.bank, args.output)
    elif args.phase == "test":
        run(args.bank, args.control, args.checkpoint, args.feeds, args.mode, args.output, args.limit)
    else:
        compare(args.control, args.changed, args.output, replay=args.replay, checkpoint=args.checkpoint)

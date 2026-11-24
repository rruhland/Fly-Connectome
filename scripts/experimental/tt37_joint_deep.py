"""Distill one joint-setpoint update from successful deep-mentor RGB and encoder observations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_joint_capture import OUTPUT, observed, samples
from .tt37_opponent import arena


MODES = ("local", "depth")
SCALES = {
    "local": np.array([.15, .115, .14, 1.1, 1.4, 1.6, .90, 1.0, 1.15, .84, .90]),
    "depth": np.array([.14, .073, .115, 1., 1.45, 1.75, .85, .85, 1.05, .85, .85]),
}
DEPTH = -1.56


def collect(sources, output):
    cases = {}
    ready = None
    hashes = {}
    for path in sources:
        snapshot = path.read_bytes()
        tape = json.loads(gzip.decompress(snapshot) if path.suffix == ".gz" else snapshot)
        if len(tape["cards"]) != len(tape["rows"]):
            raise ValueError(f"Mentor source is still incomplete: {path.name}")
        if not np.isclose(tape["depth"], DEPTH):
            raise ValueError("Different depth branch in the deep mentor data")
        if ready is None:
            ready = tape["ready"]
        elif ready != tape["ready"]:
            raise ValueError("Cannot mix centered and left-ready strike data")
        hashes[path.name] = hashlib.sha256(snapshot).hexdigest()
        for index, (card, row) in enumerate(zip(tape["cards"], tape["rows"], strict=True)):
            if not row["score"]["learner"]:
                continue
            key = card["id"], card["seed"], card["reply_seed"]
            stored = cases.get(key)
            if stored is None or row["score"]["learner"] > stored[1]["score"]["learner"]:
                cases[key] = (card, row, path.name, index)
    if ready is None or not cases:
        raise ValueError("No verified deep-mentor return is available")
    packets, history = [], []
    with centered(ready):
        environment = arena(protocol="random")
        try:
            renderer, rig, *_ = resources(environment)
            try:
                for card, row, name, index in cases.values():
                    packet = samples(environment, renderer, rig, card, row, None)
                    packets.append(packet)
                    history.append(dict(card=card, source=name, source_row=index, mentor_returns=row["score"]["learner"],
                                        frames=len(packet[0])))
                    print("deep replay", card["id"], row["score"]["learner"], len(packet[0]), flush=True)
            finally:
                renderer.close()
        finally:
            environment.close()
    states, targets, lead, marks = (np.concatenate(chunk) for chunk in zip(*packets, strict=True))
    np.savez_compressed(output, states=states[:, :11], targets_now=targets, targets_lead=lead, marks=marks)
    _write(output.with_suffix(".json"), dict(
        ready=ready, depth=DEPTH, sources=hashes, practice=history, bank_sha256=_hash(output),
        frames=len(states), legal=sum(row["mentor_returns"] for row in history),
    ))


class DeepField:
    """Interpolate one desired joint vector using current ball and encoder state."""

    def __init__(self, archive, mode):
        self.scale = SCALES[mode]
        states = archive["states"]
        labels = archive["targets_now" if mode == "local" else "targets_lead"]
        self.branches = {}
        for sign in (-1, 1):
            use = np.sign(states[:, 4]) == sign
            x = states[use] / self.scale
            if len(x) < 12:
                raise ValueError("Not enough camera-visible states in the deep corpus")
            self.branches[sign] = cKDTree(x), x, labels[use]

    def get(self, optical, q, *, reference=None):
        center = np.r_[optical, q[:5]] / self.scale
        tree, x, label = self.branches[-1 if optical[4] < 0. else 1]
        dist, idx = tree.query(center, 12)
        confidence = float(np.clip((2.6 - dist[2]) / 1.0, 0., 1.))
        if not confidence:
            return None, 0.
        offset = x[idx] - center
        weight = np.exp(-.5 * (dist / max(.14, dist[4]))**2)
        weight /= weight.sum()
        mean, reference = weight @ offset, weight @ label[idx]
        dev = offset - mean
        fit = np.linalg.solve(dev.T @ (weight[:, None] * dev) + .045 * np.eye(len(center)),
                              dev.T @ (weight[:, None] * (label[idx] - reference)))
        return reference - np.clip(mean @ fit, -.23, .23), confidence


def play(environment, renderer, rig, stride, lower, upper, ready, card, shot, field, cap):
    environment.launch(shot, card=card)
    environment.partner.reset(seed=card["seed"])
    eye = Vision(rig, environment.control_dt, raw=True)
    reference = np.array(ready)
    home = reference.copy()
    sensed_inbound = False
    commands = []
    field_ticks = 0
    if hasattr(field, "begin_point"):
        field.begin_point()
    score_feedback = getattr(field, "score_feedback", None)
    started = process_time()
    for tick in range(environment.max_steps):
        q = environment.joints
        if hasattr(field, "clock_tick"):
            field.clock_tick(tick)
        optical = observed(eye, tick, renderer.capture() if tick % 2 == 0 else None, q) if field is not None else None
        if optical is not None and optical[4] < -.9 and optical[1] < -.90:
            sensed_inbound = True
        desired = home
        if optical is not None and (optical[4] < -.9 or (sensed_inbound and optical[1] < .50)):
            goal, confidence = field.get(optical, q, reference=reference)
            if goal is not None:
                desired = home + confidence * (goal - home)
                field_ticks += 1
        command = np.clip((np.clip(desired, lower, upper) - reference) / stride, -.8, .8)
        reference = np.clip(reference + command * stride, lower, upper)
        _, _, done, capped, referee = environment.exchange(command, FAR_COMMAND)
        commands.append(command.tolist())
        if score_feedback is not None and any(e["kind"] == "legal_return" and e["side"] == 1
                                              for e in referee["events"]):
            score_feedback(audit(environment.events, opening_side=1)["learner"])
        if done or capped or environment.legal_returns + environment.rival_returns - 1 >= cap:
            break
    if hasattr(field, "end_point"):
        field.end_point(terminal=bool(done and not capped))
    row = dict(
        shot=shot, opponent_seed=card["seed"], actions=commands, events=environment.events,
        score=audit(environment.events, opening_side=1), placements=environment.point_placements(),
        field_ticks=field_ticks, motor_audit=environment.motor_audit(), runtime_cpu_s=process_time() - started,
    )
    if hasattr(field, "diagnostics"):
        row["controller_diagnostics"] = field.diagnostics
    return row


def cards_for(receipt, plan, deck):
    if deck == "familiar":
        return [item["card"] for item in receipt["practice"]]
    if deck == "fresh_replies":
        unique = {item["card"]["id"]: item["card"] for item in receipt["practice"]}
        cards = [dict(card, reply_seed=372760 + i * 9 + j)
                 for i, card in enumerate(unique.values()) for j in range(2)]
        used = {item["card"]["reply_seed"] for item in receipt["practice"]}
        if any(card["reply_seed"] in used for card in cards):
            raise ValueError("Redrawn reply uses a mentor's frozen reply seed")
        return cards
    return _read(plan)[deck]


def run(feeds, archive, plan, deck, mode, output, *, cap, limit=None, field_hook=None, details=None,
        cards_override=None):
    source = _read(archive.with_suffix(".json"))
    if source["bank_sha256"] != _hash(archive):
        raise ValueError("Deep model receipt changed")
    cards = cards_for(source, plan, deck) if cards_override is None else cards_override
    if limit is not None:
        cards = cards[:limit]
    if deck == "held_openings" and {card["id"] for card in cards}.intersection(
            item["card"]["id"] for item in source["practice"]):
        raise ValueError("Novel opening appears in the deep training data")
    shots = measured_launches(feeds)
    ready = source["ready"]
    result = dict(ready=ready, mode=mode, deck=deck, cards=cards, opening_side=1, cap=cap, rows=[],
                  source_sha256=_hash(archive), plan_sha256=_hash(plan))
    if details is not None:
        result["candidate_source"] = details
    with np.load(archive) as bank:
        field = DeepField(bank, mode) if mode != "ready" else None
    if field_hook is not None:
        field = field_hook(field)
    with centered(ready):
        environment = arena(protocol="random")
        try:
            renderer, rig, stride, lower, upper = resources(environment)
            try:
                for card in cards:
                    row = play(environment, renderer, rig, stride, lower, upper, ready,
                               card, shots[card["id"]], field, cap)
                    result["rows"].append(row)
                    _write(output, result)
                    print(mode, deck, card["id"], row["score"]["learner"],
                          row["score"]["returns"], row["field_ticks"], flush=True)
            finally:
                renderer.close()
        finally:
            environment.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    phases = p.add_subparsers(dest="phase", required=True)
    capture = phases.add_parser("capture")
    capture.add_argument("sources", type=Path, nargs="+")
    capture.add_argument("--output", type=Path, default=OUTPUT / "joint-deep-observations.npz")
    assess = phases.add_parser("run")
    assess.add_argument("feeds", type=Path)
    assess.add_argument("deck", choices=("familiar", "fresh_replies", "held_openings", "held_replies"))
    assess.add_argument("mode", choices=("ready", *MODES))
    assess.add_argument("output", type=Path)
    assess.add_argument("--bank", type=Path, default=OUTPUT / "joint-deep-observations.npz")
    assess.add_argument("--plan", type=Path, default=OUTPUT / "opponent-plan.json")
    assess.add_argument("--cap", type=int, default=24)
    assess.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.phase == "capture":
        collect(args.sources, args.output)
    else:
        run(args.feeds, args.bank, args.plan, args.deck, args.mode, args.output, cap=args.cap, limit=args.limit)


if __name__ == "__main__":
    main()

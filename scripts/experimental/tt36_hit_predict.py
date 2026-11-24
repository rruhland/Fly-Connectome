"""Test whether issued commands and measured encoders predict optically observed outgoing velocity."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_optical_outcomes import stereo_position
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt36_hit import ARTIFACTS, OUTPUT, optical_outcomes


REFERENCE = ARTIFACTS / "tt34" / "body-novel-stock.json.gz"
HELD_IGNORE = {3025, 8239, 5530, 14832}
MOTOR_LEAD_TICKS = 4


def replay(arena, card, row, renderer=None, rig=None):
    """Recover the same encoder readings, and stereo frames when needed, from a physical tape."""
    arena.launch(row["shot"], card=card)
    arena.partner.reset(seed=row["opponent_seed"])
    eye = Vision(rig, arena.control_dt, raw=True) if renderer is not None else None
    encoders, captured = [], []
    for tick, action in enumerate(row["actions"]):
        q = arena.joints
        encoders.append(q[:5])
        if renderer is not None and tick % 2 == 0:
            x = eye.update(tick, renderer.capture(), q)
            p = stereo_position(eye.eye)
            if p is not None:
                captured.append((tick * arena.control_dt, p, x))
        arena.exchange(action, FAR_COMMAND)
    if digest_json(arena.events) != digest_json(row["events"]):
        raise ValueError("Physical action replay diverged")
    packets = optical_outcomes(captured) if renderer is not None else row["optical_outgoing"]
    contacts = [e["time"] for e in arena.events if e["kind"] == "contact" and e["side"] == 1]
    rows = []
    for packet in packets:
        reversal = packet["reversal_time"]
        index = max(1, round(reversal / arena.control_dt) - MOTOR_LEAD_TICKS)
        q = encoders[index]
        dq = (q - encoders[index - 1]) / arena.control_dt
        command = row["actions"][index - 1][:5]
        nearest = min(contacts, key=lambda time: abs(time - reversal), default=None)
        true_lead = None if nearest is None else nearest - index * arena.control_dt
        rows.append(dict(
            id=card["id"], seed=card["seed"], optical=packet["incoming"], q=q.tolist(), qdot=dq.tolist(),
            issued=command, observed_outgoing_vxyz=packet["velocity"],
            observed_outgoing_time=packet["time"], encoder_time=index * arena.control_dt,
            optical_reversal_time=reversal, audit_encoder_lead_s=true_lead,
            audit_optical_lead_s=(nearest - packet["incoming_time"] if nearest is not None
                                  and "incoming_time" in packet else None),
        ))
    return rows


def collect(practice_path, held_path):
    practice = _read(practice_path)
    held = _read(held_path)
    seen = {c["id"] for c in practice["cards"]}
    eligible = [(c, row) for c, row in zip(held["cards"], held["rows"], strict=True)
                if c["id"] not in HELD_IGNORE]
    if not seen.isdisjoint(c["id"] for c, _ in eligible) or practice["ready"] != held["ready"]:
        raise ValueError("The diagnostic holdout must use the same pose and unseen release IDs")
    train, test = [], []
    with centered(practice["ready"]):
        arena = replay_arena([])
        try:
            for card, trials in zip(practice["cards"], practice["rounds"], strict=True):
                for row in trials:
                    train.extend(replay(arena, card, row))
            renderer, rig, *_ = resources(arena)
            try:
                for card, row in eligible:
                    signals = replay(arena, card, row, renderer, rig)
                    test.extend(signals)
                    print(card["id"], len(signals), flush=True)
            finally:
                renderer.close()
        finally:
            arena.close()
    return train, test, [c for c, _ in eligible]


def inputs(rows, *, motor):
    incoming = np.array([row["optical"] for row in rows])
    if not motor:
        return incoming
    proprioception = np.array([row["q"] + row["qdot"] + row["issued"] for row in rows])
    return np.concatenate((incoming, proprioception), axis=1)


def regress(train, other, *, motor):
    x = inputs(train, motor=motor)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    standardized = (x - mean) / scale
    y = np.array([row["observed_outgoing_vxyz"] for row in train])
    counts = Counter(row["id"] for row in train)
    weights = np.sqrt([1. / counts[row["id"]] for row in train])
    baseline = np.average(y, axis=0, weights=weights)
    gram = standardized.T @ (weights[:, None] * standardized)
    gram.flat[::len(gram) + 1] += 4.
    coef = np.linalg.solve(gram, standardized.T @ (weights[:, None] * (y - baseline)))
    predictions = (inputs(other, motor=motor) - mean) / scale @ coef + baseline
    return predictions


def metrics(actual, predicted):
    diff = predicted - actual
    true_speed = np.linalg.norm(actual, axis=1)
    pred_speed = np.linalg.norm(predicted, axis=1)
    cos = np.sum(actual * predicted, axis=1) / np.maximum(true_speed * pred_speed, 1.e-8)
    supported = abs(actual[:, 0]) > .20
    return dict(rms_by_component_mps=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
                vector_rms_mps=float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))),
                speed_mae_mps=float(np.mean(abs(pred_speed - true_speed))),
                direction_median_error_deg=float(np.median(np.rad2deg(np.arccos(np.clip(cos, -1., 1.))))),
                lateral_sign_hits=int(np.sum(supported & (np.sign(predicted[:, 0]) == np.sign(actual[:, 0])))),
                lateral_sign_chances=int(supported.sum()))


def experiment(practice, held, output):
    train, test, cards = collect(practice, held)
    actual = np.array([row["observed_outgoing_vxyz"] for row in test])
    evidence = {}
    for motor in False, True:
        name = "incoming_and_motor" if motor else "incoming"
        prediction = regress(train, test, motor=motor)
        evidence[name] = dict(held=metrics(actual, prediction), predicted=prediction.tolist())
        grouped = []
        for row in train:
            comparable = [r for r in train if r["id"] != row["id"]]
            grouped.extend(regress(comparable, [row], motor=motor))
        evidence[name]["leave_release_out_practice"] = metrics(
            np.array([r["observed_outgoing_vxyz"] for r in train]), np.asarray(grouped))
    _write(output, dict(
        practice_sha256=_hash(practice), held_sha256=_hash(held), held_cards=cards, training=train,
        test=test, models=evidence,
        motor_lead_control_ticks=MOTOR_LEAD_TICKS,
    ))
    print({name: scores["held"] for name, scores in evidence.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", type=Path, default=OUTPUT / "hit-practice.json.gz")
    parser.add_argument("--held", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT / "hit-prediction.json.gz")
    args = parser.parse_args()
    experiment(args.practice, args.held, args.output)

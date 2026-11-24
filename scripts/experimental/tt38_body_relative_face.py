"""Decouple center and blade face within one direct encoder-conditioned joint update."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK, mujoco
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_joint_deep import DeepField, run
from .tt37_opponent import arena
from .tt38_body_relative import BANK, OUT, PLAN, Body, RelativeField, embedding


def pose_calibration(output):
    a = arena(protocol="random")
    try:
        fixture = BladeIK(a)
        lower, upper = fixture.low, fixture.high
        joints = qmc.scale(qmc.Sobol(5, scramble=True, seed=383019).random_base2(11), lower, upper)
        position = []
        normals = []
        for q in joints:
            # The outward face retains its sign at different wrist angles.
            position.append(fixture.pose(q)[0])
            rot = fixture.data.xmat[fixture.gripper].reshape(3, 3)
            normals.append(rot @ fixture.normal)
    finally:
        a.close()
    labels = np.column_stack((position, normals))
    x = embedding(joints)
    w = np.linalg.lstsq(x[:1536], labels[:1536], rcond=1.e-11)[0]
    error = x[1536:] @ w - labels[1536:]
    np.savez_compressed(output, weights=w, lower=lower, upper=upper)
    _write(output.with_suffix(".json"), dict(training_poses=1536, test_poses=512,
           mean_center_m=float(np.mean(np.linalg.norm(error[:, :3], axis=1))),
           mean_normal_error=float(np.mean(np.linalg.norm(error[:, 3:], axis=1))),
           fit_sha256=_hash(output), purpose="Offline body-only position and signed face calibration."))


def single_joint_command(relative, absolute, pose_map, lower, upper):
    """Keep the center of the relative readout and angle of the absolute readout."""
    center_move = pose_map.position(relative)[:3]
    desired_face = pose_map.position(absolute)[3:6]
    face_scale = .115
    # One unadjusted setup makes this an action update rather than a kinematic planner.
    base = relative.copy()
    face_now = pose_map.position(base)[3:6]
    if desired_face @ face_now < .60:
        return base
    weighting = np.r_[np.ones(3), np.full(3, face_scale)]
    wish = np.r_[center_move, desired_face]
    allowed = np.minimum(1., np.maximum(upper - base[:5], base[:5] - lower) / .22)
    prior = np.diag(np.array([.8, .36, .88, .95, 1.]) * allowed)
    j = (pose_map.jacobian(base) * weighting[:, None]) @ prior
    mismatch = weighting * (wish - pose_map.position(base))
    changes = prior @ np.linalg.solve(j.T @ j + .017**2 * np.eye(5), j.T @ mismatch)
    base[:5] = np.clip(base[:5] + np.clip(changes, -.17, .17), lower, upper)
    return base


class FaceField:
    def __init__(self, bank, receipt, calibration):
        self.pose = Body(calibration["weights"])
        self.low, self.high = calibration["lower"], calibration["upper"]
        self.absolute = DeepField(bank, "local")
        self.relative = RelativeField(bank, receipt, calibration["weights"][:, :3], "position")
        self.begin_point()

    def begin_point(self):
        self.relative.begin_point()
        self.diagnostics = dict(face_anchors=0, motor_adjustment_rad_sum=0.)

    def clock_tick(self, tick):
        self.relative.clock_tick(tick)

    def get(self, optical, q, *, reference=None):
        position, p_conf = self.relative.get(optical, q, reference=reference)
        if position is None or optical[4] >= -.9:
            return position, p_conf
        absolute, a_conf = self.absolute.get(optical, q, reference=reference)
        if absolute is None:
            return position, p_conf
        result = single_joint_command(position, absolute, self.pose, self.low, self.high)
        result = position + a_conf * (result - position)
        self.diagnostics["face_anchors"] += 1
        self.diagnostics["motor_adjustment_rad_sum"] += float(np.linalg.norm(result - position))
        return result, p_conf


def physical(bank, pose, feeds, deck, output):
    receipt = _read(bank.with_suffix(".json"))
    with np.load(bank) as doc, np.load(pose) as calibrated:
        field = FaceField(doc, receipt, calibrated)
    run(feeds, bank, PLAN, deck, "local", output, cap=200, field_hook=lambda _: field,
        details=dict(body_coordinates="position-with-absolute-face", body_pose_sha256=_hash(pose)))


def audit_pair(control, changed, output):
    test_cards = {3979, 6458, 10842, 5530}
    rows = []
    for path in (control, changed):
        doc = _read(path)
        seen = set()
        with centered(doc["ready"]):
            a = arena(protocol="random")
            try:
                for card, row in zip(doc["cards"], doc["rows"]):
                    if card["id"] not in test_cards or card["id"] in seen:
                        continue
                    seen.add(card["id"])
                    a.launch(row["shot"], card=card)
                    a.partner.reset(seed=card["seed"])
                    blade = a._racket_ids[1]
                    original = a._racket_contact
                    contact = []
                    for_step = []

                    def observe(side, point, v):
                        if side == 1 and not contact:
                            twist = np.empty(6)
                            mujoco.mj_objectVelocity(a.model, a.data, mujoco.mjtObj.mjOBJ_GEOM, blade, twist, 0)
                            contact.append(dict(tick=a.steps, time_s=a.data.time, ball_in_mps=list(v),
                                                center_m=a.paddle.tolist(), blade_normal=a.data.geom_xmat[blade].reshape(3, 3)[:, 2].tolist(),
                                                blade_spin_rps=twist[:3].tolist(), blade_speed_mps=twist[3:].tolist(),
                                                center_to_contact_m=(np.asarray(point) - a.paddle).tolist()))
                        return original(side, point, v)

                    with patch.object(a, "_racket_contact", observe):
                        for action in row["actions"]:
                            a.exchange(action, FAR_COMMAND)
                            if contact and len(for_step) < 4:
                                for_step.append(a.velocity.tolist())
                    if digest_json(a.events) != digest_json(row["events"]):
                        raise ValueError("Audit replay diverged")
                    rows.append(dict(condition=path.name, card=card, first_landed=row["score"]["learner"] > 0,
                                     own=row["score"]["learner"], first_contact_native=contact,
                                     four_postcontact_frames_mps=for_step))
            finally:
                a.close()
    _write(output, dict(teacher_only_diagnostic=True, sha256={p.name: _hash(p) for p in (control, changed)}, rows=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="chapter", required=True)
    fit = sub.add_parser("calibrate")
    fit.add_argument("--pose", type=Path, default=OUT / "pose.npz")
    match = sub.add_parser("physical")
    match.add_argument("output", type=Path)
    match.add_argument("--pose", type=Path, default=OUT / "pose.npz")
    match.add_argument("--bank", type=Path, default=BANK)
    match.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    match.add_argument("--deck", choices=("held_openings", "held_replies"), default="held_openings")
    a = sub.add_parser("audit")
    a.add_argument("control", type=Path)
    a.add_argument("changed", type=Path)
    a.add_argument("output", type=Path)
    args = p.parse_args()
    if args.chapter == "calibrate":
        pose_calibration(args.pose)
    elif args.chapter == "audit":
        audit_pair(args.control, args.changed, args.output)
    else:
        physical(args.bank, args.pose, args.feeds, args.deck, args.output)

"""Port a scored early return from practice to new camera-observed approaches."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_fast_camera import native_ball_pixels
from .tt35_error import triangulate
from .tt39_carry_joints import Carried
from .tt39_turn_time import turn_time

FOLDER = Path("artifacts/tt41")


def optical_impact(points, rough):
    """Fit two ballistic segments sharing a center at a camera-observed reversal."""
    near = [(t, p) for t, p in points if abs(t - rough) <= 11]
    if len(near) < 4:
        raise ValueError("Too few raw stereo samples around the reversal")
    ticks, measured = map(np.asarray, zip(*near, strict=True))
    candidates = []
    for hit in np.arange(rough - 7, rough + 4, 0.125):
        dt = (ticks - hit) * mpc.DT
        if np.sum(dt < 0) < 3 or np.sum(dt > 0) < 3:
            continue
        linear = np.c_[np.ones(len(dt)), np.minimum(dt, 0), np.maximum(dt, 0)]
        gravity = -0.5 * dt[:, None] ** 2 * np.array([0, 0, 9.81])
        coef = np.linalg.lstsq(linear, measured - gravity, rcond=None)[0]
        if coef[1, 1] >= -0.9 or coef[2, 1] <= 0.9:
            continue
        err = np.mean(np.sum((linear @ coef + gravity - measured) ** 2, axis=1))
        candidates.append((err, hit, coef))
    if not candidates:
        raise ValueError("Raw stereo did not resolve the reversal")
    err, hit, coef = min(candidates, key=lambda r: r[0])
    return float(hit), np.r_[coef[0], coef[1]], float(np.sqrt(err))


def demonstration(source, out, ident):
    doc = _read(source)
    item = next(r for r in doc["rows"] if r["id"] == ident)
    win = next(w for w in item["wins"] if w["answered"])
    home = np.asarray(doc["ready"])
    with mpc.centered(home):
        arena = mpc.arena()
        reader, rig, stride, low, high = mpc.resources(arena)
        try:
            arena.launch(win["shot"], card=item["card"])
            arena.partner.reset(seed=item["card"]["seed"])
            eye = mpc.Vision(rig, arena.control_dt, raw=True)
            eye.eye = mpc.LearnedOptics(eye.eye, mpc.ROOT / "tt37/rebound.npz")
            ref = home.copy()
            frames = []
            points = []
            for tick, u in enumerate(win["actions"]):
                q = arena.joints
                image = reader.capture() if tick % 2 == 0 else None
                x = mpc.observed(eye, tick, image, q)
                if image is not None:
                    detections = [
                        native_ball_pixels(image[c]) for c in ("left", "right")
                    ]
                    if all(v is not None and not v.clipped for v in detections):
                        xyz = triangulate([v.center for v in detections], rig)
                        if xyz is not None:
                            points.append((tick, xyz))
                ref = np.clip(ref + np.asarray(u) * stride, low, high)
                frames.append((tick, x, q.copy(), ref.copy()))
                arena.exchange(u, mpc.FAR_COMMAND)
            if digest_json(arena.events) != digest_json(win["events"]):
                raise ValueError("Practice replay changed")
        finally:
            reader.close()
            arena.close()
    used = frames[win["fork_tick"] :]
    hit = turn_time([(t, x) for t, x, *_ in used if x is None or x[1] < 0.55])
    if hit is None:
        raise ValueError("No observed reversal")
    try:
        hit, contact, rmse = optical_impact(points, hit)
    except ValueError:
        # With a temporarily occluded turn, the filtered direction can belong
        # to an older exchange. Locate a supported raw optical reversal within
        # this actually scored practice movement.
        alternatives = []
        for ti, pi in points:
            if ti < win["fork_tick"] or pi[1] > -0.75:
                continue
            before = [pt[1] for t, pt in points if ti - 8 <= t < ti]
            after = [pt[1] for t, pt in points if ti < t <= ti + 8]
            if (
                len(before) < 2
                or len(after) < 2
                or min(before) < pi[1]
                or min(after) < pi[1]
            ):
                continue
            try:
                alternatives.append(optical_impact(points, ti))
            except ValueError:
                continue
        if not alternatives:
            raise
        hit, contact, rmse = min(alternatives, key=lambda r: r[-1])
    if rmse > 0.025:
        raise ValueError("Contact fit exceeds the camera residual limit")
    span = frames[max(win["fork_tick"], round(hit) - 56) : round(hit) + 8]
    _write(
        out,
        {
            "id": item["id"],
            "input_sha256": _hash(source),
            "contact": contact.tolist(),
            "stereo_fit_rmse_m": rmse,
            "phase": [k - hit for k, *_ in span],
            "q": [q.tolist() for _, _, q, _ in span],
            "reference": [r.tolist() for _, _, _, r in span],
            "source": "Full legal and answered demonstration. Optical reversal, encoders and bounded commands only.",
        },
    )
    print("camera fit", hit, contact, flush=True)


class Rising(mpc.MPC):
    def __init__(self, bank, raw, *, variant, budget, prior, mode):
        super().__init__(bank, raw, variant=variant, budget=budget)
        self.phase = np.array(prior["phase"])
        self.refs = np.array(prior["reference"])
        self.contact = np.array(prior["contact"])
        self.mode = mode
        q = np.asarray(prior["q"])[int(np.argmin(abs(self.phase)))]
        _, _, jac, normal = self.geometry.pose(q[:5], jacobian=True)
        self.inverse = np.linalg.pinv(np.r_[jac, 0.085 * normal, 0.024 * np.eye(5)])
        self.stand = self.pose(-13.0, 0.0, self.contact[2], 0.0)

    def pose(self, phase, x, z, vx):
        goal = np.array(
            [np.interp(phase, self.phase, column) for column in self.refs.T]
        )
        delta = np.r_[
            x - self.contact[0],
            0.0,
            z - self.contact[2],
            -0.035 * (vx - self.contact[3]),
            0.0,
            0.0,
            np.zeros(5),
        ]
        goal[:5] += np.clip(self.inverse @ delta, -1.25, 1.25)
        return goal

    def begin(self):
        super().begin()
        self.last_cue = None
        self.active = self.mode == "always"

    def step(self, tick, eye, x, q, reference, base):
        self.engine.observe(tick, eye, q, reference.copy())
        now = tick * mpc.DT
        if x is not None and x[4] > 0.9:
            self.best = None
            self.last_cue = None
            self.active = self.mode == "always"
            return self.stand.copy() if self.active else 0.45 * self.stand + 0.55 * base
        with patch.object(joint, "RECEIVE_Y", self.contact[1]):
            cue = joint.flight_at_plane(eye.latest, now)
        if cue is not None and cue[0] > 0.07:
            with patch.object(joint, "RECEIVE_Y", -1.53):
                later = joint.flight_at_plane(eye.latest, now)
            if self.mode == "selective" and later is not None and later[0] > 0.19:
                self.active = bool(later[2] < 1.005)
            self.last_cue = (tick, cue)
        if not self.active:
            with patch.object(self.engine, "observe", lambda *args: None):
                return super().step(tick, eye, x, q, reference, base)
        if cue is None and self.last_cue:
            last, obs = self.last_cue
            if tick - last <= 9:
                cue = obs.copy()
                cue[0] -= (tick - last) * mpc.DT
        if cue is None:
            return self.stand.copy()
        eta, px, pz, vx, *_ = cue
        phase = -eta / mpc.DT
        desired = self.pose(np.clip(phase, -18, 8), px, pz, vx)
        if (
            x is not None
            and 0.06 < eta < 0.235
            and tick >= self.next
            and self.engine.sync()
        ):
            self.next = tick + 5
            ts = phase + np.arange(mpc.HORIZON)
            moves = [
                np.vstack([self.pose(u + s, px, pz, vx) for u in ts])
                for s in (0.0, -2.0, 2.0)
            ]
            moves.extend(s for s, _ in self.priors.priors(x, q, reference, 3))
            for i in range(24):
                moves.append(
                    mpc.structured(moves[i % 3], self.rng.normal(size=10), 0.86)
                )
            short, *_ = self.trial(moves)
            idx = np.argsort(short)[:8]
            loss, legal, *_ = self.trial([moves[j] for j in idx], long=True)
            if np.any(legal):
                candidates = np.flatnonzero(legal)[np.argsort(loss[legal])[:4]]
                chosen = idx[candidates]
                win, agree, _ = self.robust(
                    [moves[j] for j in chosen], chosen, loss[candidates]
                )
                if agree:
                    self.best, self.tick, self.lease = moves[win], tick, mpc.HORIZON
                    self.next = tick + mpc.HORIZON
                    self.diag["early_legal"] += 1
            self.diag["early_searches"] += 1
        if self.best is not None and 0 <= tick - self.tick < self.lease:
            return self.best[tick - self.tick]
        return desired


def run(a):
    prior = _read(a.prior)
    held = {r["id"] for r in _read(a.plan)[a.deck][: a.limit]}
    if prior["id"] in held:
        raise ValueError("Evaluation includes practiced release")

    def bank(file, engine):
        return Carried(
            file, engine, practice=[mpc.ROOT / "tt37/camera-time-school.npz"], held=held
        )

    def model(file, raw, **kw):
        return Rising(file, raw, **kw, prior=prior, mode=a.mode)

    with patch.object(mpc, "MPC", model), patch.object(mpc, "Fragments", bank):
        mpc.run(
            a.feeds,
            mpc.BANK,
            a.plan,
            a.deck,
            "live",
            a.out,
            optic=mpc.ROOT / "tt37/rebound.npz",
            limit=a.limit,
            variant="elastic",
            cap=240,
        )
    doc = _read(a.out)
    doc.update(motor_demonstration=_hash(a.prior), early_mode=a.mode)
    _write(a.out, doc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["extract", "always", "selective"])
    p.add_argument("out", type=Path)
    p.add_argument("--source", type=Path, default=mpc.ROOT / "tt40/long-low.json.gz")
    p.add_argument("--prior", type=Path, default=FOLDER / "early-example.json")
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--plan", type=Path, default=mpc.ROOT / "tt37/memory-course.json")
    p.add_argument("--deck", default="replication")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--id", type=int, default=14843)
    a = p.parse_args()
    if a.mode == "extract":
        demonstration(a.source, a.out, a.id)
    else:
        run(a)

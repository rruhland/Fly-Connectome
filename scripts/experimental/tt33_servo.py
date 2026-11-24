"""Distil rewarded motor experience into a continuous feedback field and refine it with rally reward."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import context
from .tt29_endurance import audit
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_partner_fast import NativeBallFlight
from .tt29_rally_physics import measured_launches
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
READY = ROOT / "tt31" / "left-ready.json"
DEMO = ROOT / "tt31" / "ready-staged-checkpoint.json.gz"


def features(x, z, phase):
    x = np.clip(x / 0.24, -1.6, 1.6)
    z = np.clip((z - 1.025) / 0.06, -2.0, 2.0)
    b = np.array([1.0, x, z, x * z])
    return np.r_[b, b * np.clip(phase / 0.11, -1.0, 1.0)]


def distill():
    new = _read(DEMO)["checkpoint"][223:]
    obs, target = [], []
    for e in new:
        x, z = e["flight"][:2]
        q = np.asarray(e["targets"])[:, :5]
        for i in range(len(q)):
            phase = (i + 1) / 60.0 - e["time_to_contact"]
            if -0.14 <= phase <= 0.12:
                obs.append(features(x, z, phase))
                target.append(q[i])
    a = np.stack(obs)
    weights = np.linalg.solve(
        a.T @ a + np.diag([0.005, 0.15, 4, 6.0, 0.3, 0.3, 4, 6.0]), a.T @ target
    )
    return {
        "weights": weights.tolist(),
        "training_legal_movements": len(new),
        "source_sha256": _hash(DEMO),
    }


class Station:
    def __init__(self, feeds, ready):
        self.shots = measured_launches(feeds)
        with centered(ready):
            self.arena = replay_arena([])
        self.renderer = CachedStereoRenderer(self.arena)
        self.rig = Rig.from_arena(self.arena)
        self.forecast = NativeBallFlight(self.arena)
        self.ready = ready

    def close(self):
        self.renderer.close()
        self.arena.close()

    def play(self, c, weights, offset, cap):
        a = self.arena
        with centered(self.ready):
            a.launch(self.shots[c["id"]], card=c)
        a.partner.reset(seed=c["seed"])
        self.rig.begin()
        eye = OpticalFlight(
            self.rig, FlightPractice(), learn=True, detector=native_ball_pixels
        )
        axes = a._arm_ids[1][2]
        increments = a._joint_increments[axes] * a.control_dt / a.model.opt.timestep
        u = np.zeros(6)
        sensed = None
        actions = []
        choices = []
        last_approach = None
        joint_offset = offset
        travel = 0.0
        max_speed = 0.0
        prev = a.joints.copy()
        for step in range(3600):
            t = step * a.control_dt
            if step % 2 == 0:
                eye.observe(t, self.renderer.capture(), a.joints)
                f = eye.latest
                if (
                    f is not None
                    and t - f.time < 0.105
                    and f.rms < 2.4
                    and f.depth_sigma < 0.16
                ):
                    if f.velocity[1] < -0.90:
                        p, v = f.advance(t)
                        path = self.forecast.path(
                            p, v, np.zeros(3), 1, bool(f.bounced and p[1] < 0.1)
                        )
                        match = context(path, -1.29)
                        if match is not None:
                            key, delay = match
                            sensed = key[:2], t + delay
                    elif (
                        f.velocity[1] > 1.05
                        and sensed is not None
                        and t - sensed[1] > 0.09
                    ):
                        sensed = None
            if sensed is not None and t - sensed[1] <= 0.18:
                (x, z), arrival = sensed
                if callable(offset):
                    if last_approach is None or arrival - last_approach > 0.6:
                        joint_offset, choice = offset(x, z)
                        choices.append(
                            {"time": t, "x": float(x), "z": float(z), "choice": choice}
                        )
                    last_approach = arrival
                goal = features(x, z, t - arrival) @ weights + joint_offset
                # Position feedback. No action sequences, IK, candidate simulation, or teacher queries.
                error = goal - a.joints[:5]
                raw = np.r_[np.clip(error / (increments[:5] * 4.0), -0.76, 0.76), 0.0]
            else:
                raw = np.zeros(6)
            u += np.clip(raw - u, -0.34, 0.34)
            _, _, end, done, _ = a.exchange(u, np.zeros(5))
            actions.append(u.tolist())
            delta = abs(a.joints - prev)
            travel += float(np.sum(delta[:5]))
            max_speed = max(max_speed, float(np.max(delta[:5] / a.control_dt)))
            prev = a.joints.copy()
            if end or done or a.legal_returns + a.rival_returns - 1 >= cap:
                break
        ref = audit(a.events, opening_side=1)
        return {
            "shot": self.shots[c["id"]],
            "opponent_seed": c["seed"],
            "score": {k: v for k, v in ref.items() if k != "shots"},
            "placements": a.point_placements(),
            "actions": actions,
            "events": a.events.copy(),
            "joint_path_rad": travel,
            "max_encoder_step_speed": max_speed,
            "motor": a.motor_audit(),
            "choices": choices,
        }


def run(args):
    base = distill()
    weights = np.array(base["weights"])
    source = _read(args.course)
    cards = source.get("cards", source.get("staged"))
    if args.training:
        # Two stable sides then genuinely varied balls; keep the independent full tests untouched.
        cards = [cards[i] for i in (4, 5, 10, 11, 12, 13, 14, 15)]
    ready = _read(READY)["ready"]
    station = Station(args.feeds, ready)
    rng = np.random.default_rng(331207)
    if args.mode == "empty":
        weights *= 0.0
        weights[0] = ready[:5]
    offset = np.zeros(5)
    if args.carry is not None:
        offset = np.array(_read(args.carry)["offset"])
    result = {
        "opening_side": 1,
        "ready": ready,
        "mode": args.mode,
        "distilled": base,
        "course_sha256": _hash(args.course),
        "cards": [],
        "rows": [],
        "updates": [],
    }
    try:
        for i, c in enumerate(cards):
            if args.mode == "adapt":
                # One paired perturbation, scored on an actual subsequent game with the same opening.
                # Only observed legal-return reward determines the update.
                delta = rng.choice([-1.0, 1.0], 5) * 0.045
                candidates = [offset, offset + delta, offset - delta]
            else:
                candidates = [offset]
            points = [
                station.play(c, weights, op, cap=8 if args.training else 16)
                for op in candidates
            ]
            own = [row["score"]["learner"] for row in points]
            winner = int(np.argmax(own))
            offset = candidates[winner].copy()
            result["updates"].append({"scores": own, "offset": offset.tolist()})
            result["cards"].append(c)
            result["rows"].append(points[winner])
            result["offset"] = offset.tolist()
            _write(args.output, result)
            print(i + 1, c["id"], own, flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("feeds", "course", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--mode", choices=("teacher", "adapt", "empty"), default="teacher")
    p.add_argument("--training", action="store_true")
    p.add_argument("--carry", type=Path)
    run(p.parse_args())

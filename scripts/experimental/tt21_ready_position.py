"""Learn a lateral waiting position from the first outgoing arc and the next incoming arc."""

import argparse

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_shoot import _normals
from .tt20_curriculum import OUTPUT as SOURCE
from .tt20_curriculum import receiver
from .tt21_body_state import StateInitializedDrySimulator
from .tt21_recovery_trial import (
    ARENA,
    OPENING,
    OUTPUT,
    Station,
    observed,
    practice_starts,
    report,
    verify,
)

PRACTICE = OUTPUT / "recovery-practice.json.gz"
HELD_SEED = 110241987


def outbound_features(history):
    if len(history) < 3:
        return None
    samples = history[-3:]
    ticks, pos = zip(*samples)
    dt = ticks[-1] - ticks[0]
    if dt > 4 or ticks[-1] < 25:
        return None
    p = np.asarray(pos)
    if not 0.27 < p[-1, 1] < 0.76 or min(np.diff(p[:, 1])) < 0.020:
        return None
    # Extrapolate a common reference on the outgoing path from observed pixels.
    vx = (p[-1, 0] - p[0, 0]) / dt
    far_x = p[-1, 0] + (0.72 - p[-1, 1]) * (p[-1, 0] - p[0, 0]) / (p[-1, 1] - p[0, 1])
    return np.array([1.0, far_x, vx * 20.0])


def inbound_features(history):
    if len(history) < 3:
        return None
    ticks, pos = zip(*history[-3:])
    p = np.asarray(pos)
    if ticks[-1] - ticks[0] > 4 or not 0.05 < p[-1, 1] < 0.68 or max(np.diff(p[:, 1])) > -0.025:
        return None
    direction = (p[-1, 0] - p[0, 0]) / (p[-1, 1] - p[0, 1])
    x_at_net = p[-1, 0] - p[-1, 1] * direction
    return np.array([1.0, x_at_net, -direction])


def training_pairs(cue_mode="outgoing"):
    pairs = []
    for row in _read(PRACTICE)["training"]:
        reads = [(t, p) for t, p, _ in row["optical"] if p is not None]
        cue = None
        outbound = False
        for i, (t, p) in enumerate(reads):
            if cue is None:
                outward = outbound_features(reads[: i + 1])
                outbound |= outward is not None
                if outbound:
                    cue = outward if cue_mode == "outgoing" else inbound_features(reads[: i + 1])
            elif i > 2 and -0.69 < p[1] < -0.42:
                dt = t - reads[i - 2][0]
                delta = np.asarray(p) - reads[i - 2][1]
                if dt <= 4 and delta[1] / dt < -0.03:
                    later_x = p[0] + (-0.88 - p[1]) * delta[0] / delta[1]
                    pairs.append({"seed": row["seed"], "cue": cue.tolist(), "next_x": later_x})
                    break
    return pairs


def fit(pairs):
    x = np.array([p["cue"] for p in pairs])
    y = np.array([p["next_x"] for p in pairs])
    return np.linalg.solve(x.T @ x + np.diag([0.002, 0.003, 0.003]), x.T @ y)


def rotated_poses(poses, pivot, angle):
    result = poses.copy()
    c, s = np.cos(angle), np.sin(angle)
    xy = poses[..., :2] - pivot
    result[..., 0] = c * xy[..., 0] - s * xy[..., 1] + pivot[0]
    result[..., 1] = s * xy[..., 0] + c * xy[..., 1] + pivot[1]
    normals = _normals(poses)
    sign = np.where(s * normals[..., 0] + c * normals[..., 1] >= 0.0, 1.0, -1.0)
    result[..., 3] = (c * normals[..., 0] - s * normals[..., 1]) * sign
    result[..., 4] = normals[..., 2] * sign
    return result


def shoulder_goal(lane, radians_per_command):
    desired_angle = np.clip(lane / 0.24, -0.98, 0.98) if abs(lane) > 0.105 else 0.0
    return float(desired_angle / radians_per_command)


class PositionStation(Station):
    def __init__(self):
        super().__init__()
        self.model.player.close()
        self.model = receiver(False, SOURCE)
        self.stroke = self.model.player
        self.stroke.sim.close()
        self.model_sim = self.stroke.sim = StateInitializedDrySimulator(self.stroke.law)
        self.catalog = self.stroke.poses
        joint = self.model_sim.model.joint("learner_shoulder_pan").id
        self.pivot = self.model_sim.twin.data.xanchor[joint, :2].copy()
        self.axis_z = self.model_sim.twin.data.xaxis[joint, 2]
        self.radians = self.model_sim.increments[0] * self.model_sim.timestep

    def point(self, seed, role, coef, prefix=(), cue_mode="outgoing"):
        arena = self.arena
        arena.reset(seed=seed, server=role, render=False)
        self.model.reset(role)
        self.model.serves.force = OPENING
        self.model_sim.twin.reset()
        self.stroke.poses = self.catalog
        self.model_sim.current = None
        tracker = self.new_scene().tracker()
        actions, reads, optics = [], [], []
        pan = 0.0
        decision = None
        outbound = False
        for tick in range(arena.max_steps):
            ball = None
            if not prefix or tick >= 23 or tick in (0, 3):
                tracker.update(arena.render_cameras(), tick=tick)
                ball = observed(tracker.ball)
                if ball is not None:
                    reads.append((tick, ball.tolist()))
                    optics.append((tick, ball.tolist()))
            if prefix and tick == len(prefix):
                self.model.phase = "receive"
                self.stroke.movement = np.sum(np.asarray(prefix), axis=0)[:5]
                self.stroke.history.extend(reads[:-1])
            opened = bool(prefix) or bool(self.stroke.plans) or self.model.actor is not None
            if opened and decision is None and ball is not None and tick >= len(prefix):
                outward = outbound_features(reads)
                outbound |= outward is not None
                cue = outward if cue_mode == "outgoing" else inbound_features(reads)
                if outbound and cue is not None:
                    lane = float(cue @ coef)
                    goal = shoulder_goal(lane, self.radians)
                    decision = {
                        "tick": tick,
                        "cue": cue.tolist(),
                        "lane": lane,
                        "goal": goal,
                    }
            if pan:
                self.model_sim.current = self.model_sim.twin.state()
            if tick < len(prefix):
                command = np.array(prefix[tick])
            else:
                command = self.model.command(tick, ball)
            if decision is not None and self.model.phase == "receive":
                delta = np.clip(decision["goal"] - pan, -1.0 - command[0], 1.0 - command[0])
                if abs(delta) > 1e-8:
                    command = command.copy()
                    command[0] += delta
                    pan += delta
                    self.stroke.poses = rotated_poses(
                        self.catalog, self.pivot, self.axis_z * pan * self.radians
                    )
            actions.append(command.tolist())
            self.model_sim.twin.advance(command)
            _, reward, done, cap, _ = arena.step(command, render=False)
            if done or cap:
                break
        return {
            "seed": seed,
            "role": role,
            "decision": decision,
            "reward": reward,
            "actions": actions,
            "optical": optics,
            "approach": None,
            "plans": self.stroke.plans,
            "outcome": arena.point_result(),
            "events": arena.events.copy(),
            "rgb": tracker.digest.hexdigest(),
        }


def ready_metrics(rows):
    metrics = report(rows)
    metrics.pop("optical_approaches")
    decisions = [r["decision"] for r in rows if r["decision"] is not None]
    metrics.update(
        predicted_lanes=len(decisions), commanded_shifts=sum(bool(abs(d["goal"]) > 1e-8) for d in decisions)
    )
    return metrics


def run(development, count, cue_mode):
    pairs = training_pairs(cue_mode)
    coef = fit(pairs)
    leave_one_out = [
        float(fit(pairs[:i] + pairs[i + 1 :]) @ p["cue"] - p["next_x"]) for i, p in enumerate(pairs)
    ]
    stage = "development" if development else "held"
    path = (
        OUTPUT / f"ready-position-{stage}.json.gz"
        if cue_mode == "outgoing"
        else OUTPUT / f"ready-position-{cue_mode}-{stage}.json.gz"
    )
    result = {
        "source": PRACTICE.name,
        "source_sha256": _hash(PRACTICE),
        "observations": pairs,
        "coefficients": coef.tolist(),
        "leave_one_out": leave_one_out,
        "arena": ARENA,
        "cue_mode": cue_mode,
        "conditions": {},
    }
    if development:
        starts = practice_starts()[:count]
    else:
        seeds = np.random.default_rng(HELD_SEED).integers(2**31, size=count).tolist()
        starts = [{"seed": s, "role": (-1, 1)[i % 2], "prefix": []} for i, s in enumerate(seeds)]
    station = PositionStation()
    try:
        for name, coeff in [("shift", coef), ("center", np.zeros_like(coef))]:
            rows = []
            for index, start in enumerate(starts):
                if name == "center":
                    shifted = result["conditions"]["shift"]["points"][index]
                    choice = shifted["decision"]
                    if choice is None or abs(choice["goal"]) < 1e-8:
                        center = None if choice is None else dict(choice, lane=0.0, goal=0.0)
                        rows.append(dict(shifted, decision=center, identical_zero_shift=True))
                        continue
                rows.append(station.point(start["seed"], start["role"], coeff, start["prefix"], cue_mode))
            result["conditions"][name] = {
                "points": rows,
                "summary": ready_metrics(rows),
                "verified": verify(rows),
            }
            _write(path, result)
            print(name, ready_metrics(rows), flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--cue", choices=("outgoing", "incoming"), default="outgoing")
    args = parser.parse_args()
    run(args.development, args.count, args.cue)

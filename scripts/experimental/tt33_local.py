"""Learn a local joint feedback law from RGB motion and encoders throughout complete demonstrations."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_fast_camera import CachedStereoRenderer, native_ball_pixels
from .tt29_rally_physics import measured_launches
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt31_curriculum import card
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
SOURCES = (
    "ready-practice-staged",
    "confirm-recovery",
    "transition-practice",
    "transition-6810-reference",
)


class Signals:
    def __init__(self, rig):
        rig.begin()
        self.eye = OpticalFlight(
            rig, FlightPractice(), learn=True, detector=native_ball_pixels
        )

    def get(self, step, arena, renderer, q, dq):
        now = step * arena.control_dt
        if step % 2 == 0:
            self.eye.observe(now, renderer.capture(), q)
        f = self.eye.latest
        if f is None or now - f.time > 0.12 or f.depth_sigma > 0.20 or f.rms > 3.0:
            return None
        position, velocity = f.advance(now)
        return np.r_[position, velocity, q[:5], dq[:5]]


def capture(output):
    samples, labels = [], []
    records = []
    seen = set()
    for n in SOURCES:
        source = ROOT / "tt31" / (n + ".json.gz")
        d = _read(source)
        with centered(d["ready"]):
            a = replay_arena([])
            renderer = CachedStereoRenderer(a)
            rig = Rig.from_arena(a)
            try:
                for c, r in zip(d["cards"], d["rows"], strict=True):
                    key = (c["id"], c["seed"])
                    if not r["score"]["learner"] or key in seen:
                        continue
                    seen.add(key)
                    a.launch(r["shot"], card=c)
                    a.partner.reset(seed=c["seed"])
                    prev = a.joints.copy()
                    sense = Signals(rig)
                    axes = a._arm_ids[1][2]
                    sequence = []
                    for i, u in enumerate(r["actions"]):
                        q = a.joints.copy()
                        x = sense.get(i, a, renderer, q, (q - prev) / a.control_dt)
                        a.exchange(u, np.zeros(5))
                        sequence.append((x, a.data.ctrl[axes][:5].copy()))
                        prev = q
                    if digest_json(a.events) != digest_json(r["events"]):
                        raise ValueError("Demonstration replay changed")
                    for i, (x, _) in enumerate(sequence[:-3]):
                        if x is not None:
                            samples.append(x)
                            labels.append(sequence[i + 3][1])
                    records.append(
                        {
                            "id": c["id"],
                            "seed": c["seed"],
                            "source": _hash(source),
                            "n": len(sequence),
                        }
                    )
            finally:
                renderer.close()
                a.close()
    x = np.stack(samples)
    scale = np.maximum(x.std(axis=0), 0.01)
    # Put most distance weight on visible motion, followed by posture and joint velocity.
    scale *= np.r_[np.ones(6), np.full(5, 1.8), np.full(5, 4.0)]
    np.savez_compressed(output, x=x, targets=labels, scale=scale)
    _write(
        output.with_suffix(".json"),
        {"records": records, "samples": len(x), "model_sha256": _hash(output)},
    )


class Local:
    def __init__(self, model):
        self.x = model["x"] / model["scale"]
        self.q = model["targets"]
        self.tree = cKDTree(self.x)
        self.scale = model["scale"]

    def get(self, observed):
        x = observed / self.scale
        dist, idx = self.tree.query(x, 24)
        dx = self.x[idx] - x
        w = np.exp(-0.5 * (dist / max(0.15, dist[8])) ** 2)
        w /= max(1e-12, w.sum())
        q = w @ self.q[idx]
        center = w @ dx
        cov = dx - center
        coef = np.linalg.solve(
            cov.T @ (w[:, None] * cov) + 0.08 * np.eye(len(x)),
            cov.T @ (w[:, None] * (self.q[idx] - q)),
        )
        return q - np.clip(center @ coef, -0.22, 0.22)


def evaluate(feeds, model, plan, output, *, practice=False, checkpoint=None):
    with np.load(model) as archive:
        data = dict(archive)
    trained = Local(data)
    rng = np.random.default_rng(336711)
    if practice:
        cards = [
            card(
                (12330, 12451, 11909, 13140)[i % 4],
                ("periodic", "full", "continuous")[(i // 4 + i) % 3],
                i,
                336710 + i,
            )
            for i in range(48)
        ]
    else:
        cards = _read(plan)["cards"]
    ready = _read(ROOT / "tt31" / "left-ready.json")["ready"]
    shots = measured_launches(feeds)
    history = _read(model.with_suffix(".json"))
    if not practice and {r["id"] for r in history["records"]} & {
        c["id"] for c in cards
    }:
        raise ValueError("Training and evaluation have a shared release")
    result = {
        "ready": ready,
        "cards": cards,
        "opening_side": 1,
        "model_sha256": _hash(model),
        "plan_sha256": None if practice else _hash(plan),
        "rows": [],
        "practice": practice,
    }
    with centered(ready):
        arena = replay_arena([])
        renderer = CachedStereoRenderer(arena)
        rig = Rig.from_arena(arena)
        try:
            for number, c in enumerate(cards):
                bias = (
                    0.055 * rng.choice([-1.0, 1.0], 5)
                    if practice and number % 3
                    else np.zeros(5)
                )
                arena.launch(shots[c["id"]], card=c)
                arena.partner.reset(seed=c["seed"])
                sense = Signals(rig)
                axes = arena._arm_ids[1][2]
                stride = (
                    arena._joint_increments[axes]
                    * arena.control_dt
                    / arena.model.opt.timestep
                )
                prev = arena.joints.copy()
                cmd = arena.data.ctrl[axes].copy()
                actions = []
                eligible = []
                opening_credit = True
                learned = 0
                for i in range(3600):
                    q = arena.joints.copy()
                    x = sense.get(i, arena, renderer, q, (q - prev) / arena.control_dt)
                    u = np.zeros(6)
                    if x is not None:
                        u[:5] = np.clip(
                            (trained.get(x) + bias - cmd[:5]) / (3.0 * stride[:5]),
                            -0.8,
                            0.8,
                        )
                    if practice and x is not None:
                        eligible.append((i, x.copy(), trained.get(x) + bias))
                    _, _, end, cap, info = arena.exchange(u, np.zeros(5))
                    if practice:
                        eligible = [item for item in eligible if i - item[0] < 60]
                        for event in info["events"]:
                            if event["kind"] != "legal_return" or event["side"] != 1:
                                continue
                            if opening_credit:
                                opening_credit = False
                            elif eligible:
                                data["x"] = np.vstack(
                                    (data["x"], [item[1] for item in eligible])
                                )
                                data["targets"] = np.vstack(
                                    (data["targets"], [item[2] for item in eligible])
                                )
                                learned += len(eligible)
                                trained = Local(data)
                            eligible = []
                    actions.append(u.tolist())
                    cmd = np.clip(
                        cmd + u * stride,
                        arena._joint_low[axes],
                        arena._joint_high[axes],
                    )
                    prev = q
                    if (
                        end
                        or cap
                        or arena.legal_returns + arena.rival_returns - 1 >= 16
                    ):
                        break
                score = audit(arena.events, opening_side=1)
                row = {
                    "shot": shots[c["id"]],
                    "opponent_seed": c["seed"],
                    "actions": actions,
                    "events": arena.events.copy(),
                    "learned_sensor_actions": learned,
                    "bias": bias.tolist(),
                    "placements": arena.point_placements(),
                    "score": {k: v for k, v in score.items() if k != "shots"},
                }
                result["rows"].append(row)
                _write(output, result)
                print(c["id"], score["learner"], flush=True)
        finally:
            renderer.close()
            arena.close()
    if practice:
        np.savez_compressed(checkpoint, **data)
        _write(
            checkpoint.with_suffix(".json"),
            {
                "records": history["records"]
                + [{"id": c["id"], "seed": c["seed"]} for c in cards],
                "samples": len(data["x"]),
                "model_sha256": _hash(checkpoint),
                "parent_sha256": _hash(model),
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "play", "practice"))
    p.add_argument("model", type=Path)
    p.add_argument("--feeds", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    if a.mode == "capture":
        capture(a.model)
    else:
        evaluate(
            a.feeds,
            a.model,
            a.plan,
            a.output,
            practice=a.mode == "practice",
            checkpoint=a.checkpoint,
        )

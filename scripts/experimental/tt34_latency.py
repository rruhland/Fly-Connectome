"""Learn camera timing from observed arm motion and encoder readings on missed shots."""

import argparse
from collections import deque
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_source_rally import blade_sighting
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources

ROOT = Path(__file__).resolve().parents[2] / "artifacts"
PUPIL = ROOT / "tt33/joint-raw.json.gz"
SOURCE = ROOT / "tt33/local-stream.json.gz"
OUT = ROOT / "tt34"


def pixels(frames):
    centers = [blade_sighting(frames[k]) for k in ("left", "right")]
    if any(c is None for c in centers):
        return None
    return np.concatenate([c[0] * frames["left"].shape[0] / 360 for c in centers])


def capture(output):
    src = _read(SOURCE)
    poses, centers, flights, pairs, games = [], [], [], [], []
    with centered(src["ready"]):
        arena = replay_arena([])
        stereo, rig, *_ = resources(arena)
        try:
            for c, r in zip(src["cards"], src["rows"], strict=True):
                if r["score"]["learner"]:
                    continue
                arena.launch(r["shot"], card=c)
                arena.partner.reset(seed=c["seed"])
                eye = Vision(rig, arena.control_dt, raw=True)
                optical = []
                for i, u in enumerate(r["actions"]):
                    if i % 2 == 0:
                        frame = stereo.capture()
                        xy = pixels(frame)
                        q = arena.joints.copy()
                        if xy is not None:
                            poses.append(q[:5])
                            centers.append(xy)
                        optical.append(eye.update(i, frame, q))
                    arena.exchange(u, FAR_COMMAND)
                if digest_json(arena.events) != digest_json(r["events"]):
                    raise ValueError("Physical source replay changed")
                for j, x in enumerate(optical[:-5]):
                    future = optical[j : j + 6]
                    if x is not None and all(f is not None for f in future):
                        flights.append(x)
                        pairs.append(future)
                games.append([c["id"], c["seed"]])
                if len(games) == 20:
                    break
        finally:
            stereo.close()
            arena.close()
    np.savez_compressed(output, q=poses, xy=centers, flight=flights, next=pairs)
    _write(
        output.with_suffix(".json"),
        {
            "games": games,
            "samples": len(poses),
            "flight_samples": len(flights),
            "source_sha256": _hash(SOURCE),
            "model_sha256": _hash(output),
        },
    )


class Map:
    """Fit a local weighted ridge regression around each observed query."""

    def __init__(self, x, y, scale):
        self.x = x / scale
        self.y = y
        self.scale = scale
        self.tree = cKDTree(self.x)

    def __call__(self, values):
        x = values / self.scale
        dist, i = self.tree.query(x, 24)
        dx = self.x[i] - x
        w = np.exp(-0.5 * (dist / max(0.05, dist[8])) ** 2)
        w /= max(1e-12, w.sum())
        center = w @ dx
        cov = dx - center
        ybar = w @ self.y[i]
        beta = np.linalg.solve(
            cov.T @ (w[:, None] * cov) + 0.01 * np.eye(dx.shape[1]),
            cov.T @ (w[:, None] * (self.y[i] - ybar)),
        )
        return ybar - center @ beta


class Clock:
    """Infer RGB delivery age in camera frames from visible and encoder motion."""

    def __init__(self, bank):
        self.body = Map(bank["q"], bank["xy"], np.maximum(bank["q"].std(axis=0), 0.18))
        self.hist = deque(maxlen=8)
        self.last = None
        self.losses = deque(maxlen=18)
        self.lag = 0
        self.rows = []

    def begin(self):
        self.hist.clear()
        self.last = None
        self.losses.clear()

    def update(self, q, frame):
        self.hist.append(self.body(q[:5]))
        sight = pixels(frame) if frame is not None else None
        previous = self.last
        self.last = sight
        if previous is None or sight is None or len(self.hist) < 7:
            return
        delta = sight - previous
        hist = np.asarray(self.hist)
        expected = np.array([hist[-1 - j] - hist[-2 - j] for j in range(6)])
        if max(np.max(abs(delta)), np.max(abs(expected))) < 0.65:
            return
        self.losses.append(np.linalg.norm(expected - delta, axis=1).clip(0, 18))
        if len(self.losses) >= 6:
            total = np.mean(self.losses, axis=0)
            estimate = int(np.argmin(total))
            if np.ptp(total) > 0.35:
                self.lag = estimate
                self.rows.append([estimate, len(self.losses), total.tolist()])


def run(args):
    model = _read(PUPIL)
    src = _read(args.course)
    archive = np.load(args.bank)
    clock = Clock(archive)
    train = archive["next"] - archive["flight"][:, None, :]
    forward = Map(
        archive["flight"],
        train.reshape(len(train), -1),
        np.maximum(archive["flight"].std(axis=0), 0.05),
    )
    if args.state is not None:
        fitted = _read(args.state)
        if fitted["bank_sha256"] != _hash(args.bank) or fitted["pupil_sha256"] != _hash(
            PUPIL
        ):
            raise ValueError(
                "Calibration and evaluation must use the same observer and joint pupil"
            )
        clock.lag = fitted["lag"]
        if clock.lag not in range(6):
            raise ValueError("Saved age exceeds the observation history")
    if args.mode == "carry" and args.state is None:
        raise ValueError("Carried evaluation requires a saved fit")
    cards = src["cards"]
    if args.limit:
        cards = cards[: args.limit]
    schedule = [(c, delay) for delay in (args.blocks or [args.frames]) for c in cards]
    shots = measured_launches(args.feeds)
    result = {
        "cards": [c for c, _ in schedule],
        "schedule": [delay for _, delay in schedule],
        "ready": model["ready"],
        "opening_side": 1,
        "mode": args.mode,
        "image_frames": args.frames,
        "initial_lag": clock.lag,
        "fit_sha256": _hash(args.state) if args.state else None,
        "observation_model": args.predictor,
        "pupil_sha256": _hash(PUPIL),
        "bank_sha256": _hash(args.bank),
        "rows": [],
    }
    with centered(model["ready"]):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for c, delivery_lag in schedule:
                arena.launch(shots[c["id"]], card=c)
                arena.partner.reset(seed=c["seed"])
                policy = JointPolicy(model, stride, lo, hi)
                vision = Vision(rig, arena.control_dt, raw=True)
                clock.begin()
                first = len(clock.rows)
                frames = deque()
                actions = []
                lags = []
                start = process_time()
                for i in range(3600):
                    q = arena.joints.copy()
                    image = None
                    if i % 2 == 0:
                        frames.append(renderer.capture())
                        if len(frames) > delivery_lag:
                            image = frames.popleft()
                        if args.mode == "online":
                            clock.update(q, image)
                    # The learner only receives delivery time and RGB, never the capture timestamp.
                    optical = vision.update(i, image, q)
                    lag = clock.lag if args.mode != "oracle" else delivery_lag
                    lags.append(lag)
                    if lag and optical is not None:
                        if args.predictor == "linear":
                            optical = np.r_[
                                optical[:3] + optical[3:] * lag / 30.0, optical[3:]
                            ]
                        elif args.predictor == "kinematic":
                            optical = np.r_[
                                *vision.eye.latest.advance(
                                    i * arena.control_dt + lag / 30.0
                                )
                            ]
                        else:
                            optical = optical + forward(optical).reshape(6, 6)[lag]
                    u = policy.command(optical, q)
                    _, _, done, capped, _ = arena.exchange(u, FAR_COMMAND)
                    actions.append(u.tolist())
                    if (
                        done
                        or capped
                        or arena.legal_returns + arena.rival_returns - 1 >= 16
                    ):
                        break
                stats = audit(arena.events, opening_side=1)
                row = {
                    "shot": shots[c["id"]],
                    "opponent_seed": c["seed"],
                    "score": stats,
                    "actions": actions,
                    "events": arena.events.copy(),
                    "placements": arena.point_placements(),
                    "lags": lags,
                    "estimates": clock.rows[first:],
                    "runtime_cpu_s": process_time() - start,
                }
                result["rows"].append(row)
                result["lag"] = clock.lag
                _write(args.output, result)
                print(c["id"], stats["learner"], clock.lag, flush=True)
        finally:
            renderer.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("capture", "frozen", "online", "carry", "oracle"))
    p.add_argument("--bank", type=Path, default=OUT / "lag-bank.npz")
    p.add_argument("--feeds", type=Path)
    p.add_argument("--course", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--state", type=Path)
    p.add_argument("--frames", type=int, default=3, choices=range(6))
    p.add_argument("--blocks", type=int, nargs="+", choices=range(6))
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--predictor", choices=("learned", "kinematic", "linear"), default="learned"
    )
    a = p.parse_args()
    if a.mode == "capture":
        capture(a.bank)
    else:
        run(a)

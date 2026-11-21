"""Learn a direct fixed-camera SO-101 policy from scored RGB and joint-rate demonstrations."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt22_rally_trial import RallyArena, opponent


def color_table():
    values = np.indices((32, 32, 32)).transpose(1, 2, 3, 0).reshape(-1, 3)
    high, low = values.max(axis=1), values.min(axis=1)
    r, g, b = values.T
    hue = np.arctan2(np.sqrt(3.0) * (g - b), 2 * r - g - b)
    groups = (np.rint(hue * 6 / np.pi).astype(int) + 12) % 12
    return np.where(high - low >= np.maximum(3, high * 0.22), groups, 12 + (high > 14))


COLORS = color_table()
COUNT = 14


class VisualFeatures:
    """Measure every color group using the same area, centroid and spread statistics."""

    def __init__(self):
        self.first = None
        self.previous = None

    def observe(self, rgb):
        if self.first is None:
            self.first = rgb.copy()
        yy, xx = np.nonzero(np.max(np.abs(rgb.astype(np.int16) - self.first), axis=2) > 28)
        r, g, b = (rgb[yy, xx].astype(int) // 8).T
        label = COLORS[r * 1024 + g * 32 + b]
        area = np.bincount(label, minlength=COUNT)
        x = xx / rgb.shape[1] - 0.5
        y = yy / rgb.shape[0] - 0.5

        def mean(values):
            return np.bincount(label, weights=values, minlength=COUNT) / np.maximum(1, area)

        position = np.stack([mean(x), mean(y)], axis=-1)
        moments = np.column_stack([np.log1p(area) / 8, position, mean(x * x), mean(y * y)])
        old = moments if self.previous is None else self.previous
        self.previous = moments
        return np.column_stack((moments, moments - old)).ravel()


def own_features(target, action):
    return np.r_[target[:5], action[:5]]


def extract(source, output):
    data = _read(source)
    if data["privileged"]:
        raise ValueError("expected RGB teacher data")
    arena = RallyArena(data["spread"], data["pace"])
    opponent(arena, data["rival"])
    vision, body, delta, points = [], [], [], []
    try:
        for row in data["rows"]:
            arena.reset(seed=row["seed"], server=-1, render=False)
            featurizer = VisualFeatures()
            actions = np.array(row["actions"])
            targets = np.vstack((np.zeros(6), np.cumsum(actions, axis=0)))
            for t, act in enumerate(actions):
                vision.append(featurizer.observe(arena.render_cameras()["left"]))
                body.append(own_features(targets[t], actions[t - 1] if t else np.zeros(6)))
                delta.append((targets[min(t + 3, len(actions))] - targets[t])[:5])
                points.append(row["seed"])
                arena.step(act, render=False)
            if row["outcome"] != arena.point_result():
                raise ValueError("teacher replay changed")
            print("demonstration", row["seed"], len(actions), flush=True)
    finally:
        arena.close()
    np.savez_compressed(output, vision=vision, body=body, delta=delta, points=points)


class Apprentice:
    def __init__(self, dataset, points=None):
        selected = (
            np.ones(len(dataset["points"]), bool) if points is None else np.isin(dataset["points"], points)
        )
        v = dataset["vision"][selected]
        self.body_scale = np.maximum(0.35, dataset["body"][selected].std(axis=0))
        self.b = dataset["body"][selected] / self.body_scale
        self.vscale = np.maximum(0.012, v.std(axis=0))
        self.v = v / self.vscale
        self.y = dataset["delta"][selected]
        # Tune the contribution of each interchangeable color bin from motor prediction.
        self.weight = np.ones(COUNT)
        point_ids = dataset["points"][selected]
        errors = []
        for ident in np.unique(point_ids):
            train = point_ids != ident
            if not train.any():
                continue
            sq = []
            for t in np.flatnonzero(~train)[::4]:
                db = np.mean((self.b[train] - self.b[t]) ** 2, axis=1)
                dv = np.mean((self.v[train] - self.v[t]).reshape(-1, COUNT, 10) ** 2, axis=2)
                idx = np.argmin(db[:, None] + dv, axis=0)
                sq.append(np.mean((self.y[train][idx] - self.y[t]) ** 2, axis=1))
            errors.append(np.mean(sq, axis=0))
        if errors:
            error = np.mean(errors, axis=0)
            # Reward information that improves local motor prediction over ignoring the camera.
            self.weight = np.exp(-8 * (error - error.min()) / max(1e-6, np.mean(error)))
        self.weight /= sum(self.weight)

    def action(self, vision, body):
        physical = np.mean((self.b - body / self.body_scale) ** 2, axis=1)
        visual = np.mean((self.v - vision / self.vscale).reshape(-1, COUNT, 10) ** 2, axis=2)
        distance = physical + 0.9 * (visual @ self.weight)
        idx = np.argsort(distance)[:5]
        w = 1 / np.maximum(0.005, distance[idx]) ** 2
        delta = np.average(self.y[idx], weights=w, axis=0)
        return np.r_[np.clip(delta / 2.0, -1.0, 1.0), 0.0]


def play(dataset, output, seed, count):
    policy = Apprentice(np.load(dataset))
    arena = RallyArena(0.30, 0.82)
    opponent(arena, "anchor")
    rows = []
    try:
        for ident in np.random.default_rng(seed).integers(2**31, size=count):
            arena.reset(seed=int(ident), server=-1, render=False)
            featurizer = VisualFeatures()
            target = np.zeros(6)
            action = np.zeros(6)
            commands = []
            for tick in range(arena.max_steps):
                visual = featurizer.observe(arena.render_cameras()["left"])
                action = policy.action(visual, own_features(target, action))
                target += action
                commands.append(action.tolist())
                _, _, done, capped, _ = arena.step(action, render=False)
                if done or capped:
                    break
            rows.append(
                dict(
                    seed=int(ident), result=arena.point_result(), events=arena.events.copy(), actions=commands
                )
            )
            print(int(ident), rows[-1]["result"], flush=True)
            _write(output, dict(rows=rows, seed=seed, weights=policy.weight.tolist()))
    finally:
        arena.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("extract", "play"))
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--seed", type=int, default=273146)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args()
    if args.mode == "extract":
        extract(args.source, args.destination)
    else:
        play(args.source, args.destination, args.seed, args.count)

"""Learn which independently simulated motor outcomes turn into actual legal exchanges."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write


def features(predicted):
    r = np.atleast_2d(predicted)
    # Only outcomes of the learner's *own nominal* simulations enter this feature.
    # The selection can inspect margin on both sides of net, table and paddle.
    return np.column_stack(
        (
            r[:, 3],
            r[:, 4] - 0.65,
            r[:, 5] - 0.14,
            r[:, 8] - 1.1,
            r[:, 9] - 3.6,
            r[:, 10] - 3.5,
            r[:, 11] - 4.5,
            r[:, 12] - 0.035,
            np.abs(r[:, 3]),
            np.abs(r[:, 4] - 0.65),
            np.maximum(0.0, 0.08 - r[:, 5]),
        )
    )


SCALE = np.array([0.42, 0.41, 0.10, 0.17, 0.92, 0.83, 1.0, 0.020, 0.37, 0.36, 0.047])


class MotorValue:
    def __init__(self, dataset):
        self.x = np.asarray(dataset["x"]) / SCALE
        self.y = np.asarray(dataset["y"])
        self.mean = self.y.mean(axis=0)
        z = np.c_[np.ones(len(self.x)), self.x]
        penalty = np.diag(np.r_[0.01, np.full(self.x.shape[1], 12.0)])
        self.coef = np.linalg.solve(z.T @ z + penalty, z.T @ self.y)

    def predict(self, outcomes):
        return self.predict_features(features(outcomes))

    def predict_features(self, data):
        x = np.asarray(data) / SCALE
        logits = np.c_[np.ones(len(x)), x] @ self.coef
        ds = np.mean((x[:, None, :] - self.x[None, :, :]) ** 2, axis=2)
        w = np.exp(-1.5 * np.minimum(ds, 100.0))
        residual = self.y - np.c_[np.ones(len(self.x)), self.x] @ self.coef
        return np.clip(
            logits + w @ residual / (8.0 + w.sum(axis=1, keepdims=True)), 0.01, 0.99
        )

    def cost(self, outcomes):
        p = self.predict(outcomes)
        return 7 * (1.0 - p[:, 0]) + 4 * (1.0 - p[:, 1]) + 0.23 * outcomes[:, 0]


class OnlineMotorValue(MotorValue):
    """Credit the last issued proposal once both physical scores have resolved."""

    def __init__(self, dataset):
        super().__init__(dataset)
        self.prior_x = list(dataset["x"])
        self.prior_y = list(dataset["y"])
        self.examples = []
        self.begin_point()

    def begin_point(self):
        self.pending = None
        self.awaiting = None

    def issued(self, prediction):
        self.pending = (
            features(prediction)[0].tolist() if prediction is not None else None
        )

    def learn(self, x, y):
        self.examples.append((x, y))
        online = self.examples[-512:]
        super().__init__(
            {
                "x": self.prior_x + [v[0] for v in online],
                "y": self.prior_y + [v[1] for v in online],
            }
        )

    def feedback(self, sides, terminal):
        for side in sides:
            if side == 1:
                self.awaiting, self.pending = self.pending, None
            elif self.awaiting is not None:
                self.learn(self.awaiting, [1.0, 1.0])
                self.awaiting = None
        if terminal:
            if self.awaiting is not None:
                self.learn(self.awaiting, [1.0, 0.0])
            if self.pending is not None:
                self.learn(self.pending, [0.0, 0.0])
            self.begin_point()


def records(paths):
    unique = {}
    for path in paths:
        d = _read(path)
        if not str(d.get("mode")).startswith("optical-") or len(d["rows"]) != len(
            d["cards"]
        ):
            raise ValueError(f"Requires complete sensor-driven games: {path}")
        for card, r in zip(d["cards"], d["rows"], strict=True):
            moves = r["records"]
            legal = [
                (e["time"], e["side"])
                for e in r["events"]
                if e["kind"] == "legal_return"
            ]
            for i, move in enumerate(moves):
                if (
                    move["score"] is None
                    or move.get("late_repair", {}).get("status") == "corrected"
                ):
                    continue
                t0 = move["tick"] / 60.0
                t1 = moves[i + 1]["tick"] / 60.0 if i + 1 < len(moves) else np.inf
                if t1 - t0 < 0.40:
                    continue  # a later command replaced this stroke
                near = next((t for t, side in legal if side == 1 and t0 < t < t1), None)
                far = near is not None and any(
                    t > near and side == -1 for t, side in legal
                )
                k = (
                    card["id"],
                    card["seed"],
                    card["reply_seed"],
                    move["tick"],
                    *np.round(move["action"], 5),
                    *np.round(move["score"], 5),
                )
                unique[k] = (
                    card["id"],
                    features(move["score"])[0],
                    [float(near is not None), float(far)],
                )
    return unique


def fit(paths, output):
    unique = records(paths)
    idx, x, y = map(np.asarray, zip(*unique.values(), strict=True))
    dataset = {
        "x": x.tolist(),
        "y": y.tolist(),
        "release": idx.tolist(),
        "sources": {p.name: _hash(p) for p in paths},
    }
    # Release-blocked estimates, including every complete action at each opening.
    pred = np.zeros_like(y)
    naive = np.zeros_like(y)
    for fold in np.array_split(
        np.random.default_rng(38729).permutation(np.unique(idx)), 5
    ):
        test = np.isin(idx, fold)
        if not test.any():
            continue
        model = MotorValue({"x": x[~test], "y": y[~test]})
        pred[test] = model.predict_features(x[test])
        naive[test] = model.mean
    dataset.update(
        n=len(y),
        release_count=len(np.unique(idx)),
        brier=np.mean((pred - y) ** 2, axis=0).tolist(),
        constant_brier=np.mean((naive - y) ** 2, axis=0).tolist(),
        target_rates=y.mean(axis=0).tolist(),
    )
    _write(output, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    a = parser.parse_args()
    fit(a.sources, a.model)

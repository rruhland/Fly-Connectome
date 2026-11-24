"""Predict future arm returns from the observed long-term reward of previously issued moves."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt38_motor_value import MotorValue, features


def training(paths):
    records = {}
    for path in paths:
        doc = _read(path)
        if not str(doc.get("mode")).startswith("optical-") or len(doc["cards"]) != len(
            doc["rows"]
        ):
            raise ValueError(
                "Only complete sensor-control courses can train the return value"
            )
        for card, row in zip(doc["cards"], doc["rows"], strict=True):
            moves = row["records"]
            near = np.array(
                [
                    e["time"]
                    for e in row["events"]
                    if e["kind"] == "legal_return" and e["side"] == 1
                ]
            )
            far = np.array(
                [
                    e["time"]
                    for e in row["events"]
                    if e["kind"] == "legal_return" and e["side"] == -1
                ]
            )
            complete = any(e["kind"] == "point" for e in row["events"])
            for i, m in enumerate(moves):
                if (
                    m["score"] is None
                    or m.get("late_repair", {}).get("status") == "corrected"
                ):
                    continue
                t = m["tick"] / 60.0
                next_move = (
                    moves[i + 1]["tick"] / 60.0 if i + 1 < len(moves) else np.inf
                )
                if next_move - t < 0.4:
                    continue
                arrivals = near[near > t]
                landed = len(arrivals) > 0 and arrivals[0] < next_move
                if not complete and (not landed or len(arrivals) < 3):
                    continue
                reply = landed and np.any(far > arrivals[0])
                key = (
                    card["id"],
                    card["seed"],
                    card["reply_seed"],
                    m["tick"],
                    *np.round(m["action"], 5),
                    *np.round(m["score"], 5),
                )
                records[key] = (
                    card["id"],
                    features(m["score"])[0],
                    np.array(
                        [
                            landed,
                            reply,
                            landed and len(arrivals) >= 2,
                            landed and len(arrivals) >= 3,
                        ],
                        dtype=float,
                    ),
                )
    return records


class RallyValue(MotorValue):
    def cost(self, outcomes):
        p = self.predict(outcomes)
        return (1.0 - p) @ np.array([6.5, 1.5, 5.0, 3.0]) + 0.23 * outcomes[:, 0]


class OnlineRallyValue(RallyValue):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.prior_x = list(dataset["x"])
        self.prior_y = list(dataset["y"])
        self.examples = []
        self.begin_point()

    def begin_point(self):
        self.pending = None
        self.carry = []

    def issued(self, prediction):
        self.pending = (
            features(prediction)[0].tolist() if prediction is not None else None
        )

    def learn(self, labels):
        if not labels:
            return
        self.examples.extend(labels)
        data = self.examples[-512:]
        super().__init__(
            {
                "x": self.prior_x + [r[0] for r in data],
                "y": self.prior_y + [r[1] for r in data],
            }
        )

    def feedback(self, sides, terminal):
        labels = []
        for side in sides:
            if side == -1:
                if self.carry:
                    self.carry[-1][1][1] = 1.0
                continue
            for entry in self.carry:
                y = entry[1]
                if y[2]:
                    y[3] = 1.0
                else:
                    y[2] = 1.0
            while self.carry and self.carry[0][1][3]:
                labels.append(self.carry.pop(0))
            if self.pending is not None:
                self.carry.append((self.pending, [1.0, 0.0, 0.0, 0.0]))
                self.pending = None
        if terminal:
            labels.extend(self.carry)
            if self.pending is not None:
                labels.append((self.pending, [0.0, 0.0, 0.0, 0.0]))
            self.begin_point()
        self.learn(labels)


def fit(paths, out):
    ids, x, y = map(np.asarray, zip(*training(paths).values(), strict=True))
    pred = np.empty_like(y)
    baseline = np.empty_like(y)
    for g in np.array_split(
        np.random.default_rng(38729).permutation(np.unique(ids)), 5
    ):
        test = np.isin(ids, g)
        if test.any():
            model = RallyValue({"x": x[~test], "y": y[~test]})
            pred[test] = model.predict_features(x[test])
            baseline[test] = model.mean
    _write(
        out,
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "release": ids.tolist(),
            "n": len(ids),
            "sources": {p.name: _hash(p) for p in paths},
            "rates": y.mean(axis=0).tolist(),
            "brier": ((y - pred) ** 2).mean(axis=0).tolist(),
            "constant_brier": ((y - baseline) ** 2).mean(axis=0).tolist(),
            "targets": ["land", "reply", "second_own", "third_own"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    a = p.parse_args()
    fit(a.sources, a.out)

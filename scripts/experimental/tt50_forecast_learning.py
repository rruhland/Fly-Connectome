"""Learn residual forecaster calibration from the next ball the cameras actually see."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt50_far_readout import GATES, ROOT, TARGET, Predictor, descriptor


class CameraCorrection:
    """Bounded recursive least squares around the previously learned visual readout."""

    def __init__(self, params):
        self.params = params
        self.correction = {g: np.zeros((13, 5)) for g in params}
        self.covariance = {g: np.eye(13) / 18 for g in params}
        self.count = {g: 0 for g in params}

    def features(self, g, x):
        mean, scale, coef, bias = self.params[g]
        optical = (x - mean) / scale @ coef + bias
        # Correct systematic offsets that recur for this outgoing and predicted
        # next motion. No opponent goal, simulation parameters or outcome reward.
        z = np.r_[
            1.0,
            np.clip((x[:6] - mean[:6]) / scale[:6], -3, 3),
            np.clip((optical - bias) / [0.15, 0.045, 0.32, 0.45, 0.4], -3, 3),
            0.0,
        ]
        z[-1] = np.clip(z[7] * z[8], -3, 3)
        return optical, z

    def read(self, g, x):
        optical, z = self.features(g, x)
        return optical + z @ self.correction[g]

    def observe(self, g, x, y):
        optical, z = self.features(g, x)
        P = self.covariance[g]
        h = P @ z
        gain = h / (1 + z @ h)
        self.correction[g] += np.outer(gain, y - (optical + z @ self.correction[g]))
        P -= np.outer(gain, h)
        self.count[g] += 1


class Online(Predictor):
    def __init__(self, path=ROOT / "model.npz"):
        super().__init__(path)
        self.calibrator = CameraCorrection(self.params)

    def reset(self):
        super().reset()
        self.pending = {}
        self.last_in = None

    def update(self, frames, x, tick):
        # Base tracker clears its short appearance history upon every incoming
        # exposure. Keep separately the pending forecasts until their observable
        # future crosses the scoring depth, including on a complete miss.
        pending, last = self.pending, self.last_in
        result = super().update(frames, x, tick)
        self.pending, self.last_in = pending, last
        if x is not None and x[4] < -0.9:
            if last is not None and last[1] > -1.1 >= x[1] and pending:
                target = (last + (x - last) * ((-1.1 - last[1]) / (x[1] - last[1])))[
                    TARGET
                ]
                for g, feature in pending.items():
                    self.calibrator.observe(g, feature, target)
                self.pending = {}
            self.last_in = x
        if result is None:
            return None
        gate, _ = result
        f = descriptor(self.history, x)
        if gate == GATES[0] or self.last_in is not None:
            self.pending = {}
        self.pending[gate] = f.copy()
        self.last_in = None
        self.latest = (gate, self.calibrator.read(gate, f))
        return self.latest


def evaluate():
    ref = Predictor()
    paths = sorted((ROOT / "untrained-episodes").glob("game-*.json.gz"))
    history = sorted((ROOT / "practice-episodes").glob("game-*.json.gz"))
    if not paths or not history:
        raise ValueError("Run the matched native exposure captures first")
    trained = CameraCorrection(ref.params)
    for path in history:
        for leg in _read(path)["observed"]:
            for d in leg["decisions"]:
                trained.observe(
                    d["gate"], np.asarray(d["x"]), np.asarray(leg["target"])
                )
    cold = CameraCorrection(ref.params)
    score = {}
    from . import tt46_continuous_season as season

    native = _read(season.ROOT / "learning-24.json.gz")
    excluded = 0
    for i, path in enumerate(paths):
        local = CameraCorrection(ref.params)
        record = _read(path)
        truth = native["rows"][record["index"]]["placements"]["delivered"]
        far = np.array(
            [r["far_contact"]["time"] * 60 for r in truth if r["far_contact"]]
        )
        for j, leg in enumerate(record["observed"]):
            far_ix = np.searchsorted(far, leg["tick"]) - 1
            y = np.asarray(leg["target"])
            for d in leg["decisions"]:
                g = d["gate"]
                x = np.asarray(d["x"])
                forecasts = [
                    np.asarray(d["forecast"]),
                    local.read(g, x),
                    cold.read(g, x),
                    trained.read(g, x),
                ]
                precontact = far_ix >= 0 and 0 < far[far_ix] - d["tick"] < 110
                excluded += int(not precontact)
                for frame, pred in zip(
                    ("frozen", "point", "carried", "practiced_and_carried"), forecasts
                ):
                    if not precontact:
                        continue
                    for window in ("all", "first_five" if j < 5 else "six_on"):
                        key = f"{frame}/{g}/{window}"
                        score.setdefault(key, []).append(abs(pred - y))
                for model in (local, cold, trained):
                    model.observe(g, x, y)
    _write(
        ROOT / "continual.json",
        dict(
            practice_games=len(history),
            held_games=len(paths),
            excluded_from_precontact_scoring=excluded,
            stages={
                k: dict(n=len(v), mae=np.mean(v, axis=0).tolist())
                for k, v in score.items()
            },
        ),
    )


if __name__ == "__main__":
    evaluate()

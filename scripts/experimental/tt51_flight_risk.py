"""Learn the conditional lower tail of a future sensory trajectory without simulator inputs."""


import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from .tt4_transfer_trial import _read, _write
from .tt50_far_readout import GATES, Predictor, ROOT, descriptor, examples
from .tt50_group_forecast import columns


def sensor_future(y):
    # Evaluate an observable short-horizon vertical/lateral extrapolation. Real
    # scoring remains the native referee, including all later camera errors.
    x, z, vx, vy, vz = y.T
    t = 0.43 / np.maximum(1.35, -0.98 * vy)
    xx = x + t * vx
    zz = z + t * vz - 4.905 * t * t
    low = zz < 0.987
    leftlow = low & (xx < 0.025) & (xx > -0.28)
    return np.column_stack((y, xx, zz, low, leftlow))


class Kernel:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def normalize(X, m, s):
        return np.clip((X - m) / s, -5, 5)

    @staticmethod
    def gram(X, Y):
        b = cdist(X[:, :6], Y[:, :6], "sqeuclidean") / 6
        s = cdist(X[:, 6:], Y[:, 6:], "sqeuclidean") / (X.shape[1] - 6)
        # Separate smooth effects from their interaction. A new incoming ball
        # can resemble a known outgoing trajectory or a known visible motion.
        return (
            0.35 * np.exp(-b / 0.95)
            + 0.30 * np.exp(-s / 1.3)
            + 0.35 * np.exp(-b / 2.8 - s / 2.2)
        )

    @classmethod
    def fit(cls, X, Y, weights):
        idx = columns("position_face")
        X = X[:, idx]
        m = np.median(X, axis=0)
        s = np.maximum(np.percentile(abs(X - m), 75, axis=0), 0.001)
        u = cls.normalize(X, m, s)
        b = np.average(Y, axis=0, weights=weights)
        K = cls.gram(u, u)
        C = solve(K + np.diag(0.22 / weights), Y - b, assume_a="pos")
        return cls((idx, m, s, u, C, b))

    def predict(self, X):
        idx, m, s, u, C, b = self.data
        return self.gram(self.normalize(np.atleast_2d(X)[:, idx], m, s), u) @ C + b


class RiskForecast(Predictor):
    def __init__(self, path=ROOT / "risk-model.npz"):
        super().__init__(ROOT / "model.npz")
        with np.load(path) as store:
            self.models = {
                g: Kernel(tuple(store[f"{i}_{k}"] for k in range(6)))
                for i, g in enumerate(GATES)
            }
        self.future = None

    def reset(self):
        super().reset()
        self.future = None

    def update(self, frames, x, t):
        before = super().update(frames, x, t)
        if before is None:
            return None
        gate, _ = before
        f = descriptor(self.history, x)
        self.future = self.models[gate].predict(f)[0]
        self.latest = (gate, self.future[:5].copy())
        return self.latest


def load():
    plan = _read(ROOT / "difficult-plan.json")["cases"]
    rare_hold = {c["id"] for c in plan if c["split"] == "held"}
    with np.load(ROOT / "model.npz") as z:
        bg = z["background"]
    rows = []
    for group in ("observations", "difficult"):
        for path in sorted((ROOT / group).glob("case-*.npz")):
            j = int(path.stem.split("-")[-1])
            target = ROOT / "cached" / group / path.name
            if not target.exists():
                target.parent.mkdir(exist_ok=True, parents=True)
                obs = examples(path, bg)
                if not obs:
                    continue
                np.savez_compressed(
                    target,
                    gate=[r[0] for r in obs],
                    stream=[r[1] for r in obs],
                    release=[r[2] for r in obs],
                    x=[r[3] for r in obs],
                    y=[r[4] for r in obs],
                )
            with np.load(target) as e:
                for gate, stream, release, x, y in zip(
                    *(e[k] for k in ["gate", "stream", "release", "x", "y"]),
                    strict=True,
                ):
                    hold = (group == "observations" and j >= 16) or int(
                        release
                    ) in rare_hold
                    selected = group == "difficult" and stream == 0
                    rows.append((gate, hold, selected, group, int(release), x, y))
    # Natural chronological practice retains common afterstates and their
    # class frequencies. Exclude both rare and ordinary held release IDs.
    reserved = set(rare_hold) | {r[4] for r in rows if r[1]}
    # The deployed model was frozen at the first 35 complete chronological games.
    for path in sorted((ROOT / "practice-episodes").glob("game-*.json.gz"))[:35]:
        episode = _read(path)
        if episode["id"] in reserved:
            continue
        for leg in episode["observed"]:
            for d in leg["decisions"]:
                rows.append(
                    (
                        d["gate"],
                        False,
                        False,
                        "continuous",
                        episode["id"],
                        np.asarray(d["x"]),
                        np.asarray(leg["target"]),
                    )
                )
    return rows


def assess(pred, y):
    low = y[:, -1].astype(bool)
    priority = pred[:, -1]
    order = np.argsort(-priority)
    top = order[: max(1, int(np.ceil(len(order) / 4)))]
    return dict(
        n=len(y),
        left_low=int(low.sum()),
        top_quarter=len(top),
        low_in_top=int(low[top].sum()),
        deep_x_mae=float(np.mean(abs(pred[:, 5] - y[:, 5]))),
        deep_z_mae=float(np.mean(abs(pred[:, 6] - y[:, 6]))),
        left_low_brier=float(np.mean((np.clip(priority, 0, 1) - low) ** 2)),
        prevalence_brier=float(low.mean() * (1 - low.mean())),
    )


def run():
    rows = load()
    if not rows:
        return
    saved = {}
    report = {}
    for i, g in enumerate(GATES):
        use = [r for r in rows if r[0] == g]
        op = [r for r in use if not r[1]]
        val = [r for r in use if r[1]]
        X = np.array([r[5] for r in op])
        Y = sensor_future(np.array([r[6] for r in op]))
        # The original offer caused selection into the difficult distribution.
        # It teaches the existence of a low path but gets less weight than each
        # independent redraw when learning its frequency.
        w = np.array([0.35 if r[2] else 1.0 for r in op])
        model = Kernel.fit(X, Y, w)
        for k, part in enumerate(model.data):
            saved[f"{i}_{k}"] = part
        item = {"fit": len(X), "fit_low_left": int(Y[:, -1].sum())}
        for name, pool in (
            (
                "ordinary",
                [
                    r
                    for r in val
                    if r[3] == "observations"
                    and r[4]
                    not in {
                        c["id"]
                        for c in _read(ROOT / "difficult-plan.json")["cases"]
                        if c["split"] == "held"
                    }
                ],
            ),
            ("hard_redraws", [r for r in val if r[3] == "difficult" and not r[2]]),
            ("hard_original", [r for r in val if r[2]]),
        ):
            if pool:
                xx = np.array([r[5] for r in pool])
                yy = sensor_future(np.array([r[6] for r in pool]))
                item[name] = assess(model.predict(xx), yy)
        report[str(g)] = item
    np.savez_compressed(ROOT / "risk-model.npz", **saved)
    _write(ROOT / "risk-evaluation.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    run()

"""Compare time-aligned opponent and outgoing ball motion to their earlier observed motion."""

import numpy as np
from .tt4_transfer_trial import _read, _write
from .tt50_far_readout import Predictor, appearance, ROOT, GATES, TARGET
from .tt51_flight_risk import Kernel, assess, sensor_future


class Path:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.marks = {}

    def observe(self, t, x, face):
        if x is None or x[4] < 0.9:
            return
        self.samples.append((t, np.array(x), face))
        for v in (-1.05, -0.75, -0.45, 0.0, 0.45, 0.85):
            if x[1] >= v and v not in self.marks:
                self.marks[v] = (t, np.array(x), face)

    def features(self, gate):
        t, x, f = self.samples[-1]
        # Public longitudinal motion serves as the clock under changes of pace.
        # Compare optical flight and visible racket displacement at the same
        # path landmarks so variations in the outgoing ball can be explained.
        subset = [0, 1, 2, 3, 4, 5, 8, 9, 14, 15, 16, 17, 18, 19, 22, 23]
        cur = f[subset]
        out = [x, cur]
        for v in (-1.05, -0.75, -0.45, 0.0, 0.45):
            if v >= gate:
                continue
            a, b, m = self.marks.get(v, (t, x, f))
            out.extend(([min(2.0, (t - a) / 60)], x - b, cur - m[subset]))
        # Same feature width across gates, with only causal gates filled.
        z = np.concatenate(out)
        return np.nan_to_num(np.pad(z, (0, 137 - len(z))))


class Forecast(Predictor):
    def __init__(self, path=ROOT / "path-model.npz"):
        super().__init__(ROOT / "model.npz")
        with np.load(path) as z:
            self.models = {
                g: Kernel(tuple(z[f"{i}_{k}"] for k in range(6)))
                for i, g in enumerate(GATES)
            }
        self.future = None

    def reset(self):
        super().reset()
        self.path = Path()
        self.future = None

    def update(self, frames, x, t):
        old = super().update(frames, x, t)
        if frames is not None and self.history and x is not None and x[4] > 0.9:
            self.path.observe(t, x, self.history[-1])
        if old is None:
            return None
        g, _ = old
        y = self.models[g].predict(self.path.features(g))[0]
        self.future = y
        self.latest = (g, y[:5])
        return self.latest


def packet(path, bg):
    with np.load(path) as z:
        pixels = z["rgb"]
        face = [appearance(f, bg) for f in pixels]
        rows = []
        for b, target in enumerate(z["label"][:, TARGET]):
            if not np.isfinite(target).all():
                continue
            model = Path()
            used = set()
            for k in np.flatnonzero(z["stream"] == b):
                x = z["ball"][k]
                t = z["tick"][k]
                if not np.isfinite(x).all() or x[4] < 0.9:
                    continue
                model.observe(t, x, face[k])
                for g in GATES:
                    if x[1] >= g and g not in used:
                        used.add(g)
                        rows.append((g, b, z["id"].item(), model.features(g), target))
        return rows


class MotionKernel(Kernel):
    @classmethod
    def fit(cls, X, Y, weights):
        # The covariance between early and late portions of the same observed
        # flight is useful, even for a paddle whose screen position changed.
        from scipy.linalg import solve

        m = np.median(X, axis=0)
        s = np.maximum(np.percentile(abs(X - m), 75, axis=0), 0.05)
        idx = np.flatnonzero(np.maximum(X.max(0) - X.min(0), 0) > 1e-8)
        m, s = m[idx], s[idx]
        u = cls.normalize(X[:, idx], m, s)
        b = np.average(Y, axis=0, weights=weights)
        C = solve(cls.gram(u, u) + np.diag(0.30 / weights), Y - b, assume_a="pos")
        return cls((idx, m, s, u, C, b))


def run():
    plan = _read(ROOT / "difficult-plan.json")["cases"]
    rare = {c["id"] for c in plan if c["split"] == "held"}
    with np.load(ROOT / "model.npz") as z:
        bg = z["background"]
    rows = []
    for group in ("observations", "difficult"):
        for file in sorted((ROOT / group).glob("case-*.npz")):
            j = int(file.stem.split("-")[-1])
            target = ROOT / "path-cache" / group / file.name
            if not target.exists():
                obs = packet(file, bg)
                if not obs:
                    continue
                target.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(
                    target,
                    **{
                        k: [r[n] for r in obs]
                        for n, k in enumerate(("g", "stream", "id", "x", "y"))
                    },
                )
            with np.load(target) as e:
                for g, b, r, x, y in zip(
                    *(e[k] for k in ("g", "stream", "id", "x", "y")), strict=True
                ):
                    held = (group == "observations" and j >= 16) or r in rare
                    rows.append((g, held, group, b, int(r), x, y))
    saved = {}
    info = {}
    for i, g in enumerate(GATES):
        tr = [r for r in rows if r[0] == g and not r[1]]
        te = [r for r in rows if r[0] == g and r[1]]
        x = np.array([r[5] for r in tr])
        y = sensor_future(np.array([r[6] for r in tr]))
        m = MotionKernel.fit(
            x,
            y,
            np.array([0.35 if r[2] == "difficult" and r[3] == 0 else 1.0 for r in tr]),
        )
        for k, v in enumerate(m.data):
            saved[f"{i}_{k}"] = v
        info[str(g)] = {"train": len(tr)}
        for label, subset in (
            (
                "ordinary",
                [r for r in te if r[2] == "observations" and r[4] not in rare],
            ),
            ("hard_new", [r for r in te if r[2] == "difficult" and r[3] != 0]),
            ("hard_stored", [r for r in te if r[2] == "difficult" and r[3] == 0]),
        ):
            if subset:
                p = m.predict(np.array([r[5] for r in subset]))
                yy = sensor_future(np.array([r[6] for r in subset]))
                info[str(g)][label] = assess(p, yy) | {
                    "mae": np.mean(abs(p[:, :5] - yy[:, :5]), axis=0).tolist()
                }
    np.savez_compressed(ROOT / "path-model.npz", **saved)
    _write(ROOT / "path-results.json", info)
    print(info, flush=True)


if __name__ == "__main__":
    run()

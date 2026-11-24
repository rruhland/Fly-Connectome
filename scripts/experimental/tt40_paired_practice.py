"""Fit motor corrections to paired practice outcomes across many naturally delivered incoming flights."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np
from scipy.stats import qmc

from . import tt40_low_lift as core
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_motor_clinic import AIM, geometry, schedules


def policies():
    rng = qmc.Sobol(9, scramble=True, seed=403013)
    grid = (rng.random_base2(7) * 2 - 1) * np.array(
        [0.14, 0.18, 0.20, 0.30, 0.18, 0.12, 0.17, 0.22, 0.035]
    )
    # Mirrored alternatives let the reward distinguish the direction of an effect.
    anchors = np.r_[
        np.zeros((1, 9)),
        np.pad(np.diag([0.09, 0.13, 0.14, 0.23, 0.16]), ((0, 0), (0, 4))),
        np.pad(-np.diag([0.09, 0.13, 0.14, 0.23, 0.16]), ((0, 0), (0, 4))),
    ]
    return np.vstack([anchors, grid, -grid[:32]])


def motions(fit, selected, eta, dt, home, rules):
    n = round((eta + 1.15) / dt)
    t = np.arange(1, n + 1) * dt
    shifted = t[None, :] + rules[:, 8, None]
    v = 0.86 * fit[1] + selected[5:]
    q = (
        fit[0]
        + selected[:5]
        + np.clip(shifted - eta + 0.030, -0.13, 0.065)[:, :, None] * v
    )
    early = np.clip(t / max(0.036, 0.58 * eta), 0, 1)
    early = early * early * (3 - 2 * early)
    drive = np.clip((t - eta + 0.084) / 0.072, 0, 1)
    q += early[None, :, None] * rules[:, :5][:, None, :]
    q[:, :, 1:4] += drive[None, :, None] * rules[:, 5:8][:, None, :]
    return np.concatenate([q, np.full((*q.shape[:2], 1), -0.1)], axis=2)


def describe(engine, obs, q, fit, chosen, eta, dt, home):
    stock = schedules(*fit, eta, chosen[None], dt, home, round((eta + 1.15) / dt))
    feedback = np.zeros((1, 9))
    _, nominal = engine.trajectories(stock, AIM, eta + 0.12, feedback=feedback)
    # Every value is available before the decision, including the incumbent's effect
    # in the independent plant initialized exclusively by RGB and encoders.
    f = feedback[0]
    return np.r_[
        obs[:6],
        q[:5],
        chosen[:5],
        chosen[5:],
        f[0],
        f[1],
        f[2:5],
        float(np.isfinite(nominal[0, 0])),
    ]


SCALE = np.r_[
    0.06,
    0.16,
    0.043,
    0.85,
    0.85,
    0.75,
    0.75,
    0.45,
    0.55,
    0.75,
    0.85,
    0.22,
    0.28,
    0.32,
    0.30,
    0.4,
    2.0,
    2.4,
    2.4,
    2.5,
    3.0,
    0.040,
    0.8,
    1.8,
    1.25,
    1.25,
    0.85,
]


def practice(sources, out):
    rng = policies()
    xs = []
    ys = []
    groups = []
    ticks = []
    tapes = {}
    for source in sources:
        data = _read(source)
        if len(data["cards"]) != len(data["rows"]) or data.get("camera_stride", 2) != 2:
            raise ValueError("Incomplete or incompatible practice archive")
        depth = data["depth"]
        core.joint.RECEIVE_Y = depth
        surf = core.setup(depth)
        if not np.allclose(surf.ready, data["ready"]):
            raise ValueError("Mismatch in starting calibration")
        start = process_time()
        with core.centered(data["ready"]):
            a = core.arena()
            renderer, rig, stride, lo, hi = core.joint.resources(a)
            private = core.OpticalClinic(body=True)
            evaluation = Rollout(a, 1)
            try:
                for card, r in zip(data["cards"], data["rows"], strict=True):
                    a.launch(r["shot"], card=card)
                    a.partner.reset(seed=card["seed"])
                    private.previous = None
                    private.bounced = False
                    eye = core.joint.Vision(rig, a.control_dt)
                    eye.eye = core.LearnedOptics(eye.eye, core.OPTICS)
                    history = core.ArrivalHistory()
                    ref = np.asarray(data["ready"]).copy()
                    rec = {
                        m["tick"]: m
                        for m in r["records"]
                        if not m.get("surface_shift", 0)
                    }
                    for tick, u in enumerate(r["actions"]):
                        q = a.joints.copy()
                        obs = eye.update(
                            tick, renderer.capture() if tick % 2 == 0 else None, q
                        )
                        private.observe(tick, eye.eye, q, ref)
                        obs = history.update(tick * a.control_dt, obs)
                        m = rec.get(tick)
                        if (
                            m
                            and obs is not None
                            and obs[2] < 1.075
                            and 0.03 < obs[0] < 0.19
                            and private.sync()
                        ):
                            if not np.allclose(
                                m["context"], np.r_[obs[1:6], q[:5]], atol=1e-8
                            ):
                                raise ValueError(
                                    "Did not reconstruct the archived camera decision"
                                )
                            params = np.asarray(m["action"])
                            fit = geometry(surf, obs)
                            x = describe(
                                private,
                                obs,
                                q,
                                fit,
                                params,
                                obs[0],
                                a.control_dt,
                                surf.ready,
                            )
                            cmds = motions(
                                fit, params, obs[0], a.control_dt, surf.ready, rng
                            )
                            _, score = evaluation.trajectories(cmds, AIM, obs[0] + 0.12)
                            xs.append(x)
                            ys.append(np.isfinite(score[:, 0]))
                            groups.append(card["id"])
                            ticks.append(tick)
                        ref = np.clip(ref + np.asarray(u) * stride, lo, hi)
                        a.exchange(u, core.clinic.FAR)
                    if digest_json(r["events"]) != digest_json(a.events):
                        raise ValueError("Different physical game on practice replay")
                    print(
                        card["id"], len(xs), round(process_time() - start), flush=True
                    )
                    np.savez_compressed(
                        out, x=xs, y=ys, release=groups, tick=ticks, moves=rng
                    )
            finally:
                renderer.close()
                private.twin.close()
                a.close()
        tapes[str(source)] = _hash(source)
    _write(
        out.with_suffix(".json"),
        {
            "n": len(xs),
            "sources": tapes,
            "sha256": _hash(out),
            "physics_forks": len(xs) * len(rng),
            "boundary": "Parallel native practice with binary landing reward. Policy descriptors are computed only from RGB, encoders, issued setpoints and incumbent prediction in the camera-initialized model.",
        },
    )


class Paired:
    def __init__(self, path, *, learn=True):
        with np.load(path) as z:
            self.x, self.y, self.ids, self.moves = (
                z[n] for n in ["x", "y", "release", "moves"]
            )
        self.learn = learn
        self.history = []
        self.begin()

    def begin(self):
        self.pending = None

    def cancel(self):
        self.pending = None

    def feedback(self, legal, terminal):
        if (legal or terminal) and self.pending is not None:
            if self.learn:
                self.history.append((*self.pending, int(legal)))
            self.pending = None

    def choice(self, x, mask=None):
        ids = self.ids if mask is None else self.ids[mask]
        state = self.x if mask is None else self.x[mask]
        y = self.y if mask is None else self.y[mask]
        delta = (state - x) / SCALE
        distance = (
            np.sum(delta[:, :6] ** 2, axis=1) * 0.58
            + np.sum(delta[:, 6:21] ** 2, axis=1) * 0.17
            + np.sum(delta[:, 21:] ** 2, axis=1) * 0.52
        )
        # Several actions in a single approach and several similar shots on one
        # release get at most two votes from that release.
        groups = {}
        indices = []
        for i in np.argsort(distance):
            k = ids[i]
            if groups.get(k, 0) >= 2:
                continue
            groups[k] = groups.get(k, 0) + 1
            indices.append(i)
            if len(indices) >= 24:
                break
        w = np.exp(
            -0.5 * distance[indices] / max(1.0, float(np.median(distance[indices[:8]])))
        )
        adv = y[indices].astype(float) - y[indices, 0, None]
        gain = w @ adv / (3 + w.sum())
        if self.history:
            # Actual recently executed interventions adjust only the chosen action.
            for hx, j, win in self.history[-120:]:
                if j:
                    similarity = np.exp(-0.5 * np.mean(((hx - x) / SCALE) ** 2) * 5)
                    gain[j] += 0.038 * similarity * (win - 0.65)
        effort = 0.009 * np.linalg.norm(self.moves / 0.18, axis=1)
        j = int(np.argmax(gain - effort))
        if gain[j] - effort[j] < 0.065 or min(distance[indices]) > 22:
            j = 0
        return j, float(gain[j]), float(w.sum())

    def plan(self, engine, surf, obs, selected, score, fit, eta):
        result = {
            "tape": None,
            "score": None,
            "eta": None,
            "trace": [],
            "note": {"issued": False},
        }
        self.pending = None
        if obs[2] > 1.075 or not 0.03 < eta < 0.19:
            return result
        q = engine.previous
        if q is None:
            return result
        x = describe(
            engine, obs, q, fit, selected, eta, engine.twin.control_dt, surf.ready
        )
        j, gain, support = self.choice(x)
        self.pending = (x, j)
        result["note"].update(mode=j, practice_advantage=gain, support=support)
        if j:
            n = max(1, round((eta + 0.075) / engine.twin.control_dt))
            result.update(
                tape=motions(
                    fit,
                    selected,
                    eta,
                    engine.twin.control_dt,
                    surf.ready,
                    self.moves[j : j + 1],
                )[0, :n],
                eta=eta,
            )
            result["commit"] = True
            result["note"]["issued"] = True
        return result


def compare_validation(source, out):
    model = Paired(source, learn=False)
    rows = []
    for i, (x, values, group) in enumerate(
        zip(model.x, model.y, model.ids, strict=True)
    ):
        j, _gain, _support = model.choice(x, model.ids != group)
        rows.append(
            {
                "release": int(group),
                "stock": bool(values[0]),
                "chosen": bool(values[j]),
                "mode": j,
                "any_available": bool(values.any()),
                "z": float(x[2]),
            }
        )
    _write(
        out,
        {
            "n": len(rows),
            "stock": sum(v["stock"] for v in rows),
            "chosen": sum(v["chosen"] for v in rows),
            "any_available": sum(v["any_available"] for v in rows),
            "changed": sum(v["mode"] != 0 for v in rows),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("practice", "validate"))
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    a = p.parse_args()
    if a.mode == "practice":
        practice(a.sources, a.out)
    else:
        compare_validation(a.sources[0], a.out)

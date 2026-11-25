"""Use the observed far rebound to prepare direct joints for a distribution of possible replies."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt80_prospective_course as game
from . import tt94_optical_rebound as sensory
from . import tt56_scored_motor_atlas as atlas
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

ROOT = sensory.ROOT
POSE = ROOT / "earned-postures.npz"
mpc = game.course.mpc


def fit():
    audit = _read(ROOT / "broad.json")
    forbidden = set(audit["reserved"])
    clips = {}
    limits = atlas.limits()
    for name, h in _read(atlas.ROOT / "fit.json")["sources"].items():
        path = Path(name)
        if _hash(path) != h:
            raise ValueError("Changed motor practice")
        d = _read(path)
        for c, r in zip(d["cards"], d["rows"]):
            if (
                c["id"] in forbidden
                or c.get("profile") != "reach"
                or c.get("opening") != "random"
            ):
                continue
            refs = atlas.reference_path(d["ready"], r["actions"], *limits)
            for o in r["placements"]["delivered"]:
                if (
                    not o.get("near_landed")
                    or not o["receive_line"]
                    or not o.get("near_contact")
                ):
                    continue
                j = o["near_contact"]["step"] - 1
                if j < 15:
                    continue
                v = sensory.response.projected(
                    np.r_[o["receive_line"]["xyz"], o["receive_line"]["velocity"]]
                )
                key = (
                    c["id"],
                    *np.round(
                        o["near_contact"]["ball"] + o["near_contact"]["velocity"], 5
                    ),
                )
                clips.setdefault(key, (v, refs[j - 12, :5], refs[j - 6, :5]))
    v, q, e = map(np.array, zip(*clips.values(), strict=True))
    np.savez_compressed(
        POSE,
        context=v,
        prepare=q,
        approach=e,
        release_ids=sorted({k[0] for k in clips}),
    )
    print(len(v), len({k[0] for k in clips}), flush=True)


class Ready:
    def __init__(self, disabled=False):
        with np.load(ROOT / "broad.npz") as p:
            self.mu, self.scale, self.yy = (p[k] for k in ("mu", "scale", "labels"))
            self.tree = cKDTree(p["z"])
        with np.load(POSE) as p:
            self.v, self.q, self.end = (
                p[k] for k in ("context", "prepare", "approach")
            )
            if set(p["release_ids"]) & {c["id"] for c in _read(game.PLAN)["practice"]}:
                raise ValueError("Held motor lesson")
        self.disabled = disabled

    def posterior(self, f):
        d, it = self.tree.query((f - self.mu) / self.scale, k=48)
        w = np.exp(-2 * (d / max(0.15, d[-1])) ** 2)
        return self.yy[it, 0], w / w.sum()

    def posture(self, f, q):
        outcome, w = self.posterior(f)
        h = outcome @ w
        p = w @ (outcome < 1.0)
        # Marginalize over the still unknown return. Balance lateral practice
        # to avoid spending the available recovery travel entirely on one wing.
        weight = np.exp(-0.5 * ((self.v[:, 0, None] - outcome) / 0.027) ** 2) @ w
        for ids in np.array_split(self.v[:, 1].argsort(), 3):
            weight[ids] /= max(1e-9, weight[ids].sum())
        weight /= weight.sum()
        # The candidate and its neighbors are all real rewarded arm references.
        # Expected late approach effort is an inexpensive way to choose a
        # compromise without interpolating between opposite wrist solutions.
        candidates = np.unique(
            np.linspace(0, len(self.q) - 1, min(96, len(self.q))).astype(int)
        )
        distances = np.max(
            abs(self.q[candidates, None] - self.end[None])
            / np.array([1, 1, 1, 1.1, 1.2]),
            axis=2,
        )
        value = distances @ weight + 0.12 * np.max(
            abs(self.q[candidates] - q[:5]), axis=1
        )
        idx = candidates[np.argmin(value)]
        return self.q[idx].copy(), float(p), float(h)


class Player(game.season.Player):
    def __init__(self, *args, policy, **kw):
        super().__init__(*args, **kw)
        self.policy = policy

    def begin(self):
        super().begin()
        self.reset_eye()
        self.forecasts = []
        self.anticipatory_moves = 0

    def reset_eye(self):
        self.rebound = sensory.Rebound()
        self.mark = -1000
        self.stamp = None
        self.prep = None
        self.was_inbound = False

    def step(self, tick, eye, x, q, reference, base):
        if x is not None and x[4] < -0.9:
            self.was_inbound = True
        if self.was_inbound and x is not None and x[4] > 0.9:
            self.reset_eye()
        if (
            x is not None
            and x[4] > 0.9
            and eye.latest is not None
            and self.stamp != eye.latest.time
        ):
            self.stamp = eye.latest.time
            f = self.rebound.update(tick, x)
            if f is not None:
                self.prep, p, h = self.policy.posture(f, q)
                self.mark = tick
                self.strength = min(0.75, 0.30 + 1.4 * p)
                self.forecasts.append(
                    dict(
                        tick=tick,
                        low=p,
                        height=h,
                        pose=self.prep.tolist(),
                        q=q.tolist(),
                    )
                )
        desired = base
        if self.prep is not None and tick - self.mark < 70 and not self.policy.disabled:
            fade = 1.0
            if x is not None and x[4] < -0.9:
                fade = np.clip((x[1] + 0.55) / 0.80, 0, 1)
                if fade == 0:
                    self.prep = None
            if self.prep is not None and fade > 0:
                desired = base.copy()
                desired[:5] += (
                    fade * self.strength * np.clip(self.prep - base[:5], -0.55, 0.55)
                )
                self.anticipatory_moves += 1
        return super().step(tick, eye, x, q, reference, desired)


def run(a):
    policy = Ready()
    rank = game.season.fresh.shot.Coverage(_read(game.season.fresh.RISK))
    original = mpc.play

    def play(*args, **kw):
        r = original(*args, **kw)
        m = args[10] if len(args) > 10 else kw["model"]
        r["rebound_forecasts"] = m.forecasts
        r["preparation_ticks"] = m.anticipatory_moves
        return r

    with (
        enabled(),
        patch.object(game.course, "PLAN", game.PLAN),
        patch.object(mpc, "play", play),
        patch.object(
            mpc, "MPC", lambda *x, **kw: Player(*x, policy=policy, coverage=rank, **kw)
        ),
    ):
        game.course.run(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "play"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        mode="live",
        deck="practice",
        carry=game.season.CARRY,
        cap=240,
        resume=False,
        out=ROOT / "live.json.gz",
    )
    a = p.parse_args()
    fit() if a.action == "fit" else run(a)

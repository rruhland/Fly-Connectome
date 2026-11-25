"""Use optically learned successful departures to repair a reachable but weak contact."""

import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt52_practice_dose as dose
from .tt4_transfer_trial import _hash, _read, _write
from .tt39_turn_time import turn_time
from .tt54_training_speedups import enabled

ROOT = Path("artifacts/tt76_learned_impulse")
season, course = dose.season, dose.course
core = course.mpc


def observe_departure(tape):
    motion = [(t, x) for t, x, *_ in tape if x is not None]
    at = turn_time(motion)
    if at is None:
        return None
    visible = [(t, x) for t, x in motion if at + 3 <= t <= at + 14 and x[4] > 0.9]
    if len(visible) < 3:
        return None
    t, x = zip(*visible, strict=True)
    dt = (np.array(t) - at)[:, None] * core.DT
    state = np.asarray(x)
    vel = state[:, 3:6]
    acc = [0, 0, -9.81] - 0.13 * np.linalg.norm(vel, axis=1, keepdims=True) * vel
    return np.median(
        np.c_[state[:, :3] - dt * vel + 0.5 * dt**2 * acc, vel - dt * acc], axis=0
    )


def profiles(tick_at):
    # The first pulse alters the pose. The second reverses before impact so
    # a feasible face can have a different velocity at nearly the same pose.
    time = np.arange(core.HORIZON) - tick_at
    leading = np.exp(-0.5 * ((time + 3) / 4.5) ** 2)
    braking = leading - 1.65 * np.exp(-0.5 * ((time + 7) / 3.0) ** 2)
    return np.asarray([leading, braking])


def velocity_step(nominal, perturbations, outcomes, desired):
    # Only fit symmetrical interventions which still contact on both sides.
    usable = outcomes[:, 0, 1].astype(bool) & outcomes[:, 1, 1].astype(bool)
    if not usable.any():
        return []
    delta = (outcomes[usable, 0, 2:5] - outcomes[usable, 1, 2:5]).T * 0.5
    # Discontinuous contacts can change branch between neighboring commands.
    midpoint = outcomes[usable, :, 2:5].mean(1)
    trust = 1 / (1 + np.linalg.norm(midpoint - nominal, axis=1) ** 2)
    a = delta * trust
    step = trust * (
        a.T @ np.linalg.solve(a @ a.T + 0.12**2 * np.eye(3), desired - nominal)
    )
    change = np.einsum("i,itj->tj", np.clip(step, -2.8, 2.8), perturbations[usable])
    return [s * change for s in (0.45, 0.95, 1.55)]


class Player(season.Player):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.departures = deque(maxlen=384)
        paid = self.priors.keep

        def remember(tape, tag):
            example = observe_departure(tape)
            if example is not None:
                self.departures.append(example)
            return paid(tape, tag)

        self.priors.keep = remember
        self.pool = None

    def trial(self, goals, *, long=False, traces=False):
        result = super().trial(goals, long=long, traces=traces)
        if self.pool is not None and not self.in_rank and not long:
            loss, _, _, _, feedback = result
            for i in np.argsort(loss)[:3]:
                f = feedback[i]
                if f[1] and f[8] < 0.079:
                    self.pool[np.asarray(goals[i]).tobytes()] = (
                        loss[i],
                        goals[i],
                        f.copy(),
                    )
        return result

    def search(self, tick, x, q, reference, base):
        self.pool = {} if len(self.departures) >= 3 else None
        try:
            super().search(tick, x, q, reference, base)
        finally:
            pool, self.pool = self.pool, None
        if (
            not pool
            or not self.notes
            or self.notes[-1]["tick"] != tick
            or self.notes[-1]["legal_trials"]
        ):
            return
        anchor = min(pool.values(), key=lambda row: row[0])
        _, goals, nominal = anchor
        tick_at = (nominal[6] - x[1]) / (x[4] * core.DT)
        if not 2 <= tick_at <= core.HORIZON:
            return
        self.diag["impulse_screens"] += 1
        learned = np.asarray(self.departures)
        dist = np.linalg.norm(
            (learned[:, :3] - nominal[5:8]) / [0.15, 0.10, 0.12], axis=1
        )
        neighbors = np.argsort(dist)[:8]
        target = np.median(learned[neighbors, 3:6], axis=0)
        basis = (
            np.concatenate(
                [f[None, :, None] * np.eye(6)[:5, None, :] for f in profiles(tick_at)]
            )
            * 0.055
        )
        candidates = np.stack((goals + basis, goals - basis), axis=1).reshape(
            -1, core.HORIZON, 6
        )
        _, _, _, _, feedback = self.trial(candidates)
        repairs = velocity_step(nominal[2:5], basis, feedback.reshape(-1, 2, 9), target)
        cand = [goals + delta for delta in repairs]
        # The actual probes also count as potentially useful motor solutions.
        take = np.argsort(
            np.where(
                feedback[:, 1] > 0,
                np.linalg.norm(feedback[:, 2:5] - target, axis=1),
                np.inf,
            )
        )[:3]
        cand.extend(candidates[i] for i in take if feedback[i, 1])
        if not cand:
            return
        loss, legal, _, _, _ = self.trial(cand, long=True)
        valid = np.flatnonzero(legal)
        if not len(valid):
            return
        ids = valid[np.argsort(loss[valid])[:3]]
        agree, quality = self.uncertainty([cand[i] for i in ids], loss[ids])
        secure = np.flatnonzero(agree == 2)
        if not len(secure):
            return
        k = secure[np.argmin(quality[secure])]
        self.best, self.tick, self.lease = cand[ids[k]], tick, core.LEASE
        self.diag["impulse_deployments"] += 1
        self.notes[-1]["learned_impulse"] = dict(
            neighbors=len(neighbors),
            predicted_velocity=nominal[2:5].tolist(),
            desired_velocity=target.tolist(),
            probes=len(candidates),
            quality=float(quality[k]),
        )


def run(a):
    ROOT.mkdir(exist_ok=True)
    rank = season.fresh.shot.Coverage(_read(season.fresh.RISK))
    actors = []

    def factory(*args, **kw):
        actor = Player(*args, coverage=rank, **kw)
        actors.append(actor)
        return actor

    with (
        enabled(),
        patch.object(course, "PLAN", dose.PLAN),
        patch.object(core, "MPC", factory),
    ):
        course.run(a)
    d = _read(a.out)
    np.savez_compressed(
        ROOT / "departures.npz", samples=np.asarray(actors[0].departures)
    )
    d["departures"] = _hash(ROOT / "departures.npz")
    _write(a.out, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        out=ROOT / "games.json.gz",
        mode="live",
        deck="practice",
        carry=season.CARRY,
        resume=False,
        cap=240,
    )
    run(p.parse_args())

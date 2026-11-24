"""Reconsider the remaining joint commands using small, smooth changes after a new observation."""

import numpy as np
from scipy.stats import qmc

from .tt37_motor_clinic import AIM, landing_error


def revise(engine, tail, eta, dt, tick):
    if not len(tail):
        return None, None
    extra = round(1.10 / dt)
    prior = np.concatenate([tail, np.broadcast_to(tail[-1], (extra, 6))])
    _, test = engine.trajectories(prior[None], AIM, eta + 0.14)
    if np.isfinite(test[0, 0]):
        return tail.copy(), {"status": "retained", "trials": 1}
    choices = 2 * qmc.Sobol(10, scramble=True, seed=37793 + tick).random_base2(6) - 1
    scale = np.r_[
        np.array([0.11, 0.15, 0.17, 0.18, 0.24]),
        np.array([0.08, 0.12, 0.14, 0.16, 0.20]),
    ]
    params = np.vstack([np.zeros(10), choices]) * scale
    elapsed = np.arange(len(prior)) * dt
    advance = np.minimum(1.0, (elapsed + dt) / max(0.025, eta * 0.5))
    bend = np.clip((elapsed - eta) / max(0.035, eta), -1.0, 1.0)
    rng = np.random.default_rng(37841 + tick)
    total = 1
    selected = None
    for _ in range(4):
        bias = params[:, :5, None] * advance + params[:, 5:, None] * (advance * bend)
        goals = np.broadcast_to(prior, (len(params), *prior.shape)).copy()
        goals[:, :, :5] += bias.transpose(0, 2, 1)
        feedback = np.zeros((len(goals), 9))
        _, result = engine.trajectories(goals, AIM, eta + 0.14, feedback=feedback)
        valid = np.flatnonzero(np.isfinite(result[:, 0]))
        total += len(goals)
        if len(valid):
            penalty = 0.4 * np.mean((params[valid] / scale) ** 2, axis=1)
            winner = valid[np.argmin(result[valid, 0] + penalty)]
            return goals[winner, : len(tail)], {
                "status": "corrected",
                "trials": total,
                "legal": len(valid),
                "delta": params[winner].tolist(),
            }
        loss = np.where(
            feedback[:, 1] > 0, landing_error(feedback), 10.0 + 20.0 * feedback[:, 0]
        )
        elite = params[np.argsort(loss)[:8]]
        params = elite[rng.integers(8, size=64)] + rng.normal(0, 0.45, (64, 10)) * scale
    return selected, {"status": "no_correction", "trials": total}

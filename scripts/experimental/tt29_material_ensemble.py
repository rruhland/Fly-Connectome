"""Score a movement under plausible changes in rebound physics using isolated models."""

from copy import copy

import mujoco
import numpy as np


class MaterialEnsemble:
    def __init__(self, native, score_model=None):
        self.nominal = score_model(native) if score_model else native
        self.variants = []
        for dissipation, mass in ((0.8, 0.95), (1.2, 1.05)):
            trial = copy(native)
            trial.model = copy(native.model)
            for geom in (*native.arena.table_panels, "learner_racket"):
                trial.model.geom(geom).solref[1] *= dissipation
            ball = trial.model.body("ball")
            ball.mass[:] *= mass
            ball.inertia[:] *= mass
            mujoco.mj_setConst(trial.model, mujoco.MjData(trial.model))
            self.variants.append(score_model(trial) if score_model else trial)

    def trajectories(self, goals, target, cutoff):
        commands, scores = self.nominal.trajectories(goals, target, cutoff)
        eligible = np.flatnonzero(np.isfinite(scores[:, 0]))
        if len(eligible) < 2:
            return commands, scores
        shortlist = eligible[np.argsort(scores[eligible, 0])[:18]]
        costs = scores[shortlist, 0].copy()
        for variant in self.variants:
            _, simulated = variant.trajectories(goals[shortlist], target, cutoff)
            alternate = simulated[:, 0]
            costs += 0.32 * np.where(np.isfinite(alternate), alternate, 5.0)
        scores[eligible, 0] = np.inf
        scores[shortlist, 0] = costs / 1.64
        return commands, scores

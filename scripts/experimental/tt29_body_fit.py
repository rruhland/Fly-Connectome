"""Estimate an arm's link loading and joint damping from its own commanded movement."""

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .tt29_encoder_delay import predicted_trace


def fit(nominal, commands, lag, observed):
    model = nominal.model
    axes = nominal._arm_ids[1][2]
    driven = model.actuator_trnid[axes, 0]
    # One mass multiplier per driven link, plus the held tool and shared damping.
    links = np.unique(np.r_[model.jnt_bodyid[driven], model.body("learner_paddle").id])
    dofs = model.jnt_dofadr[driven]
    original_mass = model.body_mass[links].copy()
    original_inertia = model.body_inertia[links].copy()
    original_damping = model.dof_damping[dofs].copy()

    def set_parameters(gains):
        model.body_mass[links] = original_mass * gains[:-1]
        model.body_inertia[links] = original_inertia * gains[:-1, None]
        model.dof_damping[dofs] = original_damping * gains[-1]
        mujoco.mj_setConst(model, nominal.data)

    # Windowed encoder changes preserve both response timing and static load response.
    def trace(values):
        return np.concatenate(
            (
                np.diff(values[:, :-1], axis=0).ravel(),
                0.24 * (values[1:, :-1] - values[0, :-1]).ravel(),
            )
        )

    def forecast(gains):
        set_parameters(gains)
        return trace(predicted_trace(nominal, commands, lag)[: len(observed)])

    target = trace(observed)
    base = np.ones(len(links) + 1)
    current = base.copy()
    before = forecast(base)
    for _ in range(2):
        prediction = forecast(current)
        perturb = 0.075
        jac = np.column_stack(
            [
                (forecast(current + step) - prediction) / perturb
                for step in np.eye(len(current)) * perturb
            ]
        )
        # Weakly regularize poorly observable individual links toward their surveyed mass.
        ridge = 0.012
        delta = lsq_linear(
            np.vstack((jac, np.eye(len(current)) * ridge)),
            np.r_[target - prediction, (1.0 - current) * ridge],
            bounds=(0.60 - current, 1.60 - current),
        ).x
        current += delta
    after = forecast(current)
    return dict(
        body_names=[model.body(int(i)).name for i in links],
        mass_gain=current[:-1].tolist(),
        damping_gain=float(current[-1]),
        initial_encoder_rms=float(np.sqrt(np.mean((target - before) ** 2))),
        fitted_encoder_rms=float(np.sqrt(np.mean((target - after) ** 2))),
    )

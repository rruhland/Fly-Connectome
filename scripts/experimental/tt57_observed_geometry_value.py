"""Check whether a body-relative description predicts real missed and returned balls."""

from pathlib import Path

import numpy as np

from . import tt56_scored_motor_atlas as atlas
from . import tt57_preparation_value as base
from .tt4_transfer_trial import _hash, _read, _write


def calibrate():
    examples = []
    for path in atlas.season.fresh.servo.SOURCES:
        doc = _read(path.with_suffix(".json"))
        with np.load(path) as d:
            x, goal, tick = d["states"], d["targets_now"], d["marks"][:, 0]
        off = 0
        for ep in doc["practice"]:
            n = ep["frames"]
            ids = np.arange(off + 6, off + n)
            ids = ids[
                (tick[ids] - tick[ids - 6] == 6) & (tick[ids] - tick[ids - 1] == 1)
            ]
            ref = goal[ids - 1, :5]
            speed = (ref - goal[ids - 6, :5]) / (5 * atlas.mpc.DT)
            for r, v, q in zip(ref, speed, x[ids, 6:11], strict=True):
                examples.append((ep["card"]["id"], r, v, q))
            off += n
    g, r, v, q = map(np.asarray, zip(*examples, strict=True))

    def fit(ids):
        return (
            np.sum(v[ids] * (q[ids] - r[ids]), axis=0)
            / (np.sum(v[ids] ** 2, axis=0) + 1)
        ).clip(-0.15, 0.025)

    error, naive = [], []
    for held in np.array_split(np.unique(g), 5):
        select = np.isin(g, held)
        error.extend((r[select] + fit(~select) * v[select] - q[select]) ** 2)
        naive.extend((r[select] - q[select]) ** 2)
    return (
        fit(np.ones(len(g), bool)),
        np.sqrt(np.mean(error, axis=0)),
        np.sqrt(np.mean(naive, axis=0)),
    )


def flight(p, v, depth):
    now = p.copy()
    vel = v.copy()
    # Ballistic shape features. At a possible table height reversal, use the
    # same nominal bounce for both body-aware and ball-only comparisons.
    eta = np.where(
        v[:, 1] < -0.7,
        np.clip((depth - p[:, 1]) / np.minimum(v[:, 1], -0.7), 0, 0.55),
        0,
    )
    ht = p[:, 2] - 0.78
    bounce = (v[:, 2] + np.sqrt(np.maximum(0, v[:, 2] ** 2 + 19.62 * ht))) / 9.81
    bounced = (bounce > 0) & (bounce < eta) & ((p[:, 1] + v[:, 1] * bounce) > -1.37)
    tb = np.where(bounced, bounce, eta)
    now += vel * tb[:, None]
    now[:, 2] -= 4.905 * tb**2
    vel[:, 2] -= 9.81 * tb
    vel[bounced, 2] *= -0.82
    vel[bounced, :2] *= 0.95
    rem = eta - tb
    now += vel * rem[:, None]
    now[:, 2] -= 4.905 * rem**2
    return now, eta


def geometry(k, shift, ik):
    p, v = k[:, :3] * atlas.P_SCALE, k[:, 3:6] * atlas.V_SCALE
    ref = k[:, 6:11] * atlas.Q_SCALE
    q = ref + k[:, 11:16] * 3.5 * shift
    geom = np.array([ik.pose(np.clip(row, ik.low, ik.high)) for row in q])
    center, normal = geom[:, 0], geom[:, 1]
    mean_center = np.array([0, -1.55, 1.05])
    fixed, tfix = flight(p, v, mean_center[1])
    moving, tmove = flight(p, v, center[:, 1])
    dev = moving - center
    return np.c_[k[:, base.BALL], fixed - mean_center, tfix], np.c_[
        dev, tmove, normal, center, (normal * v).sum(1)
    ]


def run():
    root = base.ROOT
    gfile = root / "geometry-model.npz"
    with np.load(atlas.MODEL) as d:
        x, y, g = d["keys"].astype(float), d["reward"].astype(float), d["release"]
    shift, err, original = calibrate()
    plant = atlas.mpc.arena()
    try:
        opt, body = geometry(x, shift, atlas.mpc.BladeIK(plant))
    finally:
        plant.close()
    names = {
        "optical": opt,
        "body_geometry": np.c_[opt, body],
        "geometry_and_joints": np.c_[opt, body, x[:, 6:16]],
    }
    folds = np.array_split(np.random.default_rng(57312).permutation(np.unique(g)), 5)
    result = {}
    for name, values in names.items():
        p = np.empty(len(x))
        for ids in folds:
            tr = ~np.isin(g, ids)
            p[~tr] = base.Value(values[tr], y[tr]).predict(values[~tr])
        broad = np.mean((p - y) ** 2)
        middle = (x[:, 1] * atlas.P_SCALE[1] > -1.35) & (
            x[:, 1] * atlas.P_SCALE[1] < -0.77
        )
        result[name] = dict(
            brier=float(broad),
            middle_window=float(np.mean((p[middle] - y[middle]) ** 2)),
        )
    final = base.Value(names["body_geometry"], y)
    np.savez_compressed(
        gfile,
        shift=shift,
        mu=final.mu,
        scale=final.scale,
        projection=final.projection,
        w=final.w,
    )
    _write(
        root / "geometry.json",
        dict(
            measured_encoder_naive_rmse=original.tolist(),
            observer_rmse=err.tolist(),
            fitted_seconds=shift.tolist(),
            release_held=result,
            model=_hash(gfile),
            source=_hash(atlas.MODEL),
        ),
    )


if __name__ == "__main__":
    run()

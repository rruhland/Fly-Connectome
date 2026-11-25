"""Does the opponent's recovery state add predictive information when choosing a return?"""

from pathlib import Path

import numpy as np

from . import tt56_scored_motor_atlas as atlas
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt57_opponent_memory")


def assemble():
    meta = _read(atlas.ROOT / "fit.json")
    blocked = {
        c["id"] for k in ("practice", "assessment") for c in _read(atlas.season.PLAN)[k]
    }
    blocked.update(c["id"] for c in _read(atlas.transfer.PLAN)["practice"])
    cases = {}
    for filename, checksum in meta["sources"].items():
        path = Path(filename)
        if _hash(path) != checksum:
            raise ValueError("Changed physical file")
        d = _read(path)
        for c, r in zip(d["cards"], d["rows"]):
            if c["id"] in blocked or c.get("profile") != "reach":
                continue
            offers = r["placements"]["delivered"]
            requests = r["placements"]["requested"]
            notes = [
                n
                for n in r["searches"]
                if n.get("target_errors") and n.get("contact_delay_s")
            ]
            for o, following in zip(offers, offers[1:]):
                leg = o["leg"]
                if (
                    requests[leg]["draw"] != "random"
                    or following["leg"] != leg + 1
                    or not o["near_landed"]
                    or not following.get("receive_line")
                ):
                    continue
                stamp = o["near_contact"]["time"] * 60
                m = next(
                    (
                        n
                        for n in reversed(notes)
                        if 0 < stamp - n["tick"] < 8
                        and abs(n["tick"] + n["contact_delay_s"] * 60 - stamp) < 4
                    ),
                    None,
                )
                if m is None:
                    continue
                carriage = requests[leg + 1].get("carriage")
                if carriage is None:
                    continue
                far = o["far_contact"]
                arc = np.array(m["target_errors"])
                obs = np.r_[m["p"], m.get("robust", 0), m.get("score", 0)]
                history = np.r_[far["racket"][:3], m["tick"] / 60 - far["time"]]
                delivered = following["receive_line"]
                x, y, z = delivered["xyz"]
                vx, vy, vz = delivered["velocity"]
                t = (-1.53 - y) / (0.98 * vy)
                target = np.array(
                    [z + vz * t - 4.905 * t * t, float(following["near_landed"])]
                )
                ident = (
                    c["id"],
                    tuple(np.round(o["near_contact"]["incoming"], 5)),
                    tuple(np.round(arc, 5)),
                    tuple(np.round(history, 5)),
                    tuple(np.round(carriage[:3], 5)),
                )
                if ident not in cases:
                    cases[ident] = (arc, obs, history, carriage[:3], {}, c["id"])
                cases[ident][4][
                    tuple(
                        np.round(
                            following["far_contact"]["incoming"] + following["target"],
                            6,
                        )
                    )
                ] = target
    rows = [
        (a, b, c, d, np.mean(list(t.values()), axis=0), g)
        for a, b, c, d, t, g in cases.values()
    ]
    return tuple(map(np.asarray, zip(*rows, strict=True)))


def design(x, train):
    scale = np.maximum(np.std(x[train], axis=0), 0.015)
    v = np.clip((x - x[train].mean(axis=0)) / scale, -4, 4)
    return np.c_[v, v * v]


def run():
    arc, obs, history, carriage, y, groups = assemble()
    folds = np.array_split(
        np.random.default_rng(571125).permutation(np.unique(groups)), 6
    )
    pred = {
        k: np.empty_like(y)
        for k in ("action", "action_ball", "previous_opponent", "current_carriage")
    }
    for ids in folds:
        tr = ~np.isin(groups, ids)
        a, b, c, d = [design(x, tr) for x in (arc, obs, history, carriage)]
        # Include interactions with the action itself. The opponent's posture can
        # alter which strike to prefer even if it barely changes overall difficulty.
        x = np.c_[np.ones(len(y)), a]
        for name, more in zip(pred, [None, b, c, d], strict=True):
            if more is not None:
                cross = (a[:, :7, None] * more[:, None, : more.shape[1] // 2]).reshape(
                    len(a), -1
                )
                x = np.c_[x, more, cross]
            pen = np.eye(x.shape[1]) * 12
            pen[0, 0] = 0.001
            w = np.linalg.solve(x[tr].T @ x[tr] + pen, x[tr].T @ y[tr])
            pred[name][~tr] = x[~tr] @ w
    result = {}
    for name, p in pred.items():
        mse = (p - y) ** 2
        mbase = (pred["action_ball"] - y) ** 2
        effects = np.array(
            [(mse[groups == j] - mbase[groups == j]).mean(0) for j in np.unique(groups)]
        )
        rng = np.random.default_rng(570697)
        boot = effects[rng.integers(len(effects), size=(4000, len(effects)))].mean(
            axis=1
        )
        result[name] = dict(
            height_rmse=float(np.sqrt(mse[:, 0].mean())),
            next_return_brier=float(mse[:, 1].mean()),
            release_mean_mse_difference_interval=np.quantile(
                boot, [0.025, 0.975], axis=0
            ).tolist(),
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    _write(
        ROOT / "forecast.json",
        dict(
            cases=len(y),
            releases=len(np.unique(groups)),
            metrics=result,
            carriage_xyz_spread_m=np.std(carriage, axis=0).tolist(),
            source=_hash(atlas.ROOT / "fit.json"),
            inference="Offline diagnostic. Referee far-contact position and actual carriage pose at the subsequent near hit are privileged. No learned decision or legal rally is claimed.",
        ),
    )


if __name__ == "__main__":
    run()

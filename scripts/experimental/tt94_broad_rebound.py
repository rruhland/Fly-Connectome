"""Test an optical far-rebound forecast with broader, independent training strokes."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import tt56_scored_motor_atlas as atlas
from . import tt80_prospective_course as new
from . import tt94_optical_rebound as observed
from .tt4_transfer_trial import _hash, _read, _write


def teacher_rows(reserved):
    strokes = {}
    for name, sha in _read(atlas.ROOT / "fit.json")["sources"].items():
        p = Path(name)
        if _hash(p) != sha:
            raise ValueError("Changed training game")
        doc = _read(p)
        for c, r in zip(doc["cards"], doc["rows"]):
            if (
                c["id"] in reserved
                or c.get("profile") != "reach"
                or c.get("opening") != "random"
            ):
                continue
            deliveries = r["placements"]["delivered"]
            events = r["events"]
            bounces = [
                e for e in events if e.get("kind") == "bounce" and e.get("side") == -1
            ]
            outbound = [
                e
                for e in events
                if e.get("collider") == "table" and e["velocity"][1] > 0
            ]
            for prev, nxt in zip(deliveries, deliveries[1:]):
                start = prev.get("near_contact")
                end = nxt.get("far_contact")
                ray = nxt.get("receive_line")
                if not start or not end or not ray or not prev.get("near_landed"):
                    continue
                b = next(
                    (e for e in bounces if start["time"] < e["time"] < end["time"]),
                    None,
                )
                s = next(
                    (e for e in outbound if start["time"] < e["time"] < end["time"]),
                    None,
                )
                if not b or not s:
                    continue
                a = np.r_[b["ball"], b["incoming"]]
                v = np.r_[s["ball"], s["velocity"]]
                pre = observed.estimate([(0, a)], False)
                post = observed.estimate([(0, v)], True)
                if pre is None or post is None:
                    continue
                y = observed.response.projected(np.r_[ray["xyz"], ray["velocity"]])[0]
                strokes.setdefault(
                    (c["id"], *np.round(a, 7), *np.round(v, 7), round(y, 7)),
                    (np.r_[pre, post], y, c["id"]),
                )
    return list(strokes.values())


def study():
    doc = _read(observed.ROOT / "optical.json")
    rows = doc["rows"]
    reserved = (
        set(doc["held_ids"])
        | {r["id"] for r in rows}
        | {r["id"] for r in _read(new.PLAN)["practice"]}
    )
    strokes = teacher_rows(reserved)
    x, y, ids = map(np.array, zip(*strokes, strict=True))
    if set(ids) & reserved:
        raise ValueError("Release overlap")
    yy = np.c_[y, y < 0.99]
    selected = [r for r in rows if r["held"]]
    X = np.concatenate([r["features"] for r in selected])
    groups = np.repeat(np.arange(len(selected)), [r["n"] for r in selected])
    Y = np.asarray([[r["y"], r["risk"]] for r in selected])
    out = {}
    predictors = {}
    for stage, idx in [
        ("prior", None),
        ("before", slice(0, 5)),
        ("after", slice(None)),
    ]:
        if idx is None:
            p = np.tile(yy.mean(0), (len(X), 1))
        else:
            mu = np.median(x[:, idx], axis=0)
            scale = np.maximum(0.045, x[:, idx].std(0))
            z = (x[:, idx] - mu) / scale
            ds, it = cKDTree(z).query((X[:, idx] - mu) / scale, k=48)
            w = np.exp(-2 * (ds / np.maximum(ds[:, -1:], 0.15)) ** 2)
            p = (w[:, :, None] * yy[it]).sum(1) / w.sum(1, keepdims=True)
            if stage == "after":
                predictors.update(mu=mu, scale=scale, z=z, labels=yy)
        p = np.array([p[groups == i].mean(0) for i in range(len(selected))])
        out[stage] = {}
        for grp in ("all", "observations", "difficult"):
            keep = np.array([grp == "all" or r["cohort"] == grp for r in selected])
            out[stage][grp] = dict(
                contexts=int(keep.sum()),
                height_mae_mm=float(np.mean(abs(p[keep, 0] - Y[keep, 0])) * 1000),
                height_corr=float(np.corrcoef(p[keep, 0], Y[keep, 0])[0, 1])
                if idx is not None
                else None,
                low_probability_rmse=float(
                    np.sqrt(np.mean((Y[keep, 1] - p[keep, 1]) ** 2))
                ),
            )
        if stage == "after":
            for r, yp in zip(selected, p, strict=True):
                r["predicted"] = yp.tolist()
    np.savez_compressed(observed.ROOT / "broad.npz", **predictors)
    _write(
        observed.ROOT / "broad.json",
        dict(
            training=len(x),
            releases=len(set(ids)),
            reserved=sorted(reserved),
            sources=_hash(atlas.ROOT / "fit.json"),
            sensor_packet=_hash(observed.ROOT / "optical.json"),
            held=selected,
            results=out,
        ),
    )
    print(out, flush=True)


if __name__ == "__main__":
    study()

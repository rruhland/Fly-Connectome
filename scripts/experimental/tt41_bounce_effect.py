"""Extract a signed, face-relative contact law from actually played solo-ball drops."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_metric_stereo import triangulate
from .tt24_public_body import stationary, stock
from .tt26_bounce_optics import FRAME, free_acceleration, impact_candidates, unit
from .tt29_bounce import COURSE, Decoder, load_checkpoint
from .tt29_bounce_continued import RESULT
from .tt29_bounce_promotion import RETAINED

OUT = Path("artifacts/tt41_bounce")


def gather():
    """There is no dynamics-side contact timestamp, velocity or aim label in this fit."""
    checkpoint = load_checkpoint(RETAINED)
    decoder = Decoder(checkpoint, np.zeros(3))
    a = stock(fixture="table-edge")
    try:
        a.reset(seed=0, render=False)
        eyes = stationary(a)
    finally:
        a.close()
    rows = []
    for name, source in [("initial", COURSE), ("continued", RESULT)]:
        for row in _read(source)["practice"]:
            optics = row["optics"]
            enc = np.asanyarray(optics["enc"])
            ball, gap, _ = triangulate(
                eyes, np.asarray(optics["stereo"], dtype=float), tolerance=0.018
            )
            rows.append(
                {
                    "seed": row["seed"],
                    "source": name,
                    "regime": row["regime"],
                    "joint_angles": enc,
                    "ray_gap": gap,
                    "ball": ball,
                    "reward": row["reward"],
                    "rebounds": row["rebounds"],
                    "actions": np.asarray(row["actions"]),
                }
            )
    acceleration, samples = free_acceleration(rows)
    data = []
    for row in rows:
        q = row["joint_angles"]
        p, rotations = decoder.links.read(q)
        blade = p + decoder.offset[:3] + rotations @ decoder.offset[3:]
        events = impact_candidates(row, acceleration, decoder)
        for h in events:
            k = h["tick"]
            alpha = h["sub_tick"]
            n = rotations[k] @ decoder.normal
            n = unit(n if n[2] > 0.0 else -n)
            # Track the same material point on the face, not merely the disk center.
            # An off-center disk hit gets additional velocity from the wrist rotation.
            point = np.asarray(h["position_m"])
            offset = point - blade[k]
            offset -= (offset @ n) * n
            local = rotations[k].T @ offset
            surface = blade + rotations @ local
            # A local fit around impact reduces phase and encoder noise separately
            # from the before/after optical velocity fit.
            rel = np.arange(-2, 3) - alpha
            speed = np.linalg.lstsq(
                np.c_[np.ones(5), rel * FRAME, (rel * FRAME) ** 2],
                surface[k - 2 : k + 3],
                rcond=None,
            )[0][1]
            before = np.asarray(h["incoming_m_s"])
            after = np.asarray(h["outgoing_m_s"])
            if -((before - speed) @ n) < 0.45 or np.linalg.norm(offset) > 0.071:
                continue
            data.append(
                {
                    "seed": row["seed"],
                    "course": row["source"],
                    "regime": row["regime"],
                    "time": (k + alpha) * FRAME,
                    "first": bool(h is events[0]),
                    "n": n.tolist(),
                    "pre": before.tolist(),
                    "post": after.tolist(),
                    "blade_speed": speed.tolist(),
                    "phase": float(alpha),
                    "radius": float(np.linalg.norm(offset)),
                    "score": row["rebounds"],
                    "residual": h["residual_m"],
                    "q": q[k].tolist(),
                }
            )
    return (
        rows,
        data,
        {"acceleration_m_s2": acceleration.tolist(), "acceleration_samples": samples},
    )


def features(before, speed, normal):
    rel = before - speed
    inward = np.maximum(0.0, -np.sum(rel * normal, axis=-1, keepdims=True))
    radial = inward * normal
    tangent = -rel - radial
    return radial, tangent


def forecast(before, speed, normal, law):
    radial, tangent = features(before, speed, normal)
    return before + law[0] * radial + law[1] * tangent


def estimate(rows, *, score_only=False, nomotion=False):
    rows = [r for r in rows if not score_only or r["score"]]
    if not rows:
        raise ValueError("No depth-resolved and face-aligned impacts")
    x = np.asarray([r["pre"] for r in rows])
    y = np.asarray([r["post"] for r in rows])
    blade = np.asarray([r["blade_speed"] for r in rows])
    n = np.asarray([r["n"] for r in rows])
    radial, tangent = features(x, np.zeros_like(blade) if nomotion else blade, n)
    d = np.stack((radial, tangent), axis=-1)
    weights = np.ones(len(x))
    law = np.array([1.65, 0.2])
    for _ in range(5):
        w = np.repeat(np.sqrt(weights), 3)
        law = np.linalg.lstsq(
            d.reshape(-1, 2) * w[:, None], (y - x).ravel() * w, rcond=None
        )[0]
        law = np.clip(law, [1.05, 0.0], [2.5, 0.9])
        err = np.linalg.norm(x + d @ law - y, axis=1)
        weights = np.minimum(1.0, 0.28 / np.maximum(0.01, err))
    return law


def error(row, law, *, quiet=False):
    args = [np.asarray(row[key]) for key in ("pre", "blade_speed", "n")]
    if quiet:
        args[1] = np.zeros(3)
    return (forecast(*args, law) - row["post"]).tolist()


def profile(res, mode):
    m = np.asarray([r[mode] for r in res])
    lengths = np.linalg.norm(m, axis=1)
    return {
        "median_vector_error_m_s": float(np.median(lengths)),
        "mean_vector_error_m_s": float(np.mean(lengths)),
        "median_upward_error_m_s": float(np.median(abs(m[:, 2]))),
    }


def fit(output):
    t = process_time()
    rolls, rows, extra = gather()
    # Earlier practice estimates first contact. The post-table continuation is
    # a chronological and release-disjoint check, including changed servos.
    source = [r for r in rows if r["course"] == "initial" and r["first"]]
    held = [r for r in rows if r["course"] == "continued" and r["first"]]
    learned = estimate(source)
    still = estimate(source, nomotion=True)
    rated = estimate(source, score_only=True)
    result = [
        dict(
            r,
            fitted=error(r, learned),
            static=error(r, still, quiet=True),
            reward_only=error(r, rated),
            elastic=error(r, [1.85, 0.0]),
        )
        for r in held
    ]
    law = estimate([r for r in rows if r["first"]])
    cloud = [(r["pre"], r["blade_speed"], r["n"]) for r in rows if r["first"]]
    x, u, n = [np.asarray(c) for c in zip(*cloud, strict=True)]
    stats = {
        "closing": [
            float(np.min(-np.sum((x - u) * n, axis=1))),
            float(np.max(-np.sum((x - u) * n, axis=1))),
        ],
        "surface_normal_speed": [
            float(np.min(np.sum(u * n, axis=1))),
            float(np.max(np.sum(u * n, axis=1))),
        ],
    }
    doc = dict(
        schema="tt41-bounce-effect-v1",
        sources={str(p): _hash(p) for p in [COURSE, RESULT, RETAINED]},
        model="v_out = v_in + normal_gain*max(0, -dot(v_in-v_face,n))*n + tangent_gain*project_plane(v_face-v_in)",
        fit_count=len(source),
        check_count=len(held),
        pooled=law.tolist(),
        initial=learned.tolist(),
        still=still.tolist(),
        reward_only=rated.tolist(),
        sample_range=stats,
        **extra,
        results={
            m: profile(result, m)
            for m in ("fitted", "static", "reward_only", "elastic")
        },
        practice_ticks=sum(len(r["actions"]) for r in rolls),
        practice_drops=len(rolls),
        practice_rebounds=sum(r["rebounds"] for r in rolls),
        impacts=rows,
        held=result,
        cpu_s=process_time() - t,
    )
    _write(output, doc)
    print(
        doc["fit_count"],
        doc["check_count"],
        doc["initial"],
        doc["pooled"],
        doc["sample_range"],
        doc["results"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=OUT / "effect.json")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fit(args.out)

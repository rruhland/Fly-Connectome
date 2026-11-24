"""Compare current-state reconstructions to the timely RGB observations on saved motor replays."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt34_latency import OUT, Map


def check(source, bank, output):
    doc = _read(source)
    m = np.load(bank)
    lag = doc["image_frames"]
    delta = m["next"][:, lag] - m["flight"]
    model = Map(m["flight"], delta, np.maximum(0.05, m["flight"].std(axis=0)))
    errors = {k: [] for k in ("stale", "kinematic", "learned")}
    rows = []
    with centered(doc["ready"]):
        arena = replay_arena([])
        renderer, rig, *_ = resources(arena)
        try:
            for c, row in zip(doc["cards"], doc["rows"], strict=True):
                arena.launch(row["shot"], card=c)
                arena.partner.reset(seed=c["seed"])
                eye = Vision(rig, arena.control_dt, raw=True)
                history = []
                start = len(errors["stale"])
                for i, u in enumerate(row["actions"]):
                    if i % 2 == 0:
                        current = eye.update(i, renderer.capture(), arena.joints)
                        history.append((current, eye.eye.latest))
                        if (
                            current is not None
                            and len(history) > lag
                            and history[-lag - 1][0] is not None
                        ):
                            old, flight = history[-lag - 1]
                            for k, x in [
                                ("stale", old),
                                (
                                    "kinematic",
                                    np.r_[*flight.advance(i * arena.control_dt)],
                                ),
                                ("learned", old + model(old)),
                            ]:
                                errors[k].append(
                                    [
                                        float(np.linalg.norm(x[:3] - current[:3])),
                                        float(np.linalg.norm(x[3:] - current[3:])),
                                    ]
                                )
                    arena.exchange(u, FAR_COMMAND)
                if digest_json(row["events"]) != digest_json(arena.events):
                    raise ValueError("Time-matched physical replay changed")
                rows.append({"id": c["id"], "frames": len(errors["stale"]) - start})
        finally:
            renderer.close()
            arena.close()
    summary = {
        k: {
            "median": np.median(v, axis=0).tolist(),
            "mean": np.mean(v, axis=0).tolist(),
        }
        for k, v in errors.items()
    }
    _write(
        output,
        {
            "source_sha256": _hash(source),
            "bank_sha256": _hash(bank),
            "rows": rows,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    check(args.source, OUT / "lag-bank.npz", args.output)

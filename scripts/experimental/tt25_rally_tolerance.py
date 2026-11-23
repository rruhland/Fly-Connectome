"""Measure how far the two learned full-size continuations tolerate launch changes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED
from .tt25_reference_receiver import ReferenceReceiver
from .tt25_recorded_flight import OUTPUT
from .tt25_second_pupil import play


def run(path):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    source = _read(OUTPUT / "recorded-second-pupil.json.gz")
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as f:
        first = [f[k] for k in ("image_keys", "joint_goals", "scale")]
    with np.load(OUTPUT / "recorded-second-pupil.npz", allow_pickle=False) as f:
        memory = [f[k] for k in ("triggers", "negatives", "states", "rates")]
        scale = f["scale"]
    rng = np.random.default_rng(248731)
    arena = ReferenceReceiver(stationary_opponent=False)
    check = ReferenceReceiver()
    arena.max_steps = check.max_steps = 450
    install_rally_opponent(arena, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
    tape = FarTape(arena)
    rows = []
    try:
        for sid in source["donors"]:
            for name, std in [("narrow", (0.005, 0.02, 0.5)), ("wide", (0.03, 0.15, 5.0))]:
                for i in range(4):
                    shot = shots[sid].copy()
                    for prefix, sigma in zip(("pos", "vel", "w_vel"), std, strict=True):
                        for a, d in zip("xyz", rng.normal(0, sigma, 3), strict=True):
                            shot[prefix + "_" + a] += d
                    pair = {}
                    for label, model in [("continuation", memory), ("block", None)]:
                        row = play(arena, shot, model, scale, first, tape)
                        check.launch(shot)
                        for near, far in zip(row["actions"], row["rival_actions"], strict=True):
                            check._aim = check._command_to = lambda *args, far=far: np.asarray(far)
                            _, _, done, cap, _ = check.step(near, render=False)
                        if not (done or cap) or digest_json(check.events) != digest_json(row["events"]):
                            raise ValueError("Changed perturbed-point replay")
                        pair[label] = dict(
                            near=row["near"],
                            far=row["far"],
                            activation=row["activated"],
                            outcome=row["outcome"],
                            referee=digest_json(row["events"]),
                        )
                    rows.append(dict(id=sid, perturbation=name, repeat=i, results=pair))
                    print(sid, name, i, pair["continuation"]["near"], pair["block"]["near"], flush=True)
        _write(
            OUTPUT / "recorded-rally-tolerance.json.gz",
            dict(seed=248731, rows=rows, verified=len(rows) * 2),
        )
    finally:
        arena.close()
        check.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)

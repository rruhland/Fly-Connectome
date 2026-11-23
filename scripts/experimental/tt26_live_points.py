"""Let a physical opponent play out paired fresh feeds with or without bounce experience."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED
from .tt25_second_pupil import play
from .tt26_grip_course import Grip, OUT, PRIOR, ReferenceReceiver, load_match
from .tt26_receiving_speed import draw


def run(path):
    shots, first = load_match(path)
    donors = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"]
    source = _read(OUT / "grip-source.json.gz")["records"]
    with np.load(PRIOR / "recorded-second-pupil.npz", allow_pickle=False) as archive:
        memory = [archive[k] for k in ("triggers", "negatives", "states", "rates")]
        scale = archive["scale"]
    original = ReferenceReceiver(stationary_opponent=False)
    check = ReferenceReceiver()
    original.max_steps = check.max_steps = 450
    grip, other = Grip(original.model), Grip(check.model)
    install_rally_opponent(original, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
    tape = FarTape(original)
    cases = []
    try:
        for src in source:
            angle = src["angle"]
            grip.set(angle)
            other.set(angle)
            rng = np.random.default_rng(1819323)
            for idx in range(24):
                shot = draw(shots, donors, rng, idx)
                results = {}
                for name, bias in (("bounce", src["selected"]), ("cold", 0.0)):
                    goals = first[1].copy()
                    goals[:, 4] += bias
                    rec = play(original, shot, memory, scale, (first[0], goals, first[2]), tape)
                    check.launch(shot)
                    for near, far in zip(rec["actions"], rec["rival_actions"], strict=True):
                        check._aim = check._command_to = lambda *args, far=far: np.asarray(far)
                        _, _, done, cap, _ = check.step(near, render=False)
                    if not (done or cap) or digest_json(rec["events"]) != digest_json(check.events):
                        raise ValueError("Paired point did not replay")
                    results[name] = rec
                cases.append(dict(angle=angle, idx=idx, shot=shot, results=results))
            print(
                angle,
                {
                    k: [
                        sum(r["results"][k][metric] for r in cases if r["angle"] == angle)
                        for metric in ("near", "far")
                    ]
                    for k in ("bounce", "cold")
                },
                flush=True,
            )
            _write(OUT / "live-points.json.gz", dict(cases=cases, verified=2 * len(cases)))
    finally:
        original.close()
        check.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)

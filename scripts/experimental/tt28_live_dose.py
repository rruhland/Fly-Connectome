"""Play complete regulation points using the tool estimate saved after solo practice."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_trial import COACHED
from .tt26_grip_course import Grip, PRIOR, ReferenceReceiver, load_match
from .tt26_match_continuation import archived_step
from .tt26_optical_transfer import decoder, transport
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT as EARLIER, load_bank
from .tt27_full_points import one
from .tt27_outcome_receiver import independent
from .tt27_reply import DEV_DONORS, Flight, PoseMap, examples
from .tt27_reply_recorded import transported
from .tt28_dose import MOUNTS, OUTPUT, target_banks


def games(shots):
    rng = np.random.default_rng(2840129)
    ids = [r["id"] for r in load_bank()["rows"]]
    old = _read(PRIOR / "recorded-live-opponent.json.gz")["conditional_roster"]
    rows = [("previous_opening", shots[i]) for i in old]
    rows += [("practice_variation", draw(shots, ids, rng, j)) for j in range(8)]
    rows += [(c, s) for c, s in independent(shots, seed=285143, near_count=0, unrestricted_count=12)]
    return rows


def run(path, tool, dose):
    lookup, _ = load_match(path)
    arena = ReferenceReceiver(stationary_opponent=False)
    check = ReferenceReceiver()
    control = ReferenceReceiver()
    try:
        arena.max_steps = check.max_steps = 450
        install_rally_opponent(arena, target_profile=_read(COACHED)["metadata"]["opening"]["far_profile"])
        for env in (arena, check):
            Grip(env.model).set(*MOUNTS[tool])
        src = _read(OUTPUT / f"dose-solo-{tool}.json.gz")["checkpoints"][str(dose)]
        # The model used for joint control knows the nominal robot and the estimated rotation.
        # Only the two physical arenas above receive the actual mounting change.
        nominal = Grip(control.model)
        poses = PoseMap(control)
        traces = [
            e for e in examples(_read(EARLIER / "reply-traces.json.gz")["traces"]) if e["id"] in DEV_DONORS
        ]
        memory = transported(traces, nominal, poses, 0.0)
        rig = decoder()
        rot = np.asarray(src["rotation"])
        for e in memory:
            fitted = transport(rig, e["q"], rot, (poses.low, poses.high))
            g, when, rate, lead = e["action"]
            e["action"] = ((np.array(g) + fitted - e["q"]).tolist(), when, rate, lead)
            e["q"] = fitted
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        angle = np.linalg.norm(rotvec)
        nominal.set(angle, rotvec / angle if angle > 1e-8 else (0, 0, 1))
        learned = load_bank()
        goals = np.stack([r["goal"] for r in learned["rows"]])
        initial = (
            np.stack([r["key"] for r in learned["rows"]]),
            target_banks(control, goals, {str(dose): src})[str(dose)],
            np.array(learned["scale"]),
        )
        tape = FarTape(arena)
        flight = Flight()
        rows = []
        for group, s in games(lookup):
            point = one(arena, tape, s, initial, memory, poses, flight, None, "repeat")
            check.launch(s)
            for t, cmd in enumerate(point["actions"]):
                _, _, done, cap, _ = archived_step(check, point["opponent"], t, cmd)
            if not (done or cap) or digest_json(check.events) != digest_json(point["events"]):
                raise ValueError("Full physical point failed replay")
            rows.append(dict(group=group, id=s["id"], shot=s, result=point))
            if len(rows) % 6 == 0:
                print(tool, dose, len(rows), sum(r["result"]["near"] for r in rows), flush=True)
        _write(OUTPUT / f"live-{tool}-{dose}.json.gz", dict(tool=tool, dose=dose, snapshot=src, rows=rows))
    finally:
        arena.close()
        check.close()
        control.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("tool", choices=MOUNTS)
    p.add_argument("dose", type=int, choices=(0, 16, 192))
    args = p.parse_args()
    run(args.feeds, args.tool, args.dose)

"""Replay the record point to time visible cues of its unusually low final ball."""
import argparse
from pathlib import Path

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_opportunity_map import projected_height
from .tt41_anticipate import outbound_projection, physical_projection
from .tt43_outgoing_coverage import ROOT


def collect(source, output, optic):
    d = _read(source)
    if len(d["rows"]) != 1:
        raise ValueError("Expected an isolated one-point tape")
    ready, card, row = d["ready"], d["cards"][0], d["rows"][0]
    leg, target = row["placements"]["delivered"][-2:]
    outgoing, returned = leg["near_contact"], target["far_contact"]
    if not leg["near_landed"] or target["near_landed"] or not target["receive_line"]:
        raise ValueError("Expected the final miss after an answered legal strike")
    frames = []
    with mpc.centered(ready):
        arena = mpc.arena()
        camera, rig, *_ = mpc.resources(arena)
        try:
            eye = mpc.Vision(rig, arena.control_dt, raw=True)
            eye.eye = mpc.LearnedOptics(eye.eye, optic)
            arena.launch(row["shot"], card=card)
            arena.partner.reset(seed=card["seed"])
            dt = arena.control_dt
            start = max(0, outgoing["step"] - 55)
            for tick, u in enumerate(row["actions"]):
                if tick >= start:
                    q = arena.joints
                    x = mpc.observed(eye, tick, camera.capture() if tick % 2 == 0 else None, q)
                    if x is not None and tick % 2 == 0:
                        f = eye.eye.latest
                        frames.append(dict(tick=tick, time=tick * dt, optical=x.tolist(),
                                           bounced=f.bounced, encoders=q.tolist(),
                                           far_bounce=outbound_projection(x, f.bounced),
                                           deep_intercept=physical_projection(x, f.bounced)))
                arena.exchange(u, mpc.FAR_COMMAND)
            if digest_json(arena.events) != digest_json(row["events"]):
                raise RuntimeError("The original physical command tape diverged")
        finally:
            camera.close()
            arena.close()
    out = next((f for f in frames if outgoing["time"] < f["time"] < returned["time"]
                and -1.15 < f["optical"][1] < -.60 and f["far_bounce"]
                and .05 < f["far_bounce"][1] < .65), None)
    received = next((f for f in frames if returned["time"] < f["time"] and f["deep_intercept"]
                     and f["deep_intercept"][1] < .95 and f["deep_intercept"][2] > .04), None)
    o = dict(card=card, role="evaluator times causal optical estimates during the recorded low terminal",
             tape_sha256=_hash(source), optical_sha256=_hash(optic),
             source_events_sha256=digest_json(row["events"]), own=row["score"]["learner"],
             causal_short_outgoing=out, causal_low_incoming=received,
             cue_before_far_contact_s=None if out is None else returned["time"] - out["time"],
             actual_previous_near_hit=outgoing, actual_next_far_hit=returned,
             actual_next_near_bounce=target["near_bounce"],
             actual_final_projected_height_m=projected_height(target["receive_line"]), frames=frames)
    _write(output, o)
    print(o["cue_before_far_contact_s"], o["causal_low_incoming"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=course.ROOT / "random-38-tape.json.gz")
    p.add_argument("--optic", type=Path, default=course.OPTIC)
    p.add_argument("--output", type=Path, default=ROOT / "record-terminal-cues.json")
    args = p.parse_args()
    collect(args.source, args.output, args.optic)

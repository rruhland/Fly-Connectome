"""Verify learned central returns on fresh solver instances and export RGB-earned options."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_central_lesson as lesson
from . import tt54_scored_central_practice as practice
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance, guarded, verify_clearance
from .tt4_transfer_trial import _hash, _read, _write
from .tt54_training_speedups import enabled

mpc = lesson.mpc


def run():
    found = _read(practice.OUT)
    sources = _read(lesson.INPUT)["training"]
    base = {(r["release"], r["dz"], r["dy"]): r for r in _read(lesson.OUT)["rows"]}
    receipt = dict(source=_hash(practice.OUT), rows=[], templates=[])
    seen = set()
    base_draw = lesson.opponent.draw
    for case in found["rows"]:
        win = case["success"]
        if win is None or win["kind"] == "same":
            continue
        # Some in-range clamped goal changes generate the identical solver
        # contact. Count each distinct gained physical ball exactly once.
        key = case["attempts"][-1]["incoming"]
        if key in seen:
            continue
        seen.add(key)
        d0 = next(s for s in sources if s["case"]["release"] == case["release"])
        spec = d0["case"]
        original = _read(Path(spec["source"]))
        r, c = original["rows"][spec["index"]], original["cards"][spec["index"]]
        trial = base[c["id"], case["dz"], case["dy"]]
        tape = (
            r["actions"][: trial["anchor"]]
            + trial["variants"]["phase"]["controls"][: win["cut"] - trial["anchor"]]
            + win["controls"]
        )

        def draw(card, k, protocol, rng):
            goal, tag = base_draw(card, k, protocol, rng)
            if k == case["leg"]:
                return tuple(
                    np.clip(
                        np.array(goal) + [0, case["dy"], case["dz"], 0],
                        lesson.B[:, 0],
                        lesson.B[:, 1],
                    )
                ), "central_practice"
            return goal, tag

        with (
            patch.object(lesson.opponent, "draw", draw),
            mpc.centered(original["ready"]),
            enabled(),
        ):
            a = mpc.arena()
            cad = mpc.arena(detail="cad")
            clear = CourtClearance(cad.model, articulated=True)
            camera, rig, stride, lo, hi = mpc.resources(a)
            ik = mpc.BladeIK(a)
            try:
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                guarded(a, clear)
                eye = mpc.Vision(rig, a.control_dt, raw=True)
                eye.eye = mpc.LearnedOptics(eye.eye, lesson.mined.season.course.OPTIC)
                logs = []
                ref = np.array(original["ready"])
                for t, u in enumerate(tape):
                    q = a.joints.copy()
                    x = mpc.observed(
                        eye, t, camera.capture() if t % 2 == 0 else None, q
                    )
                    sent = np.clip(ref + np.asarray(u) * stride, lo, hi)
                    if t >= spec["start"]:
                        logs.append(
                            dict(
                                tick=t,
                                q=q,
                                ref=ref.copy(),
                                x=None if x is None else x.copy(),
                                next_ref=sent.copy(),
                            )
                        )
                    a.exchange(u, mpc.FAR_COMMAND)
                    ref = sent
                actual = lesson.native_result(a, case["leg"])
                if (
                    not actual["answered"]
                    or digest_json(a.events) != digest_json(win["events"])
                    or actual["incoming"] != key
                ):
                    raise ValueError(
                        "Full physical replay differs from the claimed earned return"
                    )
                verify_clearance(a)
                learned = lesson.mined.school.extract(
                    logs, ik, f"earned-center-{c['id']}-{case['index']}", spec["start"]
                )
                if learned is None:
                    raise ValueError("No observable scored motor phase")
                observed = [
                    v
                    for v in logs
                    if learned["source_phase_tick"] - 65
                    <= v["tick"]
                    <= learned["source_phase_tick"] + 5
                ]
                receipt["templates"].append(
                    dict(
                        template=lesson.mined.school.plain(learned),
                        approach=lesson.mined.school.plain(observed),
                        release=c["id"],
                    )
                )
                receipt["rows"].append(
                    dict(
                        release=c["id"],
                        goal=[case["dz"], case["dy"]],
                        actual=actual,
                        kind=win["kind"],
                        controls=len(tape),
                        events=digest_json(a.events),
                        clearance=a.clearance_result(),
                        command_tape=tape,
                    )
                )
            finally:
                camera.close()
                cad.close()
                a.close()
    _write(lesson.ROOT / "central-earned-replay.json.gz", receipt)
    print(
        [
            (r["actual"]["physical_x"], r["actual"]["physical_z"], r["clearance"])
            for r in receipt["rows"]
        ],
        flush=True,
    )


if __name__ == "__main__":
    run()

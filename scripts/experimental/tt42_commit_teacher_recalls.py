"""Explain teacher archive crowding on the same physically re-observed motor context."""

import argparse
from pathlib import Path

import numpy as np

from . import tt38_joint_mpc as mpc
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt42_commit_teacher import OUT
from .tt42_commit_teacher_student import FRAGMENTS, OPTIC, WithTeacher


def describe(bank, x, q, ref, geometry):
    def offers(self):
        self.use = True
        self.teacher = bank
        return list(WithTeacher.offers(self, x, q, ref))

    # Teacher proposal generation only needs its sensory bank, not a live actor.
    class Retrieval:
        pass

    rows = []
    for seq, (source, contact), distance in offers(Retrieval())[:6]:
        end = min(len(seq) - 1, max(2, int((-1.54 - x[1]) / x[4] / mpc.DT)))
        center, n, j, _ = geometry.pose(seq[end, :5], jacobian=True)
        # Intended signed face kinematics from the requested reference curve.
        # These are not native player outcomes or authorization.
        speed = j @ (seq[end, :5] - seq[end - 1, :5]) / mpc.DT
        rows.append(
            {
                "teacher_practice_point": source,
                "learned_contact_centiticks": contact,
                "normalized_context_distance": distance,
                "requested_center_m": center.tolist(),
                "requested_incident_normal": n.tolist(),
                "requested_deck_velocity_mps": speed.tolist(),
                "reference_curve_sha256": digest_json(seq.tolist()),
                "references": seq.tolist(),
            }
        )
    return rows


def run(a):
    source = _read(a.tape)
    card, game = next(
        (c, r)
        for c, r in zip(source["cards"], source["rows"], strict=True)
        if c["id"] == 3401
    )
    banks = {
        key: OUT / f"earned-camera{suffix}.npz"
        for key, suffix in (("pilot", ""), ("full", "-41"))
    }
    model = None
    with mpc.centered(source["ready"]):
        arena = mpc.arena(protocol="random")
        camera, rig, stride, lo, hi = mpc.resources(arena)
        arena.launch(game["shot"], card=card)
        arena.partner.reset(seed=card["seed"])
        v = mpc.Vision(rig, arena.control_dt, raw=True)
        v.eye = mpc.LearnedOptics(v.eye, OPTIC)
        reference = np.asarray(source["ready"]).copy()
        rows = []
        try:
            model = mpc.MPC(mpc.BANK, None, variant="elastic", budget=28)
            templates = {k: FRAGMENTS(p, model.engine) for k, p in banks.items()}
            for tick, action in enumerate(game["actions"][:170]):
                q = arena.joints.copy()
                picture = camera.capture() if tick % 2 == 0 else None
                x = mpc.observed(v, tick, picture, q)
                if tick in (159, 164, 169):
                    old = next(s for s in game["searches"] if s["tick"] == tick)
                    recalled = next(
                        s for s in game["recall_attempts"] if s["tick"] == tick
                    )
                    if not np.array_equal(x[:3], old["p"]):
                        raise ValueError("RGB/encoder classification replay changed")
                    ranks = {
                        k: describe(b, x, q, reference, model.geometry)
                        for k, b in templates.items()
                    }
                    if [
                        s["normalized_context_distance"] for s in ranks["pilot"][:2]
                    ] != [e[2] for e in recalled["query"] if e[0] == "admitted"]:
                        raise ValueError(
                            "Restored optical and encoder lookup is not exact"
                        )
                    rows.append(
                        {
                            "tick": tick,
                            "camera_position_and_velocity": x.tolist(),
                            "encoders": q.tolist(),
                            "active_reference": reference.tolist(),
                            "underlying_treatment_result": old.get("chosen")
                            or old.get("prepare"),
                            "ranks": ranks,
                        }
                    )
                    print(
                        tick,
                        {
                            k: [
                                (
                                    r["teacher_practice_point"],
                                    r["learned_contact_centiticks"],
                                    round(r["normalized_context_distance"], 3),
                                )
                                for r in z[:2]
                            ]
                            for k, z in ranks.items()
                        },
                        flush=True,
                    )
                arena.exchange(action, mpc.FAR_COMMAND)
                reference = np.clip(reference + stride * np.asarray(action), lo, hi)
        finally:
            if model is not None:
                model.close()
            camera.close()
            arena.close()
    _write(
        a.out,
        {
            "role": "retrospective rank audit, not an additional held evaluation",
            "source_sha256": _hash(a.tape),
            "bank_sha256": {k: _hash(p) for k, p in banks.items()},
            "rows": rows,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tape", type=Path, default=OUT / "course-pilot.json.gz")
    p.add_argument("--out", type=Path, default=OUT / "3401-recall-ranks.json.gz")
    run(p.parse_args())

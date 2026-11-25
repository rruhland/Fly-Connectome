"""Practice progressively lower central deliveries around genuinely rewarded camera-controlled strokes."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as opponent
from . import tt54_earned_low_flights as mined
from .tt19_view_full_game import digest_json
from .tt39_opportunity_map import projected_height
from .tt44_paired_credit import State
from .tt4_transfer_trial import _hash, _read, _write

mpc = mined.mpc
ROOT = mined.ROOT
B = np.array(opponent.PROFILES["reach"])
INPUT = ROOT / "central-first3.json"
OUT = ROOT / "central-lesson.json.gz"
SCALE = np.array([0.065, 0.13, 0.045, 0.8, 1.1, 0.7])
OFFSETS = [
    (0.0, 0.0),
    (-0.005, 0.0),
    (-0.012, 0.0),
    (-0.020, 0.0),
    (-0.032, 0.0),
    (-0.015, 0.022),
    (-0.022, 0.043),
]


def snapshot():
    raw = _read(ROOT / "central-templates.json")
    rows = raw["rows"][:3]
    if len(rows) != 3 or any(
        r["case"] != c
        for r, c in zip(
            rows, _read(ROOT / "central-sources.json")["rows"][:3], strict=True
        )
    ):
        raise ValueError(
            "First three preassigned independent sources not yet available"
        )
    _write(INPUT, dict(training=rows, offsets=OFFSETS, plan=raw["plan"]))


class Motion:
    def __init__(self, approach):
        # Input comprises only observed flight, own encoders and own target
        # positions. Archive time is converted to sequence order on ingestion.
        self.x = np.array(
            [np.full(6, np.nan) if s["x"] is None else s["x"] for s in approach]
        )
        self.q = np.array([s["q"][:5] for s in approach])
        self.ref = np.array([s["ref"] for s in approach])
        self.next = np.array([s["next_ref"] for s in approach])


class Tracking:
    def __init__(self, library):
        self.library = library
        self.motion = None
        self.index = None

    def target(self, x, q, ref):
        if self.motion is None:
            if x is None or x[4] >= -1 or not -1.25 < x[1] < 0.18:
                return None
            best = np.inf
            for motion in self.library:
                ids = np.flatnonzero(
                    np.isfinite(motion.x).all(axis=1) & (motion.x[:, 4] < -1)
                )
                if len(ids) == 0:
                    continue
                cost = np.sum(((motion.x[ids] - x) / SCALE) ** 2, axis=1)
                cost += np.sum(((motion.q[ids] - q[:5]) / 0.24) ** 2, axis=1)
                cost += np.sum(((motion.ref[ids, :5] - ref[:5]) / 0.18) ** 2, axis=1)
                ix = int(np.argmin(cost))
                if cost[ix] < best:
                    self.motion, self.index, best = motion, ids[ix], cost[ix]
        else:
            m = self.motion
            ids = np.arange(self.index, min(self.index + 4, len(m.x)))
            expected = min(self.index + 1, len(m.x) - 1)
            self.index = expected
            if x is not None and x[4] < -1:
                ids = ids[np.isfinite(m.x[ids]).all(axis=1) & (m.x[ids, 4] < -1)]
                if len(ids):
                    c = (
                        np.sum(((m.x[ids] - x) / SCALE) ** 2, axis=1)
                        + 0.18 * (ids - expected) ** 2
                    )
                    c += np.sum(((m.q[ids] - q[:5]) / 0.24) ** 2, axis=1)
                    self.index = ids[int(np.argmin(c))]
        return self.motion.next[self.index] if self.motion is not None else None


def native_result(arena, leg):
    got = arena.point_placements()["delivered"]
    if len(got) <= leg:
        return dict(offer=False, legal=False, answered=False)
    d = got[leg]
    line = d["receive_line"]
    xx = zz = None
    if line:
        t = (-1.53 - line["xyz"][1]) / (0.98 * line["velocity"][1])
        xx = line["xyz"][0] + t * line["velocity"][0]
        zz = projected_height(line)
    return dict(
        offer=True,
        legal=bool(d["near_landed"]),
        answered=len(got) > leg + 1 and bool(got[leg + 1]["near_bounce"]),
        physical_x=xx,
        physical_z=zz,
        touched=bool(d["near_contact"]),
        incoming=digest_json([d["far_contact"], d["near_bounce"]]),
    )


def run():
    src = _read(INPUT)
    library = [Motion(s["approach"]) for s in src["training"]]
    result = dict(plan=_hash(INPUT), rows=[], complete=False)
    base_draw = opponent.draw
    for example in src["training"]:
        spec = example["case"]
        path = Path(spec["source"])
        if _hash(path) != spec["source_sha256"]:
            raise ValueError("Source changed")
        d = _read(path)
        c, r = d["cards"][spec["index"]], d["rows"][spec["index"]]
        leg = spec["leg"]
        origin = r["placements"]["requested"][leg]
        anchor = int(np.floor(origin["time_s"] / mpc.DT + 1e-7))
        with mpc.centered(d["ready"]):
            arena = mpc.arena()
            camera, rig, stride, lo, hi = mpc.resources(arena)
            try:
                arena.launch(r["shot"], card=c)
                arena.partner.reset(seed=c["seed"])
                eye = mpc.Vision(rig, arena.control_dt, raw=True)
                eye.eye = mpc.LearnedOptics(eye.eye, mined.season.course.OPTIC)
                ref = np.asarray(d["ready"])
                for t, u in enumerate(r["actions"][:anchor]):
                    mpc.observed(
                        eye, t, camera.capture() if t % 2 == 0 else None, arena.joints
                    )
                    ref = np.clip(ref + np.asarray(u) * stride, lo, hi)
                    arena.exchange(u, mpc.FAR_COMMAND)
                if len(arena.partner.requests) != leg or digest_json(
                    arena.events
                ) != digest_json([e for e in r["events"] if e["step"] <= anchor]):
                    raise ValueError("Shared physical prefix changed")
                frozen = State(arena)
                for dz, dy in src["offsets"]:

                    def goal(card, k, protocol, rng):
                        target, tag = base_draw(card, k, protocol, rng)
                        if k != leg or (dz == 0 and dy == 0):
                            return target, tag
                        return tuple(
                            np.clip(
                                np.asarray(target) + [0, dy, dz, 0], B[:, 0], B[:, 1]
                            )
                        ), "central_practice"

                    arms = {}
                    for mode in ("blind", "phase"):
                        frozen.restore(arena)
                        subeye = deepcopy(eye)
                        reference = ref.copy()
                        policy = Tracking(library)
                        controls = []
                        changed = None
                        with patch.object(opponent, "draw", goal):
                            for t in range(anchor, anchor + 230):
                                q = arena.joints
                                x = (
                                    mpc.observed(
                                        subeye,
                                        t,
                                        camera.capture()
                                        if mode == "phase" and t % 2 == 0
                                        else None,
                                        q,
                                    )
                                    if mode == "phase"
                                    else None
                                )
                                archived = np.array(
                                    r["actions"][t]
                                    if t < len(r["actions"])
                                    else np.zeros(6)
                                )
                                desired = (
                                    policy.target(x, q, reference)
                                    if mode == "phase"
                                    else None
                                )
                                u = (
                                    archived
                                    if desired is None
                                    else np.clip(
                                        (np.clip(desired, lo, hi) - reference) / stride,
                                        -0.8,
                                        0.8,
                                    )
                                )
                                if (
                                    not np.allclose(u, archived, rtol=0, atol=1e-10)
                                    and changed is None
                                ):
                                    changed = t
                                reference = np.clip(reference + u * stride, lo, hi)
                                _, _, done, capped, info = arena.exchange(
                                    u, mpc.FAR_COMMAND
                                )
                                controls.append(u.tolist())
                                if (
                                    done
                                    or capped
                                    or any(
                                        e["kind"] == "legal_return" and e["side"] == -1
                                        for e in info["events"]
                                    )
                                    and len(arena.point_placements()["delivered"])
                                    > leg + 1
                                ):
                                    break
                        score = native_result(arena, leg)
                        if mode == "blind" and dz == dy == 0 and not score["answered"]:
                            raise ValueError(
                                "Original natural earned return failed to reproduce"
                            )
                        arms[mode] = dict(
                            **score,
                            first_changed=changed,
                            controls=controls,
                            events=arena.events.copy(),
                        )
                    if arms["blind"].get("incoming") != arms["phase"].get("incoming"):
                        raise ValueError("Different actual incoming balls")
                    result["rows"].append(
                        dict(
                            release=c["id"],
                            leg=leg,
                            anchor=anchor,
                            dz=dz,
                            dy=dy,
                            variants=arms,
                        )
                    )
                    _write(OUT, result)
                    print(
                        c["id"],
                        dz,
                        dy,
                        [
                            (
                                k,
                                r["physical_x"],
                                r["physical_z"],
                                r["legal"],
                                r["answered"],
                            )
                            for k, r in arms.items()
                        ],
                        flush=True,
                    )
            finally:
                camera.close()
                arena.close()
    result["complete"] = True
    _write(OUT, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("snapshot", "run"))
    a = p.parse_args()
    snapshot() if a.mode == "snapshot" else run()

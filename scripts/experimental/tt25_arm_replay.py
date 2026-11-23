"""Audit moving-arm/table clearance and actual drive speeds on archived point tapes."""

import gzip
from hashlib import sha256
import json

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, SequenceArena
from .tt25_body_contacts import MOVING, enable_table_contacts
from .tt25_recorded_flight import OUTPUT, ROOT


# Feetech STS3215-C001, 7.4 V, 0.192 s / 60 degrees, no load.
# A fast impact or passive backdrive can exceed the nominal motor-only speed.
NOMINAL_NO_LOAD = np.deg2rad(60) / .192
class SolidArm(SequenceArena):
    def _model_document(self):
        return enable_table_contacts(super()._model_document())


def run(solid=False):
    arena = (SolidArm if solid else SequenceArena)(**CONFIG)
    cases = []
    tapes = []
    try:
        table = arena.model.geom("table").id
        geoms = [i for i, b in enumerate(arena.model.geom_bodyid) if arena.model.body(b).name in MOVING]
        if not geoms:
            raise ValueError("Missing player link meshes")
        axes = [arena.model.joint("learner_" + axis).dofadr[0] for axis in
                ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
        for i in range(2):
            path = ROOT / f"artifacts/tt24/rolling-match-inspected-{i}.json.gz"
            tapes.append(dict(file=path.name, sha256=sha256(path.read_bytes()).hexdigest()))
            for item in json.loads(gzip.decompress(path.read_bytes()))["cases"]:
                row = item["results"]["calibrated"]
                arena.lane = row["lane"]
                arena.reset(seed=row["seed"], server=-1, render=False)
                speeds, distances = [], []
                worst_geom, worst_distance = None, float("inf")
                for near, far in zip(row["actions"], row["rival_actions"], strict=True):
                    rates = np.asarray(far)
                    arena._aim = arena._command_to = lambda *args, rates=rates: rates
                    _, reward, done, cap, _ = arena.step(near, render=False)
                    speeds.append(max(abs(arena.data.qvel[axes])))
                    # mj_step leaves geom poses from the start of its final solver substep.
                    signed = [mujoco.mj_geomDistance(arena.model, arena.data, g, table, .6, None) for g in geoms]
                    smallest = int(np.argmin(signed))
                    distances.append(signed[smallest])
                    if signed[smallest] < worst_distance:
                        worst_distance = signed[smallest]
                        worst_geom = geoms[smallest]
                    if done or cap:
                        break
                if not solid and (not (done or cap) or reward != row["reward"] or digest_json(arena.events) != digest_json(row["events"])):
                    raise ValueError("Point diverged from qualified tape")
                cases.append(dict(seed=row["seed"], returns=arena.point_result()["legal_returns"], ticks=len(speeds),
                                  terminal=bool(done or cap), available_ticks=len(row["actions"]),
                                  ticks_above_nominal=sum(bool(s > NOMINAL_NO_LOAD) for s in speeds),
                                  max_observed_speed_rad_s=float(max(speeds)),
                                  min_arm_table_signed_cm=float(worst_distance * 100),
                                  deepest_body=arena.model.body(arena.model.geom_bodyid[worst_geom]).name,
                                  ticks_below_minus_5mm=sum(x < -.005 for x in distances)))
    finally:
        arena.close()
    receipt = dict(source=tapes, cases=cases, nominal_rad_s=NOMINAL_NO_LOAD,
                   checked_point_tapes=len(cases), contact_enabled=solid, evaluated_cad_bodies=sorted(MOVING),
                   clearance_query="MuJoCo convex geom distance at control ticks; table only; excludes held paddle",
                   ticks=sum(r["ticks"] for r in cases),
                   below_minus_5mm=sum(r["ticks_below_minus_5mm"] for r in cases),
                   above_motor_nominal=sum(r["ticks_above_nominal"] for r in cases))
    (OUTPUT / ("recorded-arm-solid.json" if solid else "recorded-arm-audit.json")).write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(receipt["ticks"], receipt["below_minus_5mm"], receipt["above_motor_nominal"])


if __name__ == "__main__":
    run()
    run(solid=True)

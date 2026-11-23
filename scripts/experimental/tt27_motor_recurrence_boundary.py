"""Construct a sign change invisible to the three-port recurrence audit."""

from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt24_rally_trial import PublicBody
from .tt26_bounce_effects import JointEffects
from .tt27_motor_active import calibrate_passive
from .tt27_motor_active_continuous import ContinuousReceiver
from .tt27_motor_interface import AXIS_COUNT, Candidates, Interface
from .tt27_motor_recurrence import OUT as MAIN, SOURCE_SEED, Templates, identify, score, signature
from .tt27_world_camera import FEEDS, POSES


OUT = Path("artifacts/tt27/motor-recurrence-boundary.json.gz")


def interface(value):
    return Interface(tuple(value["permutation"]), tuple(value["signs"]), value["delay"])


def ordinary_flag(row, motor, fitted, possible):
    scores = motor.loss(dict(requests=row["requests"][:16], encoders=row["encoders"][:16]))
    candidate = motor.maps.index(fitted)
    usable = [motor.maps.index(rule) for rule in possible]
    best = np.min(scores[:, usable], axis=1)
    difference = scores[:, candidate] - best
    established = np.flatnonzero((difference[1:] >= 9.) & (difference[:-1] >= 9.))
    return dict(elapsed_ticks=None if not len(established) else int(established[0] + 2),
                residual_margin=difference.tolist())


def run():
    pilot = _read(MAIN)["streams"][0]["held"][1]
    expected = interface(pilot["actual"])
    prior = [interface(row) for row in pilot["prior"]]
    seen = {row["port"] for row in pilot["active_probe"]["probes"]}
    moves = np.abs([c for row in pilot["rows"] for c in row["active"]["requests"]]).sum(axis=0)
    hidden = max((k for k in range(AXIS_COUNT) if k not in seen), key=lambda k: moves[k])
    sign = list(expected.signs)
    sign[expected.permutation.index(hidden)] *= -1
    actual = Interface(expected.permutation, tuple(sign), expected.delay)
    old = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    with np.load(POSES, allow_pickle=False) as data:
        memory = tuple(data[k] for k in ("keys", "goals", "scale"))
    arena = ContinuousReceiver()
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    body = PublicBody(arena)
    motor = Candidates(JointEffects.load(), body.lower[:AXIS_COUNT], body.upper[:AXIS_COUNT], arena.control_dt)
    try:
        bank = Templates(arena, dofs)
        calibrate_passive(arena, dofs, old, motor)
        wrong, audit = identify(arena, dofs, bank, motor.maps, actual, SOURCE_SEED, prior=prior)
        nominal, reference = identify(arena, dofs, bank, motor.maps, expected, SOURCE_SEED, prior=prior)
        assert wrong == nominal == expected and wrong != actual
        assert all(audit["probes"][i]["encoder"] == reference["probes"][i]["encoder"]
                   for i in range(len(audit["probes"])))
        allowed = [x for x in motor.maps if all(signature(x, item["port"]) == tuple(item["signature"])
                                               for item in audit["probes"])]
        results = []
        recovered, exhaustive = None, None
        switched_before_feed = None
        for index, item in enumerate(pilot["rows"]):
            if index and recovered is None and results[-1]["encoder_flag"]["elapsed_ticks"] is not None:
                recovered, exhaustive = identify(arena, dofs, bank, motor.maps, actual, SOURCE_SEED)
                switched_before_feed = index + 1
            shot = item["shot"]
            before = score(arena, dofs, actual, wrong, audit, shot, memory, SOURCE_SEED, capture=True)
            clear = score(arena, dofs, expected, nominal, reference, shot, memory, SOURCE_SEED, capture=True)
            online = before if recovered is None else score(
                arena, dofs, actual, recovered, exhaustive, shot, memory, SOURCE_SEED, capture=True)
            results.append(dict(shot=shot, invisible_sign_flip=before, nominal=clear, online=online,
                                encoder_flag=ordinary_flag(before, motor, wrong, allowed),
                                nominal_flag=ordinary_flag(clear, motor, wrong, allowed)))
        _write(OUT, dict(main_sha256=_hash(MAIN), flipped_request_port=hidden, queried_ports=sorted(seen),
                         actual_sign_flip=actual.__dict__, assumed=wrong.__dict__,
                         exact_probe_observations=True, physically_compatible_maps=len(allowed),
                         full_probe_before_feed=switched_before_feed,
                         recovered_exact=recovered is not None and recovered == actual,
                         full_probe_port_count=None if exhaustive is None else len(exhaustive["probes"]),
                         results=results))
        print([(r["invisible_sign_flip"]["legal"], r["nominal"]["legal"], r["online"]["legal"],
                r["encoder_flag"]["elapsed_ticks"], r["nominal_flag"]["elapsed_ticks"]) for r in results], flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    run()

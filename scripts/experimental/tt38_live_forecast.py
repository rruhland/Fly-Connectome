"""Replay actually issued joint controls to measure how early cameras can identify a bad return."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt31_ready import centered
from .tt37_motor_clinic import AIM
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics


def trial(engine, controls, cutoff):
    residual = np.zeros((1, 9))
    _, result = engine.trajectories(controls[None], AIM, cutoff, feedback=residual)
    return {
        "legal": bool(np.isfinite(result[0, 0])),
        "gap": float(residual[0, 0]),
        "contact": bool(residual[0, 1]),
        "outgoing": residual[0, 2:5].tolist(),
    }


def evaluate(source, vision, out, n):
    doc = _read(source)
    rows = []
    with centered(doc["ready"]):
        a = arena()
        twin = OpticalClinic(body=True)
        native = Rollout(a, 1)
        renderer, rig, stride, lo, hi = joint.resources(a)
        try:
            for card, r in zip(doc["cards"][:n], doc["rows"][:n], strict=True):
                a.launch(r["shot"], card=card)
                a.partner.reset(seed=card["seed"])
                eye = joint.Vision(rig, a.control_dt)
                eye.eye = LearnedOptics(eye.eye, vision)
                twin.previous = None
                twin.bounced = False
                ref = np.asarray(doc["ready"])
                commands = np.array(r["actions"])
                goals = np.empty_like(commands)
                for i, c in enumerate(commands):
                    ref = np.clip(ref + c * stride, lo, hi)
                    goals[i] = ref
                times = defaultdict(list)
                events = r["events"]
                offered = False
                for j, e in enumerate(events):
                    if e["kind"] == "contact" and e["side"] == -1:
                        offered = True
                    elif e["kind"] == "contact" and e["side"] == 1 and offered:
                        offered = False
                        legal = any(
                            ev["kind"] == "legal_return" and ev["side"] == 1
                            for ev in events[
                                j + 1 : next(
                                    (
                                        k
                                        for k in range(j + 1, len(events))
                                        if events[k]["kind"] == "contact"
                                    ),
                                    len(events),
                                )
                            ]
                        )
                        for lead in (13, 9, 6, 4, 2):
                            tick = int(np.floor(e["time"] / a.control_dt)) - lead
                            if tick >= 0:
                                times[tick].append((e["time"], legal, lead))
                ref = np.array(doc["ready"])
                for i, c in enumerate(commands):
                    eye.update(i, renderer.capture() if i % 2 == 0 else None, a.joints)
                    twin.observe(i, eye.eye, a.joints, ref)
                    if i in times and twin.sync():
                        for impact, legal, lead in times[i]:
                            cutoff = impact - i * a.control_dt + 0.14
                            controls = goals[i : i + 95]
                            if len(controls) < 95:
                                controls = np.vstack(
                                    (
                                        controls,
                                        np.tile(controls[-1], (95 - len(controls), 1)),
                                    )
                                )
                            exact = trial(native, controls, cutoff)
                            opt = trial(twin, controls, cutoff)
                            data = {
                                "card": card["id"],
                                "tick": i,
                                "lead": lead,
                                "legal": legal,
                                "native": exact,
                                "optical": opt,
                                "p_error": (a.ball - twin.twin.ball).tolist(),
                                "v_error": (a.velocity - twin.twin.velocity).tolist(),
                            }
                            rows.append(data)
                    a.exchange(c, joint.FAR_COMMAND)
                    ref = goals[i]
                if digest_json(a.events) != digest_json(events):
                    raise ValueError(
                        "Issued actions no longer reproduce scored physics"
                    )
                _write(out, {"source": _hash(source), "vision": _hash(vision), "rows": rows})
        finally:
            renderer.close()
            twin.twin.close()
            a.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("vision", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=12)
    x = p.parse_args()
    evaluate(x.source, x.vision, x.out, x.n)

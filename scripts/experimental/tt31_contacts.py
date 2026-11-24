"""Measure which physical face and side the arm used in an exact action replay."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_conditions import cooperative_targets
from .tt29_endurance import audit
from .tt29_partner_fast import FastPartnerArena


class BladeContacts:
    """Observer called only when the live solver registers ball-to-blade contact."""

    def __init__(self, arena):
        self.arena = arena
        self.rows = []
        self.original = arena._racket_contact

    def observe(self, side, location, incoming):
        if side == 1:
            a = self.arena
            blade = a._racket_ids[1]
            ball = a._geom_ids["ball"]
            contacts = [c for c in a.data.contact if c.efc_address >= 0 and set(c.geom) == {ball, blade}]
            contact = min(contacts, key=lambda c: np.linalg.norm(c.pos - location))
            outward = contact.frame[:3] * (1 if contact.geom[0] == blade else -1)
            normal = a.data.geom_xmat[blade].reshape(3, 3)[:, 2]
            alignment = float(outward @ normal)
            shoulder = a.data.xpos[a.model.body("learner_shoulder").id]
            self.rows.append(dict(time=float(a.data.time), xyz=list(location), side_x_m=float(location[0] - shoulder[0]),
                                  face_alignment=alignment, face="rim" if abs(alignment) < .8 else
                                  ("positive" if alignment > 0 else "negative"),
                                  joints_rad=a.data.qpos[a._arm_ids[1][0][:5]].tolist()))
        return self.original(side, location, incoming)

    def scored(self, events, *, opening_side=-1):
        landings = {r["hit"] for r in audit(events, opening_side=opening_side)["shots"] if r["side"] == 1}
        contacts = [dict(r, landed=r["time"] in landings) for r in self.rows]
        if sum(r["landed"] for r in contacts) != len(landings):
            raise ValueError("A legal landing lacked its matching physical blade contact")
        return contacts


def record(arena, row, *, opening_side=-1):
    """Call after restoring the archived launch and opponent, before taking actions."""
    observer = BladeContacts(arena)
    far = np.zeros(5)
    with patch.object(arena, "_racket_contact", observer.observe):
        for action in row["actions"]:
            arena.exchange(action, far)
    if digest_json(arena.events) != digest_json(row["events"]):
        raise ValueError("Face replay diverged from archived game events")
    return observer.scored(arena.events, opening_side=opening_side)


def summarize(rows):
    legal = [c for row in rows for c in row["contacts"] if c["landed"]]
    positions = [c["side_x_m"] for c in legal]
    return dict(legal=len(legal), by_face=dict(Counter(c["face"] for c in legal)),
                left_center_right=dict(Counter("left" if x < -.03 else "right" if x > .03 else "center"
                                               for x in positions)),
                lateral_range_m=[min(positions), max(positions)] if legal else None)


def run(source, output):
    data = _read(source)
    if data.get("condition", "stock") != "stock" or data.get("pose", "lateral") != "lateral":
        raise ValueError("This replay entry point expects the stock lateral SO-101")
    targets = cooperative_targets("lateral", data.get("spread", "varied"))
    arena = FastPartnerArena(strike_ready="lateral", fast_search=True, clearance_guard=True, velocity_guard=True,
                             rate_scale=.8, motor_envelope=True, guard_table=True, max_steps=30000, targets=targets,
                             net_rescue=data.get("net_rescue", True), robust_contact=data.get("robust_contact", True))
    rows = []
    try:
        for row in data["rows"]:
            arena.launch(row["shot"])
            arena.partner.reset(seed=row["opponent_seed"])
            contacts = record(arena, row)
            rows.append(dict(id=row["shot"]["id"], seed=row["opponent_seed"], contacts=contacts))
    finally:
        arena.close()
    result = summarize(rows)
    _write(output, dict(source_sha256=_hash(source), summary=result, rows=rows))
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.source, args.output)

"""After-the-fact feasibility check of missed wide returns at identical physical states."""
import argparse
from pathlib import Path
from hashlib import sha256

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_action_memory import RallyObjective
from .tt29_conditions import cooperative_targets
from .tt29_partner_fast import FastPartnerArena, NativeBallFlight
from .tt29_rally_physics import MotionPlanner


def check(source, output):
    data = _read(source)
    if data["condition"] != "stock" or data["spread"] != "varied" or not data["motor_envelope"]:
        raise ValueError("Expected the coupled, varied, stock-physics comparison")
    arena = FastPartnerArena(strike_ready=data["pose"], fast_search=True, clearance_guard=True,
                             velocity_guard=True, rate_scale=.8, motor_envelope=True, guard_table=True,
                             max_steps=9000, targets=cooperative_targets(data["pose"], "varied"),
                             net_rescue=data.get("net_rescue", False), robust_contact=True)
    rows = []
    far = np.zeros(5)
    try:
        for r in data["rows"]:
            if r["events"][-1].get("winner") != -1:
                continue
            start = max((e["step"] for e in r["events"] if e["kind"] == "contact" and e["side"] == -1), default=0)
            arena.launch(r["shot"])
            arena.partner.reset(seed=r["opponent_seed"])
            for command in r["actions"][:start]:
                arena.exchange(command, far)
            state = sha256(arena.data.qpos.tobytes() + arena.data.qvel.tobytes() + arena.data.ctrl.tobytes()).hexdigest()
            coach = MotionPlanner(arena, side=1, seed=290110)
            coach.flight = NativeBallFlight(arena)
            coach.predict = RallyObjective(coach.predict)
            event_start = len(arena.events)
            for _ in range(180):
                _, _, done, capped, info = arena.exchange(coach.command(), far)
                landed = any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"])
                if landed or done or capped:
                    break
            else:
                raise RuntimeError("No terminal result for attempted return")
            out = dict(id=r["shot"]["id"], step=start, state=state, landed=landed,
                       decisions=len(coach.decisions), events=arena.events[event_start:])
            rows.append(out)
            print(out["id"], landed, out["decisions"], flush=True)
            _write(output, dict(source_sha256=_hash(source), rows=rows))
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    check(args.source, args.output)

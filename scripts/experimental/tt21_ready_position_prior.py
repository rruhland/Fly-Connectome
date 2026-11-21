"""Match learned shoulder preparation with a no-practice constant-velocity optical prior."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt21_ready_position import OUTPUT, PositionStation, ready_metrics, shoulder_goal
from .tt21_recovery_trial import verify

# cue = [1, x at the net, negative lateral slope]. Extrapolate to y = -.88 m.
CONSTANT_VELOCITY = np.array([0.0, 1.0, 0.88])


def run():
    source = OUTPUT / "ready-position-incoming-held.json.gz"
    held = _read(source)
    learned = held["conditions"]["shift"]["points"]
    centered = held["conditions"]["center"]["points"]
    station = PositionStation()
    rows = []
    try:
        for fit, fixed in zip(learned, centered, strict=True):
            decision = fit["decision"]
            if decision is None:
                rows.append(dict(fixed, matched_action_path="center"))
                continue
            x = float(CONSTANT_VELOCITY @ decision["cue"])
            goal = shoulder_goal(x, station.radians)
            equivalent = fixed if goal == 0.0 else fit if goal == decision["goal"] else None
            if equivalent is not None:
                rows.append(
                    dict(
                        equivalent,
                        decision=dict(decision, lane=x, goal=goal),
                        matched_action_path="center" if goal == 0.0 else "learned",
                    )
                )
            else:
                row = station.point(fit["seed"], fit["role"], CONSTANT_VELOCITY, cue_mode="incoming")
                if row["decision"] is None or row["decision"]["tick"] != decision["tick"]:
                    raise ValueError("first optical decision differs before intervention")
                rows.append(row)
        result = {
            "source": source.name,
            "source_sha256": _hash(source),
            "coefficients": CONSTANT_VELOCITY.tolist(),
            "target_examples": 0,
            "points": rows,
            "summary": ready_metrics(rows),
            "replayed": verify(rows),
        }
        _write(OUTPUT / "ready-position-prior-held.json.gz", result)
        print(result["summary"], flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    run()

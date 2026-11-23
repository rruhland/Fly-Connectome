"""Try the solo reward winner on independent releases after a short physical ready interval."""

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_integrated_stress import InstalledArm, episode, schedule
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt26_free_joints import OPTIONS
from .tt26_grip_course import Grip, OUT


class OffsetWrist(ToolEquilibrium):
    def __init__(self, adjustment):
        super().__init__(BUNDLE, 0.0557)
        self.mode = "workspace_local"
        self.adjustment = adjustment

    def begin_release(self, obs):
        super().begin_release(obs)
        self.applied = np.zeros(5)

    def act(self, obs):
        base = super().act(obs)
        physical = np.clip(base + (self.adjustment - self.applied) / (5.2 / 60), -1, 1)
        self.applied += (physical - base) * (5.2 / 60)
        return physical


def run():
    source = _read(OUT / "five-joint-practice.json.gz")["records"]
    plans = [p for p in schedule(3174591, 12) if p["body"] == "stock"]
    station = InstalledArm("stock")
    station.arena.max_steps = 480
    grip = Grip(station.arena._model)
    launcher = BounceArena(view="none")
    rows = []
    try:
        for src in source:
            grip.set(src["mount"])
            if src["mount"] == 0:
                labels = {"nominal": 0}
            else:
                labels = {"bounce": src["selected"], "cold": 0}
            agents = {k: OffsetWrist(OPTIONS[v]) for k, v in labels.items()}
            for plan in plans:
                launcher.reset(seed=plan["launcher_seed"], split="held", render=False)
                launch = [
                    a.copy()
                    for a in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin)
                ]
                results = {}
                for name, agent in agents.items():
                    result = episode(station, agent, plan, launch, "bundle")
                    again = episode(station, agent, plan, launch, "bundle", replay=result["actions"])
                    if digest_json(result["metrics"]) != digest_json(again["metrics"]):
                        raise ValueError("Ready bounce failed physical replay")
                    results[name] = result
                rows.append(dict(mount=src["mount"], plan=plan, results=results))
            _write(OUT / "ready-bounce.json.gz", dict(rows=rows, cap=480))
            print(
                src["mount"],
                {
                    k: [
                        r["results"][k]["metrics"]["longest_consecutive_rebounds"]
                        for r in rows
                        if r["mount"] == src["mount"]
                    ]
                    for k in agents
                },
                flush=True,
            )
    finally:
        station.arena.close()
        launcher.close()


if __name__ == "__main__":
    run()

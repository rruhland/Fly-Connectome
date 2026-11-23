"""Use a reward-discovered joint adjustment to update the estimated tool attitude."""

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_integrated_stress import InstalledArm, episode, schedule
from .tt26_free_joints import OPTIONS
from .tt26_grip_course import Grip, OUT
from .tt26_ready_bounce import OffsetWrist


class RecalledTool(OffsetWrist):
    def begin_release(self, obs):
        super().begin_release(obs)
        # Use the arm's existing calibrated encoder-to-link mapping. If a new
        # posture restores effective play, estimate the corresponding tool
        # orientation in the frame of its new final link.
        decoder = self.sensor.decoder
        _, rotations = decoder.links.read(np.stack((self.reference, self.reference + self.adjustment)))
        self.face_normals[self.calibration_index] = rotations[1].T @ rotations[0] @ decoder.normal


def run():
    source = _read(OUT / "five-joint-practice.json.gz")["records"]
    plans = [p for p in schedule(5715842, 12) if p["body"] == "stock"]
    station = InstalledArm("stock")
    station.arena.max_steps = 480
    launcher = BounceArena(view="none")
    grip = Grip(station.arena._model)
    rows = []
    try:
        for src in source:
            if not src["mount"]:
                continue
            grip.set(src["mount"])
            offset = OPTIONS[src["selected"]]
            actors = {
                "adjustment": OffsetWrist(offset),
                "tool_and_adjustment": RecalledTool(offset),
                "cold": OffsetWrist(np.zeros(5)),
            }
            for p in plans:
                launcher.reset(seed=p["launcher_seed"], split="held", render=False)
                launch = [
                    a.copy()
                    for a in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin)
                ]
                outcomes = {}
                for name, actor in actors.items():
                    live = episode(station, actor, p, launch, "bundle")
                    repeat = episode(station, actor, p, launch, "bundle", replay=live["actions"])
                    if digest_json(live["metrics"]) != digest_json(repeat["metrics"]):
                        raise ValueError("Attitude-aware controller failed independent action replay")
                    outcomes[name] = live
                rows.append(dict(angle=src["mount"], plan=p, results=outcomes))
            _write(OUT / "tool-recall.json.gz", dict(rows=rows, cap=480))
            print(
                src["mount"],
                {
                    k: [
                        r["results"][k]["metrics"]["longest_consecutive_rebounds"]
                        for r in rows
                        if r["angle"] == src["mount"]
                    ]
                    for k in actors
                },
                flush=True,
            )
    finally:
        station.arena.close()
        launcher.close()


if __name__ == "__main__":
    run()

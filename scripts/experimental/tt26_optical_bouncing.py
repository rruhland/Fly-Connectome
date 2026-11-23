"""Apply the same visually inferred tool rotation in wrist bouncing and stereo receiving."""

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_integrated_stress import InstalledArm, episode, schedule
from .tt26_grip_course import OUT, Grip
from .tt26_optical_transfer import MOUNTS, transport
from .tt26_ready_bounce import OffsetWrist


class OpticalWrist(OffsetWrist):
    def __init__(self, rotation, bounds, *, position_scale=0.008):
        super().__init__(np.zeros(5))
        self.position_scale = position_scale
        self.rotation = rotation
        self.bounds = bounds

    def begin_release(self, obs):
        super().begin_release(obs)
        rig = self.sensor.decoder
        self.adjustment = (
            transport(rig, self.reference, self.rotation, self.bounds, position_scale=self.position_scale)
            - self.reference
        )
        self.face_normals[self.calibration_index] = self.rotation @ rig.normal


def run():
    mounts = _read(OUT / "optical-face.json.gz")["mounts"]
    station = InstalledArm("stock")
    arena = station.arena
    arena.max_steps = 480
    grip = Grip(arena._model)
    launcher = BounceArena(view="none")
    plans = [p for p in schedule(197422, 12) if p["body"] == "stock"]
    rows = []
    try:
        for name in ("yaw_plus", "yaw_minus", "pitch_plus"):
            angle, axis = MOUNTS[name]
            grip.set(angle, axis)
            rotation = np.asarray(mounts[name]["rotation_from_nominal"])
            bounds = (arena._control_low[:5], arena._control_high[:5])
            actors = {"learned": OpticalWrist(rotation, bounds), "cold": OffsetWrist(np.zeros(5))}
            for p in plans:
                launcher.reset(seed=p["launcher_seed"], split="held", render=False)
                launch = [
                    a.copy()
                    for a in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin)
                ]
                outcomes = {}
                for label, actor in actors.items():
                    row = episode(station, actor, p, launch, "bundle")
                    replay = episode(station, actor, p, launch, "bundle", replay=row["actions"])
                    if digest_json(row["metrics"]) != digest_json(replay["metrics"]):
                        raise ValueError("Visually retargeted bounce diverged")
                    outcomes[label] = row
                rows.append(dict(mount=name, plan=p, results=outcomes))
            _write(OUT / "optical-bouncing.json.gz", dict(rows=rows))
            print(
                name,
                {
                    k: [
                        r["results"][k]["metrics"]["longest_consecutive_rebounds"]
                        for r in rows
                        if r["mount"] == name
                    ]
                    for k in actors
                },
                flush=True,
            )
    finally:
        launcher.close()
        arena.close()


if __name__ == "__main__":
    run()

"""Deterministic evaluation changes to the real simulated plant and sensor pipeline."""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .tt19_so101 import WRIST_CAMERA
from .tt19_view_trial import CAMERAS


@dataclass(frozen=True)
class Conditions:
    ball_mass: float = 1.0
    table_damping: float = 24.0
    paddle_damping: float = 25.0
    paddle_friction: float = 0.54
    paddle_mass: float = 1.0
    elbow_mass: float = 1.0
    wrist_mass: float = 1.0
    joint_damping: float = 1.0
    outside_shift_m: float = 0.0
    camera_yaw_deg: float = 0.0
    command_delay: int = 0
    image_delay: int = 0
    command_noise: float = 0.0
    streams: tuple = ("left", "right")


CONDITIONS = {
    "stock": Conditions(),
    "two_surface": Conditions(),
    "narrow_table": Conditions(),
    "outside": Conditions(streams=("left",)),
    "wrist": Conditions(streams=("wrist",)),
    "materials": Conditions(
        ball_mass=1.08, table_damping=31.0, paddle_damping=33.0, paddle_friction=0.47
    ),
    "loaded_arm": Conditions(
        paddle_mass=1.13, elbow_mass=1.16, wrist_mass=1.20, joint_damping=1.22
    ),
    "moved_camera": Conditions(
        outside_shift_m=0.08, camera_yaw_deg=5.0, streams=("left", "wrist")
    ),
    "latency": Conditions(command_delay=2, image_delay=1, command_noise=0.025),
    "combined_wrist": Conditions(
        ball_mass=0.96,
        table_damping=29.0,
        paddle_damping=28.0,
        paddle_mass=1.08,
        elbow_mass=1.12,
        wrist_mass=1.18,
        joint_damping=1.13,
        camera_yaw_deg=-4.0,
        command_delay=1,
        image_delay=2,
        command_noise=0.018,
        streams=("wrist",),
    ),
}


def apply(arena, profile):
    """Apply once to a fresh arena, before building any sensor calibration or opponent."""
    if getattr(arena, "_conditioned", False):
        raise ValueError("Conditions have already been applied to this plant")
    m = arena.model
    for name, factor in (
        ("ball", profile.ball_mass),
        ("learner_paddle", profile.paddle_mass),
        ("learner_elbow", profile.elbow_mass),
        ("learner_wrist", profile.wrist_mass),
    ):
        body = m.body(name)
        body.mass[:] *= factor
        body.inertia[:] *= factor
    for name in arena.table_panels:
        m.geom(name).solref[1] = -profile.table_damping
    m.geom("learner_racket").solref[1] = -profile.paddle_damping
    m.geom("learner_racket").friction[0] = profile.paddle_friction
    for actuator in arena._arm_ids[1][2]:
        joint = m.actuator_trnid[actuator, 0]
        m.dof_damping[m.jnt_dofadr[joint]] *= profile.joint_damping
    for name in ("left", "right", "wrist"):
        camera = m.camera(WRIST_CAMERA if name == "wrist" else CAMERAS[name][0])
        if name != "wrist":
            camera.pos[0] += profile.outside_shift_m
        adjustment = Rotation.from_euler("y", profile.camera_yaw_deg, degrees=True)
        nominal = Rotation.from_quat(np.roll(camera.quat, -1))
        camera.quat[:] = np.roll((nominal * adjustment).as_quat(), 1)
    mujoco.mj_setConst(m, arena.data)
    arena.streams = profile.streams
    arena.reset(server=-1, render=False)
    arena._conditioned = True


class Delay:
    """The learner sees capture timestamps and encoders from the same delayed sensor frame."""

    def __init__(self, conditions, seed):
        self.cfg = conditions
        self.rng = np.random.default_rng(seed)
        self.observations = deque()
        self.actions = deque()

    def observe(self, t, frames, q):
        self.observations.append((t, frames, q.copy()))
        if len(self.observations) > self.cfg.image_delay:
            return self.observations.popleft()
        return None

    def command(self, u):
        self.actions.append(np.asarray(u).copy())
        cmd = (
            self.actions.popleft()
            if len(self.actions) > self.cfg.command_delay
            else np.zeros_like(u)
        )
        if self.cfg.command_noise:
            cmd = cmd + self.rng.normal(0, self.cfg.command_noise, len(cmd))
        return np.clip(cmd, -1.0, 1.0)


def cooperative_targets(pose, spread):
    """Select independent partner target positions and arrival speeds inside a reachable region."""
    from .tt29_partner import FOREHAND_LANES, DEEP_LANES

    familiar = FOREHAND_LANES if pose == "lateral" else DEEP_LANES
    if spread == "practice":
        return familiar
    if spread != "varied" or pose != "lateral":
        raise ValueError(
            "The varied challenge uses the lateral receiving configuration"
        )
    rng = np.random.default_rng(290804)
    lo = np.array([0.126, -1.365, 0.985, 3.12])
    hi = np.array([0.320, -1.237, 1.117, 4.08])
    # Stratify each dimension independently, then draw a different target on each far return.
    strata = np.stack([rng.permutation(96) for _ in range(4)], axis=1)
    return lo + (strata + rng.uniform(size=(96, 4))) / 96 * (hi - lo)


def table_settings(name):
    from .tt18_plant import PlantSettings

    if name == "two_surface":
        return PlantSettings(left_damping=18.0, right_damping=32.0)
    if name == "narrow_table":
        return PlantSettings(table_half_width=0.705, table_half_length=1.34)
    return PlantSettings()

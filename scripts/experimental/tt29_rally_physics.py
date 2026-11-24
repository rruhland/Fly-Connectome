"""Candidate strokes for an uninterrupted regulation-table MuJoCo rally."""

import argparse
from copy import copy
import ctypes
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .tt4_arena import (
    BALL_INERTIA, BALL_MASS, BALL_RADIUS, NET_HEIGHT, SOLVER_SUBSTEPS, SPIN_DECAY,
    TABLE_HEIGHT, mujoco,
)
from .tt4_transfer_trial import _write
from .tt19_so101 import GRIP_BODY, _GRASP_POS
from .tt20_opponent import aim_angles
from .tt25_arm_replay import NOMINAL_NO_LOAD
from .tt25_recorded_flight import OUTPUT as FLIGHT_OUTPUT
from .tt25_reference_receiver import ReferenceReceiver
from .tt29_arm_clearance import ArmClearance

OUTPUT = FLIGHT_OUTPUT.parent / "tt29"
REFERENCE = json.loads((FLIGHT_OUTPUT / "recorded-secondary-reference.json").read_text())
DRAG, LIFT = json.loads((FLIGHT_OUTPUT / "recorded-flight.json").read_text())["held_scores"]["match_fit"]["coefficients"]
TABLE_TOP = TABLE_HEIGHT + BALL_RADIUS
NET_TOP = TABLE_HEIGHT + NET_HEIGHT + BALL_RADIUS
STRIKE_READY = (.02386, .79048, -1.13817, -1.38049, -.75920, -.10)
DEEP_STRIKE_READY = (-.097, 1.745, 1.093, 1.584, -.642, -.10)
HIGH_STRIKE_READY = (1.824, -1.744, -.54, 1.516, .479, -.10)
LATERAL_STRIKE_READY = (1.52, -1.56, .28, 1.44, -2.56, -.10)
SIDE_ON_STRIKE_READY = (-.787360, -1.501574, -.890527, 1.654627, 1.290903, -.045)


def measured_launches(path):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != REFERENCE["sha256"]:
        raise ValueError("The launch file differs from the pinned DeepMind recording")
    return {s["id"]: s for s in json.loads(raw)}


class RallyArena(ReferenceReceiver):
    """Use the measured opening once and physical ball simulation thereafter."""

    def __init__(self, *, fixture="table-edge", width=192, height=144, frameskip=4, max_steps=9000,
                 strike_ready=False, clearance_guard=False, fast_search=False, rate_scale=1.0,
                 velocity_guard=False, robust_contact=False, clearance_margin=0.0, motor_envelope=False,
                 **settings):
        if not 0.0 < rate_scale <= 1.0:
            raise ValueError("expected a target slew scale in (0, 1]")
        if clearance_margin < 0.0:
            raise ValueError("expected a nonnegative clearance margin")
        self.strike_ready = strike_ready
        self.clearance_margin = clearance_margin
        self.clearance_guard = clearance_guard or clearance_margin > 0.0
        self.fast_search = fast_search
        self.rate_scale = rate_scale
        self.velocity_guard = velocity_guard
        self.robust_contact = robust_contact
        self.motor_envelope = motor_envelope
        self._motor_limits = None
        if strike_ready == "side-on":
            settings["paddle_grip"] = "side-on"
        self._cad_reader = None
        self._far_action = np.zeros(5)
        super().__init__(fixture=fixture, width=width, height=height, frameskip=frameskip, max_steps=max_steps,
                         stationary_opponent=False, **settings)
        axes = self._arm_ids[1][2]
        driven = self.model.actuator_trnid[axes, 0]
        self._near_dofs = self.model.jnt_dofadr[driven]
        self._observed_speed = np.zeros(len(axes))
        self._observed_force = np.zeros(len(axes))

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if hasattr(self, "_near_dofs"):
            self._observed_speed.fill(0.0)
            self._observed_force.fill(0.0)
        return result

    def sample_clearance(self, live=False):
        if live:
            self._cad_reader = (ArmClearance(self.model), mujoco.MjData(self.model))
        else:
            self._cad_reader = None
        self._cad_count = 0
        self._cad_min = None
        self._cad_violations = 0

    def clearance_result(self):
        return dict(samples=self._cad_count, penetration_samples=self._cad_violations, nearest=self._cad_min)

    def motor_audit(self):
        axes = self._arm_ids[1][2]
        rates = self._joint_increments[axes] / self.model.opt.timestep
        speeds = np.maximum(self._observed_speed, np.abs(self.data.qvel[self._near_dofs]))
        forces = np.maximum(self._observed_force, np.abs(self.data.actuator_force[axes]))
        return dict(target_rate_limit_rad_s=(rates * self.rate_scale).tolist(),
                    no_load_reference_rad_s=NOMINAL_NO_LOAD, linear_motoring_envelope=self.motor_envelope,
                    force_limit_nm=self.model.actuator_forcerange[axes].tolist(),
                    max_measured_rate_rad_s=speeds.tolist(), max_measured_force_nm=forces.tolist())

    def _aerodynamics(self):
        super()._aerodynamics()
        speed = np.abs(self.data.qvel[self._near_dofs])
        force = np.abs(self.data.actuator_force[self._arm_ids[1][2]])
        np.maximum(self._observed_speed, speed, out=self._observed_speed)
        np.maximum(self._observed_force, force, out=self._observed_force)
        if self._cad_reader is not None:
            clearance, shadow = self._cad_reader
            shadow.qpos[:] = self.data.qpos
            self._cad_count += 1
            nearest = min(clearance.distances(shadow, horizon=.025), default=None, key=lambda x: x[2])
            if nearest is not None:
                self._cad_violations += int(nearest[2] < -clearance.tolerance)
                if self._cad_min is None or nearest[2] < self._cad_min["signed_m"]:
                    self._cad_min = dict(first=nearest[0], second=nearest[1], signed_m=nearest[2], time=self.data.time)

        if self.motor_envelope:
            axes = self._arm_ids[1][2]
            self._motor_limits = self.model.actuator_forcerange[axes].copy()
            normalized = self.data.qvel[self._near_dofs] / NOMINAL_NO_LOAD
            self.model.actuator_forcerange[axes, 0] *= np.clip(1.0 + normalized, 0.0, 1.0)
            self.model.actuator_forcerange[axes, 1] *= np.clip(1.0 - normalized, 0.0, 1.0)

    def _contacts(self, incoming):
        if self._motor_limits is not None:
            self.model.actuator_forcerange[self._arm_ids[1][2]] = self._motor_limits
            self._motor_limits = None
        super()._contacts(incoming)

    def _arm_home(self, side):
        if self.strike_ready and side == 1 and self.fixture == "table-edge":
            modes = {"deep": DEEP_STRIKE_READY, "high": HIGH_STRIKE_READY, "lateral": LATERAL_STRIKE_READY,
                     "side-on": SIDE_ON_STRIKE_READY}
            return modes.get(self.strike_ready, STRIKE_READY), ()
        return super()._arm_home(side)

    def _aim(self, side, *args):
        if side == -1:
            return self._far_action
        return super()._aim(side, *args)

    def _command_to(self, side, goal, scale=1.0):
        if side == -1:
            return self._far_action
        return super()._command_to(side, goal, scale)

    def exchange(self, near, far):
        self._far_action = np.asarray(far)
        return self.step(near, render=False)


def _aero(velocity, spin, drag, lift):
    force = BALL_MASS * (-drag * np.linalg.norm(velocity, axis=-1, keepdims=True) * velocity
                         + lift * np.cross(spin, velocity))
    torque = -BALL_INERTIA * SPIN_DECAY * spin
    return np.concatenate((force, torque), axis=-1)


def _strings(values):
    return " ".join(map(str, values))


class BallFlight:
    """Forecast on an independent solver with the compiled playing surfaces."""

    def __init__(self, arena):
        model = arena.model
        self.drag = DRAG
        self.lift = LIFT
        self.dt = model.opt.timestep
        names = (*arena.table_panels, "net", "floor", "ball")
        types = dict(box=6, plane=0, sphere=2)
        geoms = {}
        for name in names:
            g = model.geom(name)
            attributes = dict(
                name=name,
                type=next(k for k, v in types.items() if v == g.type[0]),
                size=_strings(g.size if name != "ball" else g.size[:1]),
                pos=_strings(g.pos),
                quat=_strings(g.quat),
                contype=g.contype[0],
                conaffinity=g.conaffinity[0],
                solref=_strings(g.solref),
                solimp=_strings(g.solimp),
                condim=g.condim[0],
                priority=g.priority[0],
                friction=_strings(g.friction),
                margin=g.margin[0],
                gap=g.gap[0],
            )
            if name == "ball":
                attributes["mass"] = BALL_MASS
            xml = " ".join(f'{k}="{v}"' for k, v in attributes.items())
            geoms[name] = f"<geom {xml}/>"
        surfaces = "".join(geoms[k] for k in names if k != "ball")
        self.model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option timestep="{self.dt}" gravity="0 0 -9.81" integrator="implicitfast" iterations="25"/>'
            f'<worldbody>{surfaces}<body><freejoint/>{geoms["ball"]}</body></worldbody></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.surface = {self.model.geom(name).id for name in arena.table_panels}

    def path(self, position, velocity, spin, side, bounced=False, horizon=0.75):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = np.r_[position, 1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = np.r_[velocity, spin]
        mujoco.mj_forward(self.model, d)
        touching = False
        rows = []
        for _ in range(round(horizon / self.dt)):
            w = np.empty(3)
            mujoco.mju_rotVecQuat(w, d.qvel[3:], d.qpos[3:])
            d.xfrc_applied[1] = _aero(d.qvel[:3], w, self.drag, self.lift)
            mujoco.mj_step(self.model, d)
            p, v = d.qpos[:3], d.qvel[:3]
            table = any(
                c.efc_address >= 0 and (c.geom[0] in self.surface or c.geom[1] in self.surface)
                and abs(c.frame[2]) > 0.55 for c in d.contact
            )
            if table and not touching:
                if bounced or side * p[1] >= 0:
                    break
                bounced = True
            touching = table
            if side * v[1] > -0.2 or p[2] < TABLE_HEIGHT - .04 or abs(p[1]) > 2:
                break
            if bounced and not table and p[2] > TABLE_TOP + .008:
                rows.append(np.r_[d.time, p, v, w])
        return np.asarray(rows).reshape(-1, 10)


class BladeIK:
    """Fit the geometric center and normal of the grasped near-side blade."""

    def __init__(self, arena):
        self.model = arena.model
        self.data = mujoco.MjData(self.model)
        self.joints, _, actuators, _, _ = arena._arm_ids[1]
        self.joints = self.joints[:5]
        self.low, self.high = arena._joint_low[actuators[:5]], arena._joint_high[actuators[:5]]
        self.gripper = self.model.body(GRIP_BODY).id
        blade = self.model.geom("learner_racket")
        weld = self.model.eq_data[self.model.equality("tt19_grasp").id, 3:10]
        relative_grasp = weld[:3] if np.linalg.norm(weld[:3]) > .01 else _GRASP_POS
        held_rotation = np.empty(9)
        mujoco.mju_quat2Mat(held_rotation, weld[3:])
        held_rotation = held_rotation.reshape(3, 3)
        self.center = relative_grasp + held_rotation @ blade.pos
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, blade.quat)
        self.normal = held_rotation @ rotation.reshape(3, 3)[:, 2]
        self.dofs = np.array([self.model.jnt_dofadr[self.model.joint(f"learner_{axis}").id] for axis in (
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
        )])
        self.linear = np.zeros((3, self.model.nv))
        self.angular = np.zeros_like(self.linear)

    def pose(self, joints, *, jacobian=False):
        d = self.data
        d.qpos[self.joints] = joints
        mujoco.mj_kinematics(self.model, d)
        rotation = d.xmat[self.gripper].reshape(3, 3)
        center = d.xpos[self.gripper] + rotation @ self.center
        normal = rotation @ self.normal
        if normal[1] < 0:
            normal = -normal
        if not jacobian:
            return center, normal
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_jac(self.model, d, self.linear, self.angular, center, self.gripper)
        return center, normal, self.linear[:, self.dofs].copy(), np.cross(self.angular[:, self.dofs].T, normal).T

    def fit(self, center, normal, start):
        last = None

        def pose(q):
            nonlocal last
            if last is None or not np.array_equal(q, last[0]):
                last = q.copy(), self.pose(q, jacobian=True)
            return last[1]

        def error(q):
            p, n, *_ = pose(q)
            return np.r_[p - center, .10 * (n - normal), .006 * (q - start)]

        def jacobian(q):
            _, _, p, n = pose(q)
            return np.concatenate((p, .10 * n, .006 * np.eye(5)))

        result = least_squares(
            error, np.clip(start, self.low + 1e-7, self.high - 1e-7), jac=jacobian,
            bounds=(self.low, self.high), max_nfev=25, ftol=1e-4, gtol=1e-5,
        )
        return result.x, self.pose(result.x, jacobian=True)


def desired_shot(p, vx, side, landing, pace):
    """Approximate launch direction for a starting search over real impacts."""
    dx, dy = np.asarray(landing) - p[:2]
    duration = np.expm1(DRAG * abs(dy)) / (DRAG * pace)
    drag_rate = DRAG * pace
    decay = np.exp(-drag_rate * duration)
    vertical_speed = (TABLE_TOP - p[2] + 9.81 * duration / drag_rate) * drag_rate / (1 - decay) - 9.81 / drag_rate
    goal = np.array([dx * pace / abs(dy), side * pace, vertical_speed])
    normal = goal - vx
    return normal / np.linalg.norm(normal)


def _swing(timeline, time, lead=.03):
    return np.clip(timeline - time + lead, -.12, .10)


def _normalized(value):
    return value / np.linalg.norm(value)


@lru_cache(maxsize=1)
def _native_trials():
    package = Path(mujoco.__file__).parent
    library = next(package.glob("libmujoco.so.*"))
    workspace = TemporaryDirectory(prefix="tt29-strokes-")
    target = Path(workspace.name) / "strokes.so"
    source = Path(__file__).with_name("tt29_rally_rollout.cpp")
    subprocess.run([
        "c++", "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC", "-fopenmp",
        "-I", str(package / "include"), str(source), str(library), f"-Wl,-rpath,{package}",
        "-o", str(target),
    ], check=True)
    function = ctypes.CDLL(str(target)).trial_strokes_v2
    pointer, integer, number = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    function.argtypes = (
        pointer, pointer, pointer, integer, integer, integer, integer, pointer, integer, pointer,
        integer, pointer, pointer, pointer, pointer, pointer, ctypes.c_bool, number, number, number,
        pointer, pointer, pointer, pointer, pointer, integer, number, number, pointer, number, number,
        number, pointer, pointer, integer,
    )
    function.restype = None
    return function, workspace


class Rollout:
    """Evaluate candidate schedules on isolated data under the native contacts."""

    def __init__(self, arena, side):
        self.arena, self.side = arena, side
        self.model = arena.model
        self.axes = arena._arm_ids[side][2].astype(np.intc)
        self.velocity_dofs = np.ascontiguousarray(
            self.model.jnt_dofadr[self.model.actuator_trnid[self.axes, 0]], dtype=np.intc)
        self.other = arena._arm_ids[-side][2].astype(np.intc)
        self.near_axes = arena._arm_ids[1][2].astype(np.intc)
        self.near_dofs = arena._near_dofs.astype(np.intc)
        self.other_home = np.asarray(arena._arm_home(-side)[0], dtype=float)
        self.micro = arena.frameskip * SOLVER_SUBSTEPS
        self.ids = np.array([
            arena._geom_ids["ball"], arena._racket_ids[side], arena._geom_ids["table"],
            arena.model.geom("table_right").id if "table_right" in arena.table_panels else -1,
            arena._geom_ids["net"], arena._ball_body, arena._ball_joint, arena._ball_dof,
        ], dtype=np.intc)
        # The far-side carriage has its own planner. Keep its collision out of the
        # near search, which only scores the outgoing ball before a return.
        if side == 1 and arena.model.body(arena.model.geom("rival_racket").bodyid[0]).name == "tt29_partner_carriage":
            self.model = copy(arena.model)
            geom = self.model.geom("rival_racket")
            geom.contype[0] = geom.conaffinity[0] = 0
        self.trial, _ = _native_trials()
        guard = ArmClearance(self.model) if side == 1 and arena.clearance_guard else None
        self.first = np.ascontiguousarray(guard.first if guard else [], dtype=np.intc)
        self.second = np.ascontiguousarray(guard.second if guard else [], dtype=np.intc)

    def trajectories(self, goals, target, cutoff):
        a = self.arena
        goals = np.ascontiguousarray(goals, dtype=float)
        count, frames, _ = goals.shape
        result = np.empty((count, 13))
        rates = np.zeros_like(goals)
        self.trial(
            self.model._address, a.data._address, goals.ctypes.data, count, frames, self.micro, self.side,
            self.axes.ctypes.data, len(self.axes), self.other.ctypes.data, len(self.other), self.other_home.ctypes.data,
            a._joint_increments.ctypes.data, a._joint_low.ctypes.data, a._joint_high.ctypes.data, self.ids.ctypes.data,
            a._received_bounce, cutoff, DRAG, LIFT, target.ctypes.data, rates.ctypes.data, result.ctypes.data,
            self.first.ctypes.data, self.second.ctypes.data, len(self.first),
            a.rate_scale if self.side == 1 else 1.0,
            NOMINAL_NO_LOAD if self.side == 1 and a.velocity_guard else 0.0, self.velocity_dofs.ctypes.data,
            45.0 if self.side == 1 and a.robust_contact else 0.0, a.clearance_margin,
            NOMINAL_NO_LOAD if a.motor_envelope else 0.0,
            self.near_axes.ctypes.data, self.near_dofs.ctypes.data, len(self.near_axes),
        )
        return rates, result


class MotionPlanner:
    """Physics-state coach that chooses a stroke and issues signed joint rates."""

    def __init__(self, arena, side=1, *, seed=0):
        self.arena, self.side = arena, side
        self.ik = BladeIK(arena) if side == 1 else None
        self.flight = BallFlight(arena)
        self.predict = Rollout(arena, side)
        self.rng = np.random.default_rng(seed)
        self.home = np.asarray(arena._arm_home(side)[0])
        self.axes = arena._arm_ids[side][2]
        self.rates = arena._joint_increments[self.axes] / arena.model.opt.timestep
        self.reset()

    def reset(self):
        self.decisions = []
        self.tape = None
        self.start = -1
        self.reception = -1
        self.attempts = 0

    def command(self):
        a = self.arena
        if (a._racket_ids[self.side] in a._previous_touches and self.tape is not None
                and 0 <= a.steps - self.start < len(self.tape)):
            return self.tape[a.steps - self.start]
        if (a._last_hitter == self.side or self.side * a.velocity[1] > -.3
                or a._racket_ids[-self.side] in a._previous_touches):
            if a.velocity_guard and self.side == 1:
                age = a.steps - self.start
                if a._last_hitter == self.side and self.tape is not None and 0 <= age < len(self.tape):
                    # The complete follow-through already passed the simulated
                    # speed screen. A separate home sweep has no such guarantee.
                    return self.tape[age]
                return np.zeros(len(self.axes))
            return self.home_command()
        reception = a.contacts + a.rival_contacts
        if reception != self.reception:
            self.reception, self.tape, self.attempts = reception, None, 0
        if self.tape is not None:
            age = a.steps - self.start
            if age < len(self.tape):
                return self.tape[age]
        if self.attempts < 4 and self.side * a.ball[1] > -1.45 and a.steps - self.start >= 3:
            self.tape = self.plan()
            self.start = a.steps
            self.attempts += 1
            if self.tape is not None:
                return self.tape[0]
        if a.velocity_guard and self.side == 1:
            return np.zeros(len(self.axes))
        return self.home_command()

    def home_command(self):
        a = self.arena
        rate = min(.8, a.rate_scale) if self.side == 1 else .8
        return np.clip((self.home - a.data.ctrl[self.axes]) / (a.control_dt * self.rates), -rate, rate)

    def target(self):
        # Far places shots into a small varied corridor suitable for the shorter arm.
        if self.side == -1:
            if self.arena.strike_ready == "lateral":
                return np.r_[self.rng.uniform(.17, .25), 1.29, self.rng.uniform(1.025, 1.10),
                             self.rng.uniform(3.25, 3.9)]
            if self.arena.strike_ready in {"deep", "high", "side-on"}:
                z = self.rng.uniform(1.07, 1.15) if self.arena.strike_ready == "high" else self.rng.uniform(1.03, 1.12)
                return np.r_[self.rng.uniform(-.055, .055), 1.34, z, self.rng.uniform(3.3, 3.85)]
            return np.r_[self.rng.uniform(-.036, .036), self.arena.base_depth - .27, self.rng.uniform(1.13, 1.18),
                         self.rng.uniform(3.3, 3.75)]
        return np.r_[self.rng.uniform(-.038, .038), 1.32, 1.08, self.rng.uniform(3.45, 3.95)]

    def candidates(self, windows, target, timeline):
        a, side = self.arena, self.side
        rates = self.rates[:5]
        qstart = a.data.ctrl[self.axes][:5]
        landing = [target[0], side * (target[1] - .27)]
        goals, params = [], []
        for time, *row in windows:
            p, v = np.array(row[:3]), np.array(row[3:6])
            tilts = (-.065, .09, .245) if self.ik else (-.11, .07, .20, .32, .45)
            offsets = (.0, .036) if self.ik else (.0,)
            for pace, dp, dy, offset in product((4.9, 6.4), tilts, (-.055, .055), offsets):
                normal = _normalized(desired_shot(p, v, side, landing, pace) + [dy, 0.0, dp])
                center = p - normal * .025 + [0.0, 0.0, offset]
                if self.ik:
                    joints, (at, face, linear, angular) = self.ik.fit(center, normal, qstart)
                    displacement = at - p
                    if abs(displacement @ face + .025) > .045 or np.linalg.norm(np.cross(displacement, face)) > .075:
                        continue
                    if face @ normal < .82:
                        continue
                    if np.any(abs(joints - qstart) > rates * (time + .045) + .09):
                        continue
                    direction = np.linalg.lstsq(
                        np.r_[linear, .042 * angular, .013 * np.eye(5)],
                        np.r_[normal, np.zeros(8)], rcond=None,
                    )[0]
                    direction /= max(1, np.max(np.abs(direction) / rates))
                    for motion, lead in ((0.0, 0.0), (.85, .008), (1.32, .044), (1.5, .076), (-.52, .025)):
                        trajectory = joints + _swing(timeline, time, lead)[:, None] * direction * motion
                        goals.append(np.c_[trajectory, np.full(len(trajectory), self.home[5])])
                        params.append([time, p.tolist(), normal.tolist(), pace, motion, lead])
                else:
                    pitch = np.arcsin(normal[2])
                    heading = np.arctan2(normal[0], -normal[1])
                    for motion, lead in product((-.50, .35, 1.15, 2.10), (.015, .057, .105)):
                        depths = center[1] + side * motion * _swing(timeline, time, lead)
                        angles = aim_angles(a.rival_base_depth, center[0], depths, center[2], pitch)
                        goals.append(np.c_[angles, -angles[:, 0] + heading])
                        params.append([time, p.tolist(), normal.tolist(), pace, motion, lead])
        return np.asarray(goals), params

    def plan(self, path=None):
        started = perf_counter()
        a = self.arena
        if path is None:
            path = self.flight.path(a.ball, a.velocity, a.spin, self.side, a._received_bounce)
        if self.ik:
            if a.strike_ready == "lateral":
                depths = np.array([1.38, 1.35, 1.32, 1.29, 1.26, 1.23, 1.20])
            elif a.strike_ready in {"deep", "high", "side-on"}:
                depths = np.array([1.50, 1.45, 1.40, 1.36, 1.32, 1.28, 1.24])
            else:
                depths = a.base_depth - np.array([.18, .22, .26, .30, .34, .38, .42])
            region = path[(abs(path[:, 2]) > depths[-1] - .04) & (path[:, 3] > .89) & (path[:, 3] < 1.34)]
        else:
            depths = np.array([1.24, 1.30, 1.37, 1.45, 1.54, 1.62])
            region = path[(path[:, 3] > .87) & (path[:, 3] < 1.45)]
        options = []
        for depth in depths:
            if len(region) and abs(region[0, 2]) - .006 <= depth <= abs(region[-1, 2]) + .006:
                choice = region[np.argmin(abs(abs(region[:, 2]) - depth))]
                if all(abs(o[0] - choice[0]) > .005 for o in options):
                    options.append(choice)
        target = self.target()
        record = dict(step=a.steps, side=self.side, ball=a.ball.tolist(), windows=np.asarray(options).tolist(),
                      target=target.tolist(), candidates=0, successful=0)
        if not options:
            self.decisions.append(record)
            return None
        frames = int(np.ceil((max(w[0] for w in options) + .68) / a.control_dt))
        timeline = (1 + np.arange(frames)) * a.control_dt
        goals, params = self.candidates(options, target, timeline)
        record["candidates"] = len(goals)
        if not len(goals):
            self.decisions.append(record)
            return None
        cutoff = max(w[0] for w in options) + .07
        selected = np.arange(len(goals))
        if a.fast_search and self.ik and len(goals) > 280:
            rng = np.random.default_rng(290_037 + a.steps)
            trial = np.sort(rng.choice(len(goals), size=240, replace=False))
            first_rates, first_outcomes = self.predict.trajectories(goals[trial], target, cutoff)
            good = np.isfinite(first_outcomes[:, 0])
            if good.any() and np.min(first_outcomes[good, 0]) < 2.0:
                selected = trial
                rates, outcomes = first_rates, first_outcomes
            else:
                rates, outcomes = self.predict.trajectories(goals, target, cutoff)
        else:
            rates, outcomes = self.predict.trajectories(goals, target, cutoff)
        successful = np.flatnonzero(np.isfinite(outcomes[:, 0]))
        record["successful"] = len(successful)
        record["elapsed"] = perf_counter() - started
        if len(successful):
            index = int(successful[np.argmin(outcomes[successful, 0])])
            cost, strike, landing_time, x, y, clearance, rx, ry, rz, pace, incoming, outgoing, radius = outcomes[index]
            chosen = dict(cost=cost, strike_time=strike, landing_time=landing_time, landing=[x, y],
                          clearance=clearance, receiver=[rx, ry, rz, pace], incoming=incoming, outgoing=outgoing,
                          contact_radius=radius)
            record.update(chosen=chosen, parameters=params[selected[index]], assessed=len(selected))
            self.decisions.append(record)
            return rates[index]
        self.decisions.append(record)
        return None



def event_metrics(events):
    """Count alternating physically struck returns and sample separation speeds."""
    strikes = {1: None, -1: None}
    stats = []
    for event in events:
        kind = event["kind"]
        if kind == "contact":
            strikes[event["side"]] = dict(time=event["time"], in_speed=list(event["incoming"]))
        if kind == "separation" and event["collider"] in {"learner_racket", "rival_racket"}:
            side = 1 if event["collider"] == "learner_racket" else -1
            if strikes[side] is not None:
                strikes[side]["out_speed"] = list(event["velocity"])
        if kind == "legal_return" and strikes[event["side"]] is not None:
            side = event["side"]
            hit = strikes[side]
            hit.update(side=side, landing_time=event["time"], location=list(event["location"]))
            stats.append(hit)
            strikes[side] = None
    if any(prev["side"] == current["side"] for prev, current in zip(stats, stats[1:])):
        raise ValueError("The referee reported nonalternating consecutive returns")
    return stats


def replay(arena, record):
    """Reproduce every return and fault starting again at the measured opening."""
    arena.launch(record["launch"])
    if "opponent_seed" in record:
        arena.partner.reset(seed=record["opponent_seed"])
    arena.sample_clearance(True)
    digest = sha256()
    started = perf_counter()
    for near, far in record["actions"]:
        arena.exchange(near, far)
        digest.update(arena.data.qpos.astype("<f8").tobytes())
        digest.update(arena.data.qvel.astype("<f8").tobytes())
    if json.dumps(arena.events) != json.dumps(record["events"]):
        raise ValueError("Live rally events changed on deterministic action replay")
    state_hash = digest.hexdigest()
    if state_hash != record.get("live_state_sha256", state_hash):
        raise ValueError("Live state trajectory changed on deterministic action replay")
    result = dict(state_sha256=state_hash, event_count=len(arena.events), motor=arena.motor_audit(),
                  clearance=arena.clearance_result(), elapsed=perf_counter() - started)
    arena.sample_clearance(False)
    return result


def rally(arena, shot, *, seed=0, target=100, display=False, checkpoint=None):
    arena.launch(shot)
    near = MotionPlanner(arena, 1, seed=seed)
    far = MotionPlanner(arena, -1, seed=seed + 1) if getattr(arena, "partner", None) is None else None
    actions = []
    state_hash = sha256()
    returns = 0
    ledger = 0
    started = perf_counter()
    partner = getattr(arena, "partner", None)

    def record():
        volleys = event_metrics(arena.events)
        if len(volleys) != returns:
            raise ValueError("Unmatched legal return or physical strike")
        config = dict(fixture=arena.fixture, strike_ready=arena.strike_ready, frameskip=arena.frameskip,
                      clearance_guard=arena.clearance_guard, fast_search=arena.fast_search,
                      velocity_guard=arena.velocity_guard, rate_scale=arena.rate_scale,
                      robust_contact=arena.robust_contact, clearance_margin=arena.clearance_margin,
                      motor_envelope=arena.motor_envelope)
        if partner is not None:
            config.update(partner_lanes=partner.targets.tolist(), guard_table=partner.guard_table)
        return dict(seed=seed, launch=shot, returns=returns, result=arena.point_result(), events=arena.events,
                    volleys=volleys, motor=arena.motor_audit(), configuration=config, actions=actions,
                    decisions=near.decisions + (partner.reports if far is None else far.decisions),
                    elapsed=perf_counter() - started, live_state_sha256=state_hash.hexdigest())

    for _ in range(arena.max_steps):
        u = near.command()
        v = np.zeros(5) if far is None else far.command()
        _, _, done, cap, info = arena.exchange(u, v)
        actions.append([u.tolist(), v.tolist()])
        state_hash.update(arena.data.qpos.astype("<f8").tobytes())
        state_hash.update(arena.data.qvel.astype("<f8").tobytes())
        for event in info["events"]:
            if event["kind"] == "contact":
                ledger += 1
            elif ledger and event["kind"] == "legal_return":
                returns += 1
                if display:
                    print("legal", arena.steps, returns, event["side"], event["location"], flush=True)
        if checkpoint is not None and returns and returns % 5 == 0 and any(
            event["kind"] == "legal_return" for event in info["events"]
        ):
            checkpoint(record())
        if done or cap or returns >= target:
            break
    return record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feeds", type=Path)
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=803145)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--fixture", default="table-edge")
    poses = parser.add_mutually_exclusive_group()
    poses.add_argument("--camera-ready", dest="ready", action="store_const", const=False)
    poses.add_argument("--deep-ready", dest="ready", action="store_const", const="deep")
    poses.add_argument("--high-ready", dest="ready", action="store_const", const="high")
    poses.add_argument("--lateral-ready", dest="ready", action="store_const", const="lateral")
    poses.add_argument("--side-on-ready", dest="ready", action="store_const", const="side-on")
    parser.set_defaults(ready=True)
    parser.add_argument("--partner", action="store_true")
    parser.add_argument("--guard-table", action="store_true")
    parser.add_argument("--guard-arm", action="store_true")
    parser.add_argument("--clearance-margin", type=float, default=0.0)
    parser.add_argument("--fast-search", action="store_true")
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--guard-speed", action="store_true")
    parser.add_argument("--motor-envelope", action="store_true")
    parser.add_argument("--robust-contact", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT / "rally-physics.json.gz")
    args = parser.parse_args()
    launches = measured_launches(args.feeds)
    settings = dict(fixture=args.fixture, strike_ready=args.ready, clearance_guard=args.guard_arm,
                    fast_search=args.fast_search, rate_scale=args.rate_scale, velocity_guard=args.guard_speed,
                    robust_contact=args.robust_contact, clearance_margin=args.clearance_margin,
                    motor_envelope=args.motor_envelope)
    if args.partner:
        from .tt29_partner import FOREHAND_LANES, PartnerArena
        if args.ready == "lateral":
            zone = FOREHAND_LANES
        elif args.ready in {"deep", "high", "side-on"}:
            rise = .04 if args.ready == "high" else 0.
            zone = [[x, -1.34 + y, z + rise, v] for x, y, z, v in (
                (-.042, 0., 1.05, 3.40), (.042, -.012, 1.10, 3.60), (.015, .009, 1.07, 3.30),
                (-.02, -.012, 1.12, 3.7), (0., .002, 1.10, 3.50),
            )]
        else:
            zone = [[x, -1.07 + y, z, v] for x, y, z, v in (
                (-.042, 0., 1.15, 3.30), (.042, -.012, 1.17, 3.50), (.015, .009, 1.14, 3.25),
                (-.02, -.012, 1.18, 3.65), (0., .002, 1.16, 3.45),
            )]
        arena = PartnerArena(targets=zone, guard_table=args.guard_table, **settings)
    else:
        arena = RallyArena(**settings)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(row):
        _write(args.output, [*rows, dict(row, checkpoint=True)])

    try:
        for sid in args.ids:
            row = rally(arena, launches[sid], seed=args.seed, target=args.target, display=True, checkpoint=checkpoint)
            checkpoint(row)
            row["replay"] = replay(arena, row)
            rows.append(row)
            print(sid, row["returns"], row["elapsed"], row["result"], flush=True)
            _write(args.output, rows)
    finally:
        arena.close()


if __name__ == "__main__":
    main()

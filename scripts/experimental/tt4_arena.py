"""Regulation-size table tennis with MuJoCo contacts and a stereo RGB interface."""

import math
import os
from pathlib import Path

import numpy as np


# Select the headless graphics backend before MuJoCo imports its renderer.
if 'DISPLAY' not in os.environ and 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco


TABLE_HEIGHT = .76
TABLE_HALF_WIDTH = .7625
TABLE_HALF_LENGTH = 1.37
NET_HEIGHT = .1525
BALL_RADIUS = .020
BALL_MASS = .0027
PADDLE_RADIUS = .076
BASE_DEPTH, BASE_HEIGHT, LINK_LENGTH = 1.94, .96, .38
TIMESTEP = 1 / 240
SOLVER_SUBSTEPS = 2
GRAVITY = 9.81
TABLE_BOUNCE = .88  # Estimate for the serve generator and diagnostic controllers.
TABLE_FORWARD = .93
RACKET_BOUNCE = .85
AIR_DRAG = .17345556
MAGNUS_COEFFICIENT = .0003220273
SPIN_DECAY = .18
BALL_INERTIA = .4 * BALL_MASS * BALL_RADIUS**2
ARM_AXES = ('yaw', 'shoulder', 'elbow', 'wrist')
JOINT_RATES = np.array((4.5, 7.0, 8.5, 7.5))
JOINT_LIMITS = np.array(((-1.18, 1.18), (-1.18, .68), (.40, 2.36), (-.75, .82)))
HOME = np.array((0., -.58, 1.37, .31))
RIVAL_SERVE_HOME = np.array((0., -.74, 1.82, .15))
MODEL_PATH = Path(__file__).with_suffix('.xml')
COLORS = np.array(((244, 205, 70), (227, 71, 45), (59, 181, 227),
                   (197, 91, 218), (99, 203, 76)), dtype=np.uint8)


def _time_to_height(z, vz, height):
    return (vz + math.sqrt(max(0., vz * vz + 2 * GRAVITY * (z - height)))) / GRAVITY


def _vertical(z, vz, elapsed, drag_rate):
    decay = math.exp(-drag_rate * elapsed)
    gravity_term = GRAVITY / drag_rate
    return z + (vz + gravity_term) * (1 - decay) / drag_rate - gravity_term * elapsed, (
        (vz + gravity_term) * decay - gravity_term)


def _flight_to_depth(position, velocity, depth):
    """Fast approximate interception guess for the scripted players."""
    x, y, z = position
    vx, vy, vz = velocity
    floor = TABLE_HEIGHT + BALL_RADIUS
    for _ in range(3):
        distance = abs(depth - y)
        speed = max(.2, abs(vy))
        drag = AIR_DRAG * min(15., max(.5, math.sqrt(vx * vx + vy * vy + vz * vz)))
        duration = math.expm1(AIR_DRAG * distance) / (AIR_DRAG * speed)
        hit_z, hit_vz = _vertical(z, vz, duration, drag)
        if hit_z >= floor or duration <= 0.:
            return x + (depth - y) * vx / vy, hit_z, vy * math.exp(-drag * duration), hit_vz
        bounce = _time_to_height(z, vz, floor)
        for _ in range(2):
            estimate_z, estimate_vz = _vertical(z, vz, bounce, drag)
            bounce = max(0., bounce + (estimate_z - floor) / max(.2, -estimate_vz))
        delta_y = vy * -math.expm1(-drag * bounce) / drag
        if abs(y + delta_y) > TABLE_HALF_LENGTH or abs(delta_y) >= distance:
            return x + (depth - y) * vx / vy, hit_z, vy * math.exp(-drag * duration), hit_vz
        x += vx * delta_y / vy
        y += delta_y
        z, vz = floor, -estimate_vz * TABLE_BOUNCE
        slowdown = math.exp(-drag * bounce) * TABLE_FORWARD
        vx, vy = vx * slowdown, vy * slowdown
    return x, z, vy, vz


def _arm_angles(x, y, z, side, loft):
    toward = BASE_DEPTH + side * y
    lateral = side * x
    radial, vertical = math.hypot(lateral, toward), z - BASE_HEIGHT
    yaw = math.atan2(lateral, toward)
    cosine = (radial * radial + vertical * vertical) / (2 * LINK_LENGTH**2) - 1.
    elbow = math.acos(max(-.99, min(.99, cosine)))
    shoulder = math.atan2(vertical, radial) - elbow / 2
    return np.array((yaw, shoulder, elbow, loft))


def _racket_loft(v_forward, vz, z, distance, racket_speed):
    """Aim the physical blade by predicting its outgoing projectile."""
    floor = TABLE_HEIGHT + BALL_RADIUS
    best_angle, smallest_error = .3, float('inf')
    for i in range(43):
        angle = -.30 + .021 * i
        sn, cs = math.sin(angle), math.cos(angle)
        normal = (v_forward - racket_speed) * cs + vz * sn
        tangent = -(v_forward - racket_speed) * sn + vz * cs
        normal_delta = -(1. + RACKET_BOUNCE) * normal
        # A sphere's tangential impulse both changes its velocity and spins it.
        tangent_delta = -.24 * tangent
        forward = v_forward + normal_delta * cs - tangent_delta * sn
        up = vz + normal_delta * sn + tangent_delta * cs
        elapsed = math.expm1(AIR_DRAG * distance) / (AIR_DRAG * max(.4, forward))
        landed_height, _ = _vertical(z, up, elapsed, AIR_DRAG * max(.6, forward))
        error = abs(landed_height - floor)
        if error < smallest_error:
            best_angle, smallest_error = angle, error
        if error < .06:
            break
    return best_angle


class TT4Arena:
    """Two perspective RGB views, four signed joint rates and point-only reward.

    `step` diagnostics and the scripted controls are for external evaluators.
    Actions integrate position targets at the physics timestep. Torque-limited
    position servos and the mechanically coupled wrist determine the actual motion.
    """

    def __init__(self, seed=0, *, width=96, height=72, frameskip=8, max_steps=300,
                 render_mode='projected', view='first_person', view_seed=None, distractor=False,
                 baseline=.16, venue='studio'):
        if any(type(value) is not int for value in (width, height, frameskip, max_steps)) or (
            min(width, height) < 16 or min(frameskip, max_steps) < 1
        ):
            raise ValueError('expected integer camera dimensions ≥16 and positive timing parameters')
        if render_mode not in ('projected', 'mujoco'):
            raise ValueError('expected a known rendering backend')
        if venue not in ('studio', 'hall'):
            raise ValueError('expected a known venue')
        self.seed, self.view_seed = int(seed), view_seed
        self.width, self.height = width, height
        self.frameskip, self.max_steps = frameskip, max_steps
        self.control_dt = self.frameskip * TIMESTEP
        self.render_mode, self.view, self.venue = render_mode, view, venue
        self.baseline = baseline
        self.distractor = distractor
        if venue == 'hall':
            from .tt4_hall import hall_model

            self.model = hall_model(MODEL_PATH)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self._ball_joint = self.model.jnt_qposadr[self.model.joint('ball_free').id]
        self._ball_dof = self.model.jnt_dofadr[self.model.joint('ball_free').id]
        self._ball_body = self.model.body('ball').id
        self._decoy = self.model.body('decoy').mocapid[0]
        self._geom_ids = {name: self.model.geom(name).id for name in
                          ('ball', 'table', 'net', 'floor', 'learner_racket', 'rival_racket')}
        self._collider_names = {idx: name for name, idx in self._geom_ids.items()}
        self._joint_low = np.tile(JOINT_LIMITS[:, 0], 2)
        self._joint_high = np.tile(JOINT_LIMITS[:, 1], 2)
        self._joint_increments = np.tile(JOINT_RATES * self.model.opt.timestep, 2)
        self._racket_ids = {1: self._geom_ids['learner_racket'], -1: self._geom_ids['rival_racket']}
        self._arm_ids = {}
        self._renderer = None
        self._rig = None
        for side, player in ((1, 'learner'), (-1, 'rival')):
            self._arm_ids[side] = (
                np.array([self.model.jnt_qposadr[self.model.joint(f'{player}_{axis}').id]
                          for axis in ARM_AXES]),
                np.array([self.model.jnt_qposadr[self.model.joint(f'{player}_{axis}_link').id]
                          for axis in ('pitch', 'yaw')]),
                np.array([self.model.actuator(f'{player}_{axis}').id for axis in ARM_AXES]),
                self.model.body(f'{player}_base').id,
                self.model.body(f'{player}_elbow').id,
            )
        self._point = -1
        self.reset(render=False)

    @property
    def ball(self):
        return self.data.qpos[self._ball_joint:self._ball_joint + 3]

    @property
    def velocity(self):
        return self.data.qvel[self._ball_dof:self._ball_dof + 3]

    @property
    def spin(self):
        """Ball angular velocity in world axes, in radians per second."""
        world_spin = np.empty(3)
        mujoco.mju_rotVecQuat(world_spin, self.data.qvel[self._ball_dof + 3:self._ball_dof + 6],
                              self.data.qpos[self._ball_joint + 3:self._ball_joint + 7])
        return world_spin

    @property
    def joints(self):
        return self.data.qpos[self._arm_ids[1][0]].copy()

    @property
    def paddle(self):
        return self.data.geom_xpos[self._racket_ids[1]]

    def _opening(self):
        """Release a seeded serve past the rival's end line toward its first bounce."""
        rng = self._rng
        start_y = 1.42
        first_y = rng.uniform(.82, 1.04)
        receive_y = rng.uniform(-1.12, -.91)
        speed = rng.uniform(3.9, 4.45)
        start_x = rng.uniform(-.36, .36)
        receive_x = .62 * start_x + rng.uniform(-.25, .25)
        vx = speed * (receive_x - start_x) / (start_y + 1.41)
        first_time = math.expm1(AIR_DRAG * (start_y - first_y)) / (AIR_DRAG * speed)
        first_speed = speed * math.exp(-AIR_DRAG * speed * first_time) * TABLE_FORWARD
        flight = math.expm1(AIR_DRAG * (first_y - receive_y)) / (AIR_DRAG * first_speed)
        drag = AIR_DRAG * first_speed
        rise = GRAVITY * flight / -math.expm1(-drag * flight) - GRAVITY / drag
        impact = -rise / TABLE_BOUNCE
        before_drag = AIR_DRAG * speed
        vz = (impact + GRAVITY / before_drag) * math.exp(before_drag * first_time) - GRAVITY / before_drag
        vertical_displacement, _ = _vertical(0., vz, first_time, before_drag)
        self.ball[:] = (start_x, start_y, TABLE_HEIGHT + BALL_RADIUS - vertical_displacement)
        self.velocity[:] = (vx, -speed, vz)
        self.data.qvel[self._ball_dof + 3:self._ball_dof + 6] = rng.uniform((65., -24., -31.), (145., 24., 31.))

    def reset(self, *, seed=None, render=True):
        """Start a reproducible point and return RGB pixels if requested."""
        if seed is not None:
            self.seed, self._point = int(seed), -1
        self._point += 1
        self._rng = np.random.default_rng(np.random.SeedSequence((self.seed, self._point, 4)))
        appearance = self.seed if self.view_seed is None else int(self.view_seed)
        view_rng = np.random.default_rng(np.random.SeedSequence((appearance, self._point, 91)))
        self._colors = np.clip(view_rng.permutation(COLORS, axis=0)[:4].astype(int) +
                               view_rng.integers(-9, 10, (4, 3)), 0, 255).astype(np.uint8)
        if self.venue == 'hall':
            from .tt4_hall import hall_colors

            self._colors = hall_colors(view_rng)
        for geom, rgb in zip(('learner_racket', 'rival_racket', 'ball', 'decoy'), self._colors):
            self.model.geom(geom).rgba[:3] = rgb / 255.
        self._decoy_phase = view_rng.uniform(0., 2 * math.pi, 3)
        mujoco.mj_resetData(self.model, self.data)
        self._opening()
        for side, (qpos, passive, actuator, _, _) in self._arm_ids.items():
            home = HOME if side == 1 else RIVAL_SERVE_HOME
            self.data.qpos[qpos] = home
            self.data.qpos[passive] = (-home[1] - home[2], -home[0])
            self.data.ctrl[actuator] = home
        self.steps = self.contacts = self.legal_returns = self.rival_returns = self.rival_contacts = 0
        self._last_hitter, self._received_bounce = -1, False
        self._serving, self._server_bounce, self._serve_net = True, False, False
        self._rival_attempt, self._rival_bias, self._rival_target = False, 0., .95
        self._previous_touches = {}
        self.events = []
        self.reward, self.terminated, self.truncated, self.reason = 0, False, False, None
        self._place_decoy()
        mujoco.mj_forward(self.model, self.data)
        return self.render_cameras() if render else None

    def _place_decoy(self):
        if self.distractor:
            t = self.data.time
            a, b, c = self._decoy_phase
            self.data.mocap_pos[self._decoy] = (.55 * math.sin(1.8 * t + a),
                                                1.48 * math.sin(2.3 * t + b),
                                                TABLE_HEIGHT + .32 + .27 * math.sin(3.2 * t + c))

    def _command_to(self, side, goal, scale=1.):
        target = self.data.ctrl[self._arm_ids[side][2]]
        bounded = np.clip(goal, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return np.clip((bounded - target) / (JOINT_RATES * self.control_dt), -scale, scale)

    def _aim(self, side, bias=0., target_depth=.96, scale=1.):
        incoming = side * self.velocity[1]
        if self._last_hitter == side or incoming > -.30:
            return self._command_to(side, HOME, scale)
        strike_depth = 1.39
        predicted_x, _, _, _ = _flight_to_depth(self.ball, self.velocity, -side * strike_depth)
        strike_depth += .19 * min(1., max(0., (abs(predicted_x) - .24) / .23))
        strike_y = -side * strike_depth
        time_to_contact = side * (self.ball[1] - strike_y) / -incoming
        if time_to_contact < -.075:
            return np.zeros(4)
        if time_to_contact >= 0.:
            x, z, vy, vz = _flight_to_depth(self.ball, self.velocity, strike_y)
        else:
            x, _, z = self.ball + self.velocity * time_to_contact
            _, vy, vz = self.velocity
        z = max(TABLE_HEIGHT + .071, min(1.43, z))
        strike_speed = 2.28 * scale
        pitch = _racket_loft(side * vy, vz, z, strike_depth + target_depth, strike_speed)
        swing = max(-.22, min(.13, strike_speed * (.067 - time_to_contact)))
        blade_depth = strike_depth + (BALL_RADIUS + .006) * math.cos(pitch) - swing
        blade_z = z - (BALL_RADIUS + .006) * math.sin(pitch)
        goal = _arm_angles(x + bias, -side * blade_depth, blade_z, side, pitch)
        return self._command_to(side, goal, scale)

    def oracle_action(self):
        """Evaluator-only scripted action using kinematics and exact state."""
        return self._aim(1)

    def _choose_rival_response(self):
        self._rival_attempt = self._rng.random() < .90
        self._rival_bias = self._rng.normal(0., .023)
        self._rival_target = self._rng.uniform(.78, 1.12)

    def _aerodynamics(self):
        velocity = self.velocity
        speed = math.sqrt(float(velocity @ velocity))
        spin = self.spin
        # Free-flight coefficients were fitted to real SpinDOE trajectories in SI units.
        vx, vy, vz = velocity
        wx, wy, wz = spin
        drag = -BALL_MASS * AIR_DRAG * speed
        magnus = BALL_MASS * MAGNUS_COEFFICIENT
        force = self.data.xfrc_applied[self._ball_body]
        force[:3] = (drag * vx + magnus * (wy * vz - wz * vy),
                     drag * vy + magnus * (wz * vx - wx * vz),
                     drag * vz + magnus * (wx * vy - wy * vx))
        force[3:] = -BALL_INERTIA * SPIN_DECAY * spin

    def _event(self, kind, **details):
        self.events.append(dict(kind=kind, step=self.steps + 1, time=float(self.data.time),
                                ball=tuple(self.ball.tolist()), velocity=tuple(self.velocity.tolist()),
                                spin=tuple(self.spin.tolist()), **details))

    def _end(self, winner, reason):
        self.reward, self.terminated, self.reason = winner, True, reason
        self._event('point', winner=winner, reason=reason)

    def _table_contact(self, location, vertical, incoming):
        if vertical < .55 or location[2] < TABLE_HEIGHT - .035:
            self._end(self._last_hitter if self._received_bounce else -self._last_hitter, 'table_side')
            return
        side = 1 if location[1] < 0. else -1
        self._event('bounce', side=side, location=location, incoming=incoming)
        if self._serving and not self._server_bounce:
            if side != self._last_hitter:
                self._end(-self._last_hitter, 'serve_missed_own_half')
            else:
                self._server_bounce = True
        elif side == self._last_hitter:
            self._end(-self._last_hitter, 'own_half')
        elif self._received_bounce:
            self._end(self._last_hitter, 'second_bounce')
        elif self._serving and self._serve_net:
            self.truncated, self.reason = True, 'serve_let'
            self._event('let', reason='net_serve')
        else:
            self._received_bounce = True
            if not self._serving:
                if self._last_hitter == 1:
                    self.legal_returns += 1
                else:
                    self.rival_returns += 1
                self._event('legal_return', side=self._last_hitter, location=location)

    def _racket_contact(self, side, location, incoming):
        if side == 1:
            self.contacts += 1
        else:
            self.rival_contacts += 1
        self._event('contact', side=side, location=location, incoming=incoming,
                    racket=tuple(self.data.geom_xpos[self._racket_ids[side]].tolist()))
        if side == self._last_hitter:
            self._end(-side, 'double_hit')
        elif not self._received_bounce:
            self._end(-side, 'volley')
        else:
            self._last_hitter, self._received_bounce, self._serving = side, False, False
            if side == 1:
                self._choose_rival_response()

    def _contacts(self, incoming):
        if not (self.data.ncon or self._previous_touches):
            return
        ball_id = self._geom_ids['ball']
        touches = {}
        for contact in self.data.contact:
            if contact.efc_address < 0 or ball_id not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == ball_id else contact.geom[0])
            touches[other] = (tuple(contact.pos.tolist()), abs(float(contact.frame[2])))
        for other, (location, vertical) in touches.items():
            if other in self._previous_touches:
                continue
            if other == self._geom_ids['table']:
                self._table_contact(location, vertical, incoming)
            elif other == self._geom_ids['net']:
                self._serve_net |= self._serving
                self._event('net', location=location, incoming=incoming)
            elif other == self._racket_ids[1]:
                self._racket_contact(1, location, incoming)
            elif other == self._racket_ids[-1]:
                self._racket_contact(-1, location, incoming)
            elif other == self._geom_ids['floor']:
                self._end(self._last_hitter if self._received_bounce else -self._last_hitter, 'floor')
            if self.terminated or self.truncated:
                break
        for other, (location, _) in self._previous_touches.items():
            if other not in touches:
                collider = self._collider_names[other]
                self._event('separation', collider=collider, location=location)
        self._previous_touches = touches

    def step(self, action, *, render=True):
        """Advance one control interval and return RGB, scalar score and diagnostics."""
        if self.terminated or self.truncated:
            raise RuntimeError('reset after a point ends')
        command = np.asarray(action, dtype=float)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError('expected four finite joint rates')
        rival = self._aim(-1, self._rival_bias, self._rival_target, .87) if self._rival_attempt else (
            self._command_to(-1, HOME, .72))
        increments = np.concatenate((np.clip(command, -1., 1.), rival)) * self._joint_increments
        prior_events = len(self.events)
        for _ in range(self.frameskip * SOLVER_SUBSTEPS):
            control = self.data.ctrl
            control += increments
            np.clip(control, self._joint_low, self._joint_high, out=control)
            self._aerodynamics()
            incoming = tuple(self.velocity.tolist())
            mujoco.mj_step(self.model, self.data)
            self._contacts(incoming)
            if self.terminated or self.truncated:
                break
            x, y, z = self.ball
            if abs(x) > 2.05 or abs(y) > 2.62 or z > 3.8:
                self._end(self._last_hitter if self._received_bounce else -self._last_hitter, 'out')
                break
        if self.steps + 1 >= self.max_steps and not (self.terminated or self.truncated):
            self.truncated, self.reason = True, 'timeout'
            self._event('timeout')
        self.steps += 1
        self._place_decoy()
        info = self.point_result()
        info['events'] = tuple(self.events[prior_events:])
        return self.render_cameras() if render else None, self.reward, self.terminated, self.truncated, info

    def point_result(self):
        """Return evaluation-only contact, return, fault and score counters."""
        return dict(contacts=self.contacts, legal_returns=self.legal_returns,
                    rival_contacts=self.rival_contacts, rival_returns=self.rival_returns,
                    points_for=int(self.reward > 0), points_against=int(self.reward < 0),
                    steps=self.steps, frames=self.steps, reason=self.reason)

    def _scene_pose(self, side):
        _, _, _, base, elbow = self._arm_ids[side]
        racket = self._racket_ids[side]
        table_origin = np.array((0., 0., TABLE_HEIGHT))
        return (self.data.xpos[base] - table_origin, self.data.xpos[elbow] - table_origin,
                self.data.geom_xpos[racket] - table_origin,
                self.data.geom_xmat[racket].reshape(3, 3)[:, 2])

    @property
    def cameras(self):
        """Public camera calibration, with world heights relative to the tabletop."""
        if self._rig is None:
            from .tt4_camera import cameras

            self._rig = cameras(view=self.view, width=self.width, height=self.height,
                                baseline=self.baseline)
        return self._rig

    def _camera_rig(self):
        return self.cameras

    def _apply_camera_rig(self):
        for name, optics in zip(('eye_left', 'eye_right'), self._camera_rig()):
            camera = self.model.camera(name)
            camera.pos[:] = np.asarray(optics.eye) + (0., 0., TABLE_HEIGHT)
            right, up, forward = optics.basis
            rotation = np.column_stack((right, up, -forward))
            mujoco.mju_mat2Quat(camera.quat, np.ascontiguousarray(rotation).reshape(-1))
            camera.fovy[0] = optics.fovy
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def render_cameras(self, mode=None):
        """Capture RGB in the native renderer or its CPU perspective counterpart."""
        mujoco.mj_kinematics(self.model, self.data)
        backend = self.render_mode if mode is None else mode
        if backend == 'mujoco':
            if self._renderer is None:
                self._apply_camera_rig()
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            mujoco.mj_camlight(self.model, self.data)
            result = []
            for name in ('eye_left', 'eye_right')[:len(self._camera_rig())]:
                self._renderer.update_scene(self.data, camera=name)
                result.append(self._renderer.render())
            return np.stack(result)
        if backend == 'projected':
            from .tt4_camera import render_scene

            origin = np.array((0., 0., TABLE_HEIGHT))
            decoy = self.data.mocap_pos[self._decoy] - origin if self.distractor else None
            colors = tuple(tuple(map(int, rgb)) for rgb in self._colors)
            return np.stack(render_scene(self._camera_rig(), self._scene_pose(1), self._scene_pose(-1),
                                         self.ball - origin, colors, decoy=decoy,
                                         ball_radius=BALL_RADIUS, paddle_radius=PADDLE_RADIUS))
        raise ValueError('expected a known rendering backend')

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

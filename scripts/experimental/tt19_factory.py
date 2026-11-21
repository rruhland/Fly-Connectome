"""A factory-calibrated SO-101 controller with RGB-only ball observations."""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_arena import TABLE_HEIGHT
from .tt4_camera import PerspectiveCamera, triangulate
from .tt18_view import ImageColors, OBJECT_HUES
from .tt19_flight import optical_start
from .tt19_so101 import (_BLADE_DEPTH, _BLADE_TWIST, _GRASP_POS, AXES, MINI_TABLE, RATES,
                         SO101_HOME, SOURCE_MODEL)
from .tt19_view import _blob
from .tt19_view_trial import CAMERAS


BOUNDS = (MINI_TABLE.table_half_width, MINI_TABLE.table_half_length)
HOME = np.array(SO101_HOME[:5])
RATE = np.array(RATES) / 30
CONTACT_OFFSET = .026
FRONT = np.array([math.cos(_BLADE_TWIST), math.sin(_BLADE_TWIST), 0.])
TOOL = _GRASP_POS - [0., 0., _BLADE_DEPTH]


class FactoryModel:
    """Forward kinematics and Jacobians encoded from the vendor model and tool survey."""

    def __init__(self):
        root = ET.parse(SOURCE_MODEL).getroot().find('worldbody/body')
        bodies = {element.find('joint').get('name'): element for element in root.iter('body')
                  if element.find('joint') is not None}
        self.transforms = []
        self.limits = []
        for name in AXES[:5]:
            item = bodies[name]
            joint = item.find('joint')
            if joint.get('axis') != '0 0 1' or joint.get('pos', '0 0 0') != '0 0 0':
                raise ValueError('the factory kinematic convention changed')
            q = np.fromstring(item.get('quat'), sep=' ')
            self.transforms.append((np.fromstring(item.get('pos'), sep=' '),
                                    Rotation.from_quat(np.roll(q, -1)).as_matrix()))
            self.limits.append(np.fromstring(joint.get('range'), sep=' '))
        self.low, self.high = np.asarray(self.limits).T
        self.base = np.array([0., -BOUNDS[1] - .375, .49 - TABLE_HEIGHT])
        self.basis = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    def pose(self, q):
        p, matrix = self.base, self.basis
        centers, axes = [], []
        for value, (offset, frame) in zip(q, self.transforms, strict=True):
            p = p + matrix @ offset
            matrix = matrix @ frame
            centers.append(p)
            axes.append(matrix[:, 2].copy())
            c, s = math.cos(value), math.sin(value)
            matrix = matrix @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        tool, normal = p + matrix @ TOOL, -matrix @ FRONT
        linear = np.cross(axes, tool - centers).T
        facing = np.cross(axes, normal).T
        return tool, normal, linear, facing

    def solve(self, target, direction, seed):
        q = np.asarray(seed).copy()
        for _ in range(9):
            pos, norm, j, jn = self.pose(q)
            delta = np.r_[target - pos, .10 * (direction - norm)]
            if np.linalg.norm(delta) <= .0025:
                break
            matrix = np.vstack((j, .10 * jn))
            move = np.linalg.solve(matrix.T @ matrix + .00037 * np.eye(5), matrix.T @ delta)
            next_q = np.clip(q + np.clip(move, -.25, .25), self.low, self.high)
            if max(abs(next_q - q)) < .002:
                break
            q = next_q
        return q


class StereoBall:
    """A fixed factory survey transforms simultaneous orange RGB sightings into metres."""

    def __init__(self, width, height):
        self.rig = tuple(PerspectiveCamera(eye, target, width, height, fovy)
                         for key in ('left', 'right') for _, eye, target, fovy in (CAMERAS[key],))

    def measure(self, images):
        blobs = [_blob(ImageColors(images[key]), OBJECT_HUES['ball'], tolerance=.027, minimum=90.,
                       area=1, span=max(6, images[key].shape[0] // 10)) for key in ('left', 'right')]
        if any(blob is None for blob in blobs):
            return None
        pixels = [blob.center for blob in blobs]
        center = triangulate(self.rig, *pixels)
        if center is None:
            return None
        point = np.asarray(center)
        if abs(point[0]) > .66 or abs(point[1]) > 1.68 or not -.22 < point[2] < .85:
            return None
        if any(np.linalg.norm(np.asarray(camera.project(point)[:2]) - pixel) > 1.4
               for camera, pixel in zip(self.rig, pixels, strict=True)):
            return None
        return point


def attack(center, incoming, law, response, serving, *, rate_factor=1., pitch_bias=0.):
    """Choose face and speed with a public impulse/flight model, in metres per frame."""
    angles, rates = np.meshgrid(np.linspace(.12, 1.02 if serving else .89, 21), np.linspace(0., .052, 10))
    angles, rates = angles.ravel(), rates.ravel() * rate_factor
    yaw = np.clip(-7. * incoming[0] - .27 * center[0], -.38, .38)
    sin, cos = np.sin(angles), np.cos(angles)
    normals = np.stack([math.sin(yaw) * cos, math.cos(yaw) * cos, sin], axis=1)
    tangents = np.stack([-math.sin(yaw) * sin, -math.cos(yaw) * sin, cos], axis=1)
    relative = incoming - np.column_stack([np.zeros_like(rates), rates, np.zeros_like(rates)])
    local = np.column_stack([np.sum(normals * relative, axis=1), np.sum(tangents * relative, axis=1)])
    impulse = local @ response
    outbound = incoming + normals * impulse[:, :1] + tangents * impulse[:, 1:]
    speed = np.maximum(.002, outbound[:, 1])
    vert = outbound[:, 2]
    start_y, start_z = center[1:]

    def landing_time(height, vertical):
        return (vertical + np.sqrt(np.maximum(0., vertical**2 + 2 * law.gravity * (height - .02)))) / law.gravity

    def distance(elapsed, velocity):
        return np.log1p(np.maximum(0., elapsed) * law.drag * velocity) / law.drag

    elapsed = landing_time(start_z, vert)
    first = start_y + distance(elapsed, speed)
    loss = 3. * np.maximum(.022 - outbound[:, 1], 0.) + .40 * rates
    if serving:
        loss += 9. * (np.maximum(first + .10, 0.) + np.maximum(-.75 - first, 0.))
        start_y, start_z = first, .02
        vert = (law.gravity * elapsed - vert) * law.restitution
        speed = speed * law.tangent / (1 + elapsed * law.drag * speed)
    travel = landing_time(start_z, vert)
    final = start_y + distance(travel, speed)
    to_net = np.expm1(np.clip(-start_y * law.drag, -1., 1.)) / (law.drag * speed)
    height = start_z + to_net * (vert - .5 * law.gravity * to_net)
    lateral_time = elapsed + travel if serving else travel
    final_x = center[0] + lateral_time * outbound[:, 0]
    loss += abs(final - .37) + 5. * np.maximum(.189 - height, 0.)
    loss += 7. * (np.maximum(.04 - final, 0.) + np.maximum(final - .76, 0.))
    loss += 6. * np.maximum(abs(final_x) - .39, 0.)
    loss += 1. * np.maximum(local[:, 0] + .005, 0.)
    picked = int(np.argmin(loss))
    pitch = np.clip(angles[picked] + pitch_bias, .04, 1.1)
    normal = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    return normal, float(rates[picked])


def moving_joint_rate(model, q, pace):
    _, _, j, jn = model.pose(q)
    jac = np.vstack((j, .06 * jn))
    task = np.array((0., pace, 0., 0., 0., 0.))
    motion = np.linalg.solve(jac.T @ jac + .0018 * np.eye(5), jac.T @ task)
    return np.clip(motion, -RATE[:5], RATE[:5])


def coast_cost(center, incoming, normal, blade_velocity, law, response):
    tangent = np.array([0., 0., 1.]) - normal[2] * normal
    tangent /= np.linalg.norm(tangent)
    relative = incoming - blade_velocity
    impulse = np.array([relative @ normal, relative @ tangent]) @ response
    out = incoming + normal * impulse[0] + tangent * impulse[1]
    forward = max(.004, out[1])
    to_net = np.expm1(-law.drag * center[1]) / (law.drag * forward)
    net = center[2] + to_net * (out[2] - .5 * law.gravity * to_net)
    to_table = (out[2] + math.sqrt(max(0., out[2]**2 + 2 * law.gravity * (center[2] - .02)))) / law.gravity
    stop_y = center[1] + np.log1p(law.drag * to_table * forward) / law.drag
    stop_x = center[0] + to_table * out[0]
    return (.7 * abs(stop_y - .49) + .5 * abs(stop_x) + 4 * max(.20 - net, 0.) +
            4 * max(.04 - stop_y, 0.) + 3 * max(stop_y - .75, 0.) + 4 * max(abs(stop_x) - .39, 0.))


@dataclass
class Strike:
    tick: float
    center: np.ndarray
    incoming: np.ndarray
    normal: np.ndarray
    rate: float
    q: np.ndarray


class FactoryPlayer:
    def __init__(self, factory, tracker, law, response, *, stabilize=False, reachable=False):
        self.factory, self.tracker = factory, tracker
        self.law, self.response = law, response
        self.stabilize = stabilize
        self.reachable = reachable

    def reset(self, server, knobs):
        self.server, self.knobs = server, knobs
        self.previous = None
        self.reversals = []
        self.history = []
        self.tick = -1
        self.plan = None
        self.target = HOME.copy()
        self.samples = []
        self.own_served = False
        self.last_q = None

    def ball_state(self):
        state = optical_start(self.history, self.law.gravity)
        if state is None:
            good = [(tick, pose) for tick, pose in self.history[-6:] if pose is not None]
            if len(good) >= 2 and self.tick - good[-1][0] <= 3:
                previous = optical_start(good, self.law.gravity)
                if previous is not None:
                    positions, velocities = self.law.trajectory(*previous, self.tick - good[-1][0], bounds=BOUNDS)
                    state = positions[-1], velocities[-1]
        return state

    def intercept(self, point, velocity):
        depth, pace, pitch_bias, time_bias = self.knobs
        if self.server == 1 and not self.own_served:
            z = .249 + depth
            square = velocity[2]**2 + 2 * self.law.gravity * (point[2] - z)
            if square < 0 or self.tick > 22 or abs(velocity[1]) > .021:
                return None
            eta = (velocity[2] + math.sqrt(square)) / self.law.gravity
            if not 0. <= eta < 29.:
                return None
            center = point + eta * velocity
            center[2] = z
            incoming = velocity.copy()
            incoming[2] = -math.sqrt(square)
        elif velocity[1] < -.017:
            positions, speeds = self.law.trajectory(point, velocity, 26, bounds=BOUNDS)
            choices = np.flatnonzero((positions[:, 1] < -.89) & (positions[:, 1] > -1.19)
                                     & (positions[:, 2] > .088) & (positions[:, 2] < .39))
            if not len(choices):
                return None
            target_y = -1.04 + depth
            i = choices[np.argmin(abs(positions[choices, 1] - target_y))]
            eta, center, incoming = float(i), positions[i], speeds[i]
        else:
            return None
        return eta + time_bias, center, incoming, pace, pitch_bias

    def prepare(self, q, state):
        details = self.intercept(*state)
        if details is None:
            return None
        eta, center, incoming, pace, pitch_bias = details
        normal, rate = attack(center, incoming, self.law, self.response.response_matrix,
                              self.server == 1 and not self.own_served, rate_factor=pace, pitch_bias=pitch_bias)
        target = center - CONTACT_OFFSET * normal
        guessed = q if self.plan is None else self.plan.q
        if self.reachable and (self.server == -1 or self.own_served):
            if eta > 18:
                return None
            return self.reachable_return(q, guessed, eta, center, normal, rate, state)
        q_hit = self.factory.solve(target, normal, guessed)
        return Strike(float(self.tick + eta), center, incoming, normal, rate, q_hit)

    def reachable_return(self, q, guessed, eta, center, normal, pace, state):
        yaw = math.atan2(normal[0], normal[1])
        pitch = math.asin(normal[2])
        positions, velocities = self.law.trajectory(*state, 23, bounds=BOUNDS)
        options = []
        for frame in (-2, 0, 2):
            horizon = eta + frame
            if not 0 <= horizon < 22:
                continue
            arrival = int(np.clip(round(horizon - self.knobs[3]), 0, 22))
            ball, incoming = positions[arrival], velocities[arrival]
            if not -.13 < ball[2] - center[2] < .10 or not -1.16 < ball[1] < -.86:
                continue
            for loft in [pitch - .15, pitch, pitch + .18]:
                wanted = np.array([math.sin(yaw) * math.cos(loft), math.cos(yaw) * math.cos(loft), math.sin(loft)])
                tangent = np.array([0., 0., 1.]) - wanted[2] * wanted
                tangent /= np.linalg.norm(tangent)
                target = ball - CONTACT_OFFSET * wanted + .017 * tangent
                candidate = self.factory.solve(target, wanted, guessed)
                blade, face, jacobian, _ = self.factory.pose(candidate)
                move = moving_joint_rate(self.factory, candidate, pace)
                exit_cost = coast_cost(ball, incoming, face, jacobian @ move, self.law, self.response.response_matrix)
                position_cost = np.linalg.norm(target - blade)
                time_cost = float(np.max(abs(candidate - q) / RATE[:5]) - horizon)
                loss = exit_cost + 25. * position_cost + .11 * max(time_cost, 0.) + .024 * abs(frame)
                options.append((loss, Strike(float(self.tick + horizon), ball, incoming, face, pace, candidate)))
        return None if not options else min(options, key=lambda row: row[0])[1]

    def choose(self, images, encoders):
        self.tick += 1
        q = encoders[:5]
        ball = None if images is None else self.tracker.measure(images)
        self.history.append((self.tick, ball))
        state = self.ball_state()
        if state is not None and self.previous is not None:
            point, velocity = state
            if velocity[1] > .022 and (self.previous[1][1] < -.014 or
                                      (self.server == 1 and self.previous[1][2] < -.025)) and point[1] < -.54:
                self.reversals.append(self.tick)
                self.own_served = True
                self.plan = None
        if state is not None:
            self.previous = state
        if self.plan is not None and self.tick - self.plan.tick > 4.:
            self.plan = None
        timely = self.plan is None or self.plan.tick - self.tick > (2. if self.stabilize else 3.)
        tracked = ball is not None if self.stabilize else self.tick % 2 == 0
        if state is not None and timely and tracked:
            proposed = self.prepare(q, state)
            if proposed is not None:
                self.plan = proposed
        plan = self.plan
        if plan is None:
            desired = HOME
        else:
            motion = moving_joint_rate(self.factory, plan.q, plan.rate)
            due = self.tick + 1.65 - plan.tick
            desired = plan.q + np.clip(due, -3.6, 3.2) * motion
            if self.stabilize and -4. <= due <= 4. and self.last_q is not None:
                # Command roll against predicted pan/pitch at contact, even
                # while the other motors are still moving towards the pose.
                follow = q + np.clip(1.4 * (q - self.last_q), -.21, .21)
                follow[-1] = desired[-1]
                for _ in range(2):
                    _, normal, _, axes = self.factory.pose(follow)
                    jacobian = axes[0, -1]
                    if abs(jacobian) > .20:
                        follow[-1] += np.clip((plan.normal[0] - normal[0]) / jacobian, -.60, .60)
                desired[-1] = follow[-1]
        desired = np.clip(desired, self.factory.low, self.factory.high)
        # Commands integrate position targets; encoders expose tracking lag.
        correction = .18 * np.clip(self.target - q, -.11, .11)
        action = np.clip((desired - self.target + correction) / RATE[:5], -1., 1.)
        self.target = np.clip(self.target + RATE[:5] * action, self.factory.low, self.factory.high)
        self.last_q = q.copy()
        self.samples.append(dict(ball=None if ball is None else ball.tolist(),
                                 plan=None if plan is None else [round(plan.tick, 4), plan.center.tolist(),
                                                                  plan.normal.tolist(), round(plan.rate, 5)]))
        return np.r_[action, 0.]

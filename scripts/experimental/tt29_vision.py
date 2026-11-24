"""Causal ball flight from fixed or encoder-compensated native RGB cameras."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.spatial.transform import Rotation

from .tt18_view import CameraFit
from .tt19_so101 import AXES, GRIP_BODY, WRIST_CAMERA
from .tt19_view_trial import CAMERAS
from .tt4_arena import BALL_RADIUS, TABLE_HEIGHT
from .tt27_world_camera import Registration


BOUNCE_Z = TABLE_HEIGHT + BALL_RADIUS
VERTICAL = np.array([0., 0., -9.81])
IDENTITY = np.eye(3)


def matrix(quaternion):
    return Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()


@dataclass(frozen=True)
class Lens:
    width: int
    height: int
    fovy: float

    @property
    def focal(self):
        return self.height / (2. * np.tan(np.deg2rad(self.fovy) / 2.))

    def at(self, eye, rotation):
        axes = rotation.T * np.array([[1.], [-1.], [-1.]])
        return CameraFit(eye, axes, self.focal, self.width, self.height)


@dataclass(frozen=True)
class JointLink:
    """One surveyed rigid transform in the wrist camera's serial chain."""

    position: np.ndarray
    rotation: np.ndarray
    encoder: int | None
    axis: np.ndarray | None = None
    pivot: np.ndarray | None = None
    reference: float = 0.


@dataclass(frozen=True)
class EncoderCamera:
    links: tuple[JointLink, ...]
    lens: Lens
    gripper_links: int

    @classmethod
    def from_model(cls, model, lens):
        # Copy static mount dimensions only. The learner owns no simulation data.
        camera = model.camera(WRIST_CAMERA)
        ids = {model.joint(f'learner_{name}').id: i for i, name in enumerate(AXES)}
        chain = [JointLink(camera.pos.copy(), matrix(camera.quat), None)]
        body_id = int(camera.bodyid[0])
        gripper_index = None
        while body_id:
            body = model.body(body_id)
            if body.name == GRIP_BODY:
                gripper_index = len(chain)
            count = int(body.jntnum[0])
            if count:
                if count != 1 or int(body.jntadr[0]) not in ids:
                    raise ValueError('wrist camera must have a serial chain of measured hinge joints')
                joint = model.joint(int(body.jntadr[0]))
                chain.append(JointLink(body.pos.copy(), matrix(body.quat), ids[joint.id], joint.axis.copy(),
                                       joint.pos.copy(), float(model.qpos0[joint.qposadr[0]])))
            else:
                chain.append(JointLink(body.pos.copy(), matrix(body.quat), None))
            body_id = int(body.parentid[0])
        if gripper_index is None:
            raise ValueError('wrist lens must descend from the gripper')
        return cls(tuple(reversed(chain)), lens, len(chain) - gripper_index)

    def remount(self, position, rotation, *, focal_px=None, reference_height=144):
        """Install a carried estimate of the lens pose in the gripper frame."""
        lens = self.lens
        if focal_px is not None:
            fovy = np.rad2deg(2 * np.arctan(reference_height / (2 * focal_px)))
            lens = Lens(lens.width, lens.height, fovy)
        links = self.links[:self.gripper_links] + (JointLink(np.asarray(position), np.asarray(rotation), None),)
        return EncoderCamera(links, lens, self.gripper_links)

    def at(self, joint_angles):
        eye = np.zeros(3)
        rotation = IDENTITY
        for link in self.links:
            eye += rotation @ link.position
            rotation = rotation @ link.rotation
            if link.encoder is not None:
                turn = Rotation.from_rotvec(link.axis * (joint_angles[link.encoder] - link.reference)).as_matrix()
                eye += rotation @ (link.pivot - turn @ link.pivot)
                rotation = rotation @ turn
        return self.lens.at(eye, rotation)


class Rig:
    """Retain external registration and compute wrist pose from capture-time encoders."""

    def __init__(self, fixed, wrist, reference=None):
        self.fixed = dict(fixed)
        self.wrist = wrist
        self.reference = {} if reference is None else {
            stream: Registration(frame, np.rad2deg(2. * np.arctan(self.fixed[stream].height / (2. * self.fixed[stream].focal))))
            for stream, frame in reference.items() if stream in self.fixed
        }
        self.nominal = self.fixed.copy()
        self.checked = set()
        self.registrations = []

    @classmethod
    def from_arena(cls, arena, *, reference=None):
        model = arena.model
        fixed = {}
        for stream, (name, *_) in CAMERAS.items():
            camera = model.camera(name)
            lens = Lens(arena.width, arena.height, float(camera.fovy[0]))
            fixed[stream] = lens.at(camera.pos.copy(), matrix(camera.quat))
        wrist_camera = model.camera(WRIST_CAMERA)
        lens = Lens(arena.width, arena.height, float(wrist_camera.fovy[0]))
        return cls(fixed, EncoderCamera.from_model(model, lens), reference=reference)

    def begin(self):
        self.checked.clear()
        self.registrations.clear()

    def views(self, frames, q):
        for stream, image in frames.items():
            if stream == 'wrist':
                yield stream, self.wrist.at(q)
                continue
            if stream not in self.fixed:
                continue
            if stream in self.reference and stream not in self.checked:
                previous = self.nominal[stream]
                rotation, support, _ = self.reference[stream].calibrate(image, False, 12)
                if support:
                    self.fixed[stream] = CameraFit(previous.eye, rotation.T @ previous.rotation, previous.focal,
                                                   previous.width, previous.height, markers=support)
                self.checked.add(stream)
                self.registrations.append((stream, support))
            yield stream, self.fixed[stream]


@dataclass(frozen=True)
class Sighting:
    center: np.ndarray
    radius: float
    clipped: bool
    area: int


def ball_pixels(rgb, predicted=None):
    """Isolate the orange ball in ordinary RGB and measure its projected area."""
    r, g, b = np.moveaxis(rgb, -1, 0)
    # The integer differences avoid three full floating point image conversions.
    high = r.astype(np.int16)
    chroma = high - b
    green = g.astype(np.int16) - b
    mask = (chroma >= 64) & (100 * green > 25 * chroma) & (100 * green < 62 * chroma) & (100 * b.astype(np.int16) < 24 * high)
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    left, right, top, bottom = x.min(), x.max(), y.min(), y.max()
    regions, _ = label(mask[top:bottom + 1, left:right + 1], structure=np.ones((3, 3)))
    best = None
    max_span = min(rgb.shape[:2]) // 3
    for index, (ys, xs) in enumerate(find_objects(regions), 1):
        width, height = xs.stop - xs.start, ys.stop - ys.start
        if max(width, height) > max_span:
            continue
        cy, cx = np.nonzero(regions[ys, xs] == index)
        area = len(cx)
        center = np.array([left + xs.start + cx.mean(), top + ys.start + cy.mean()])
        compactness = min(width, height) / max(width, height) * area / (width * height)
        clipped = (min(left + xs.start, top + ys.start) == 0 or
                   left + xs.stop == rgb.shape[1] or top + ys.stop == rgb.shape[0] or compactness < .48)
        radius = np.sqrt(area / np.pi)
        cost = -min(radius, 9.) - 3. * compactness
        if predicted is not None:
            cost += min(25., np.linalg.norm(center - predicted) * .3)
        if best is None or cost < best[0]:
            best = cost, Sighting(center, radius, clipped, area)
    return None if best is None else best[1]


@dataclass(frozen=True)
class RaySample:
    time: float
    stream: str
    eye: np.ndarray
    direction: np.ndarray
    range_m: float
    depth_weight: float
    angle_weight: float

    @classmethod
    def measure(cls, time, stream, camera, sight, radius_factor):
        ray = camera.ray(sight.center)
        obliquity = np.linalg.norm(ray)
        radius = radius_factor * sight.radius
        distance = BALL_RADIUS * camera.focal * obliquity ** 1.5 / radius
        angular = camera.focal / max(.30, distance)
        depth = radius / max(.30, distance)
        if sight.clipped or sight.area == 1:
            depth = 0.0
        return cls(time, stream, camera.eye, ray / obliquity, distance, depth, angular)


def flight_terms(dt, damping):
    dt = np.asarray(dt)
    if damping < 1e-5:
        return dt, np.square(dt) * .5
    travel = -np.expm1(-damping * dt) / damping
    return travel, (dt - travel) / damping


@dataclass(frozen=True)
class Flight:
    position: np.ndarray
    velocity: np.ndarray
    time: float
    damping: float
    rms: float
    depth_sigma: float
    frames: int
    bounced: bool

    def advance(self, time):
        t = time - self.time
        travel, fall = flight_terms(t, self.damping)
        p = self.position + travel * self.velocity + fall * VERTICAL
        v = self.velocity * np.exp(-self.damping * t) + travel * VERTICAL
        return p, v


class FlightPractice:
    """Fit persistent silhouette scale and table rebound from completed optical observations."""

    def __init__(self):
        self.radius_samples = {stream: deque(maxlen=80) for stream in (*CAMERAS, 'wrist')}
        self.rebounds = deque(maxlen=16)

    def radius_factor(self, stream):
        values = self.radius_samples[stream]
        return float(np.clip(np.median(values), .74, 1.45)) if len(values) >= 8 else 1.

    @property
    def rebound(self):
        return np.median(self.rebounds, axis=0) if len(self.rebounds) >= 4 else np.array([.95, .92])

    def observe_ranges(self, samples):
        for a in samples:
            if a.depth_weight == 0.:
                continue
            for b in samples:
                mutual = a.direction @ b.direction
                if a is b or abs(mutual) > .97:
                    continue
                offset = b.eye - a.eye
                t, u = (offset @ a.direction, offset @ b.direction)
                ra = (t - mutual * u) / (1. - mutual ** 2)
                rb = (mutual * t - u) / (1. - mutual ** 2)
                gap = np.linalg.norm(a.eye + ra * a.direction - b.eye - rb * b.direction)
                if ra > .12 and rb > .12 and gap < .012:
                    self.radius_samples[a.stream].append(a.range_m / ra * self.radius_factor(a.stream))
                    break


def measurement_blocks(samples):
    direction = np.stack([s.direction for s in samples])
    eye = np.stack([s.eye for s in samples])
    ortho = IDENTITY - direction[:, :, None] * direction[:, None, :]
    angle = np.clip([s.angle_weight for s in samples], 40., 600.)
    _, index, count = np.unique([s.time for s in samples], return_inverse=True, return_counts=True)
    depth = np.array([s.depth_weight for s in samples]) * np.where(count[index] >= 2, 1., 2.60)
    projection = np.concatenate((ortho * angle[:, None, None], direction[:, None, :] * depth[:, None, None]), axis=1)
    desired = np.einsum('nji,ni->nj', projection, eye)
    desired[:, 3] += depth * [s.range_m for s in samples]
    return projection, desired


def solve(blocks, desired, regularization=None):
    shape = blocks.shape
    x = blocks.reshape(-1, shape[-1])
    y = desired.ravel()
    if regularization is not None:
        priors, labels = regularization
        x = np.concatenate((x, priors), axis=0)
        y = np.concatenate((y, labels))
    normal = x.T @ x
    normal.flat[::len(normal) + 1] += 1e-6
    beta = np.linalg.solve(normal, x.T @ y)
    rms = np.sqrt(np.mean(np.square(y - x @ beta)))
    return beta, rms, normal


class OpticalFlight:
    """A batch of recent rays is fit by recursive fixed-window, regularized least squares."""

    def __init__(self, rig, practice=None, *, learn=False, window=.135):
        self.rig = rig
        self.practice = FlightPractice() if practice is None else practice
        self.learn = learn
        self.window = window
        self.rays = deque()
        self.latest = None
        self.bounce_time = None
        self.bounce_state = None
        self.last_capture = None
        self.last_frame_views = ()
        self.trace = []

    def observe(self, capture, frames, encoders):
        """Process a new synchronized exposure using its original encoder timestamp."""
        if self.last_capture is not None and capture <= self.last_capture:
            raise ValueError('camera captures must be passed once, in timestamp order')
        self.last_capture = capture
        samples = []
        for stream, camera in self.rig.views(frames, encoders):
            predicted = None
            if self.latest is not None and capture - self.latest.time < .08:
                pixel = camera.project(self.latest.advance(capture)[0])
                if pixel[2] > .05:
                    predicted = pixel[:2]
            sight = ball_pixels(frames[stream], predicted)
            if sight is not None:
                samples.append(RaySample.measure(capture, stream, camera, sight, self.practice.radius_factor(stream)))
        self.last_frame_views = tuple(sample.stream for sample in samples)
        if self.learn and len(samples) >= 2:
            self.practice.observe_ranges(samples)
        if not samples:
            return self.latest
        self.rays.extend(samples)
        span = self.window if len(samples) > 1 else self.window * 1.35
        while self.rays[0].time < capture - span:
            self.rays.popleft()
        self._fit(capture)
        return self.latest

    def estimate(self, time, *, max_age=.080):
        """Return a real-time estimate or abstain when its latest evidence is too old."""
        s = self.latest
        if s is None or time - s.time > max_age:
            return None
        return s.advance(time)

    def _fit(self, capture):
        samples = list(self.rays)
        times = np.asarray([s.time for s in samples])
        unique = np.unique(times)
        if len(unique) < 3 or capture - unique[0] < .024:
            return
        if self.bounce_time is not None and capture - self.bounce_time > self.window * 1.6:
            self.bounce_time = None
        pos_speed = 4. if self.latest is None else np.linalg.norm(self.latest.velocity)
        damp = .133 * np.clip(pos_speed, 2.5, 12.)
        start = 0 if self.bounce_time is None else int(np.searchsorted(times, self.bounce_time))
        if len(np.unique(times[start:])) < 3:
            start = 0
        projection, target = measurement_blocks(samples)
        dt = times - capture
        travel, fall = flight_terms(dt[start:], damp)
        offsets = fall[:, None] * VERTICAL
        a = projection[start:]
        blocks = np.concatenate((a, a * travel[:, None, None]), axis=2)
        desired = target[start:] - np.einsum('nji,ni->nj', a, offsets)
        beta, error, normal = solve(blocks, desired)
        p, v = beta[:3], beta[3:]
        # Far-end hits or reversals break the local flight window.
        recent = self.latest
        if recent is not None and error > 2.6 and abs(recent.velocity[1]) > .7:
            for i in range(len(unique) - 4, max(-1, len(unique) - 6), -1):
                if i < 1:
                    break
                restart = int(np.searchsorted(times, unique[i]))
                if restart <= start:
                    continue
                candidate, candidate_error, cov = solve(blocks[restart-start:], desired[restart-start:])
                c_p, c_v = candidate[:3], candidate[3:]
                if (candidate_error < min(1.8, .55 * error) and c_v[1] * recent.velocity[1] < -.9 and
                        abs(c_p[0]) < 1.6 and abs(c_p[1]) < 2.8 and .66 < c_p[2] < 1.75):
                    while self.rays[0].time < unique[i]:
                        self.rays.popleft()
                    self.bounce_time = None
                    self.bounce_state = None
                    p, v, error, normal = c_p, c_v, candidate_error, cov
                    start = restart
                    break
        bounced = self.bounce_time is not None
        if not bounced and len(unique) >= 5 and p[2] < BOUNCE_Z + .23 and abs(p[1]) < 1.7 and error > .40:
            bounce = self._split_bounce(times, unique, projection, target, damp, error)
            if bounce is not None:
                p, v, error, sigma, impact, before, after = bounce
                self.bounce_time = impact
                self.bounce_state = (before, after)
                bounced = True
                if self.learn and abs(before[1]) > 1. and before[2] < -.8:
                    value = np.array([after[1] / before[1], -after[2] / before[2]])
                    if .60 < value[0] < 1.25 and .45 < value[1] < 1.25:
                        self.practice.rebounds.append(value)
            else:
                sigma = self._sigma(normal, error)
        else:
            sigma = self._sigma(normal, error)
        valid = (abs(p[0]) < 1.65 and abs(p[1]) < 3.2 and .60 < p[2] < 2.4 and
                 np.linalg.norm(v) < 19. and error < 7. and sigma < .40)
        if valid:
            self.latest = Flight(p, v, capture, damp, float(error), sigma, len(np.unique(times[start:])), bounced)
            self.trace.append(self.latest)

    @staticmethod
    def _sigma(normal, error, count=3):
        return float(max(.45, error) * np.sqrt(np.max(np.diag(np.linalg.inv(normal))[:count])))

    def _split_bounce(self, times, unique, projection, target, damping, smooth_error):
        ratio, restitution = self.practice.rebound
        relation = np.diag([ratio, ratio, -restitution])
        regularization = np.concatenate((-relation, IDENTITY), axis=1) * 3.4
        priors = np.column_stack((np.zeros((3, 2)), regularization))
        height = np.array([0., 0., BOUNCE_Z])
        best = None
        for first, second in zip(unique[1:-2], unique[2:-1], strict=True):
            for weight in (.18, .50, .82):
                impact = first + (second - first) * weight
                dt = times - impact
                travel, fall = flight_terms(dt, damping)
                base = height + fall[:, None] * VERTICAL
                weighted = projection * travel[:, None, None]
                before = (dt < 0.)[:, None, None]
                blocks = np.concatenate((projection[..., :2], weighted * before, weighted * ~before), axis=2)
                desired = target - np.einsum('nji,ni->nj', projection, base)
                fit, cost, normal = solve(blocks, desired, (priors, np.zeros(3)))
                incoming, outgoing = fit[2:5], fit[5:]
                if (cost >= min(.86 * smooth_error, 2.8) or abs(fit[0]) > .83 or abs(fit[1]) > 1.45 or
                        incoming[2] > -.45 or outgoing[2] < .20 or outgoing[1] * incoming[1] < .7 or
                        not .58 < outgoing[1] / incoming[1] < 1.3 or
                        not .4 < outgoing[2] / -incoming[2] < 1.35):
                    continue
                if best is None or cost < best[0]:
                    t = dt[-1]
                    end = np.r_[fit[:2], BOUNCE_Z] + travel[-1] * outgoing + fall[-1] * VERTICAL
                    speed = outgoing * np.exp(-damping * t) + travel[-1] * VERTICAL
                    best = cost, (end, speed, cost, self._sigma(normal, cost, 2), impact, incoming, outgoing)
        return None if best is None else best[1]

"""An idealized, force-driven far-court carriage with native rigid paddle contact."""

from dataclasses import dataclass
from itertools import product
import math
import xml.etree.ElementTree as ET

from mujoco import rollout
import numpy as np

from .tt4_arena import BALL_INERTIA, BALL_MASS, BALL_RADIUS, PADDLE_RADIUS, SPIN_DECAY, TABLE_HEIGHT, mujoco
from .tt29_rally_physics import BallFlight, DRAG, LIFT, NET_TOP, RallyArena, TABLE_TOP


CARRIAGE = "tt29_partner_carriage"
MASS = .65
INERTIA = .004
LINEAR_SPEED = 3.45
LINEAR_ACCELERATION = 46.
ANGULAR_SPEED = 10.
ANGULAR_ACCELERATION = 175.
HOME_NORMAL = np.array([0., -1., 0.20])
HOME_NORMAL /= np.linalg.norm(HOME_NORMAL)
LANES = np.array([
    [-.055, -1.055, 1.154, 3.9], [.052, -1.070, 1.182, 4.3],
    [-.021, -1.040, 1.161, 4.5], [.040, -1.090, 1.180, 4.1],
    [-.038, -1.065, 1.210, 4.15], [.056, -1.052, 1.196, 4.5],
])


DEEP_LANES = LANES.copy()
DEEP_LANES[:, 1] -= .27
DEEP_LANES[:, 2] -= .075
DEEP_LANES[:, 3] -= .45
FOREHAND_LANES = np.array([
    [.222, -1.296, 1.035, 3.50], [.253, -1.315, 1.055, 3.65],
    [.235, -1.286, 1.025, 3.40], [.250, -1.325, 1.047, 3.56],
    [.211, -1.311, 1.061, 3.46], [.262, -1.288, 1.042, 3.60],
])


def _string(values):
    return " ".join(map(str, values))


def _cap(vector, magnitude):
    norm = np.linalg.norm(vector)
    return vector * min(1., magnitude / max(norm, 1e-12))


def _face(normal):
    orientation = np.empty(4)
    mujoco.mju_quatZ2Vec(orientation, normal)
    return orientation


def add_carriage(document):
    """Preserve the original collision geom and referee name on the moving stage."""
    world = document.find("worldbody")
    base = world.find("body[@name='rival_base']")
    parents = (body for body in base.iter("body") if body.find("geom[@name='rival_racket']") is not None)
    parent = next(parents)
    blade = parent.find("geom[@name='rival_racket']")
    parent.remove(blade)
    if parent.find("inertial") is None:
        ET.SubElement(parent, "inertial", pos="0 0 0", mass=".13", diaginertia=".0002 .0002 .0004")
    for g in base.iter("geom"):
        g.attrib.update(rgba="0 0 0 0", contype="0", conaffinity="0")
        g.attrib.pop("material", None)
    end = float(world.find("geom[@name='table']").get("size").split()[1])
    stage = ET.SubElement(world, "body", name=CARRIAGE, pos=_string((0., end + .30, 1.14)),
                          quat=_string(_face(HOME_NORMAL)), gravcomp="1")
    ET.SubElement(stage, "freejoint", name=CARRIAGE)
    ET.SubElement(stage, "inertial", pos="0 0 0", mass=str(MASS), diaginertia=_string([INERTIA] * 3))
    blade.attrib.update(pos="0 0 0", quat="1 0 0 0")
    stage.append(blade)
    ET.SubElement(stage, "geom", name="tt29_partner_racket_mount", type="cylinder", pos="0 0 -.025",
                  size=".029 .019", rgba=".26 .32 .37 1")
    # A static rail and its posts identify the idealized opponent in stadium views.
    # Their rails are outside the struck-ball corridor. Only the blade is a collider.
    gray = ".29 .36 .40 1"
    rail = ET.SubElement(world, "body", name="tt29_partner_static_frame", pos=_string((0., end + .44, 0.)))
    ET.SubElement(rail, "geom", type="box", pos="0 0 1.53", size=".97 .037 .029", rgba=gray)
    for x in (-.93, .93):
        ET.SubElement(rail, "geom", type="box", pos=_string((x, 0., .765)), size=".031 .035 .765", rgba=gray)
    return document


class IsolatedImpact:
    """Screen blade normals and speeds with the native sphere-cylinder contact."""

    def __init__(self, arena):
        model = arena.model
        geoms = []
        for name, kind, dims in (("ball", "sphere", 1), ("rival_racket", "cylinder", 2)):
            g = model.geom(name)
            attrs = dict(name=name, type=kind, size=_string(g.size[:dims]), contype=g.contype[0],
                         conaffinity=g.conaffinity[0], solref=_string(g.solref), solimp=_string(g.solimp),
                         condim=g.condim[0], priority=g.priority[0], friction=_string(g.friction),
                         margin=g.margin[0], gap=g.gap[0])
            geoms.append("<geom " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>")
        self.dt = model.opt.timestep
        self.half_thickness = float(model.geom("rival_racket").size[1])
        self.model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option timestep="{self.dt}" gravity="0 0 -9.81" integrator="implicitfast" iterations="25"/>'
            f'<worldbody><body><freejoint/><inertial pos="0 0 0" mass="{BALL_MASS}" '
            f'diaginertia="{_string(model.body_inertia[arena._ball_body])}"/>{geoms[0]}</body>'
            f'<body gravcomp="1"><freejoint/><inertial pos="0 0 0" mass="{MASS}" '
            f'diaginertia="{_string([INERTIA]*3)}"/>{geoms[1]}</body></worldbody></mujoco>'
        )
        self.data = mujoco.MjData(self.model)

    def solve(self, incoming, normal, velocity):
        dt = self.dt
        n = len(incoming)
        q = np.zeros((n, 1 + self.model.nq + self.model.nv))
        q[:, 1:4] = incoming[:, 1:4]
        q[:, 4] = 1
        # Relative approach creates a real contact after a short separation.
        delta = np.maximum(.001, (np.sum((velocity - incoming[:, 4:7]) * normal, axis=1)) * dt) + .001
        center = incoming[:, 1:4] - (BALL_RADIUS + self.half_thickness + delta[:, None]) * normal
        q[:, 8:11] = center
        q[:, 11:15] = np.array([_face(row) for row in normal])
        q[:, 15:21] = incoming[:, 4:10]
        q[:, 21:24] = velocity
        with rollout.Rollout() as engine:
            state, _ = engine.rollout(self.model, self.data, q, nstep=9)
        leave = state[:, -1]
        w = np.array([_world_spin(s[18:21], s[4:8]) for s in leave])
        return leave[:, 1:4], leave[:, 15:18], w, center, q[:, 11:15]


    def verify_approach(self, incoming, stroke):
        """Track the actual planned drive around first contact in a native solver."""
        model, data, dt = self.model, self.data, self.dt
        now = stroke.start + incoming[0]
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = incoming[1:4]
        data.qvel[:6] = incoming[4:10]
        center = stroke.translation(now)[0]
        back = stroke.translation(now - dt)[0]
        data.qpos[7:10] = center
        data.qpos[10:14] = stroke.orientation
        data.qvel[6:9] = (center - back) / dt
        mujoco.mj_forward(model, data)
        touched = False
        end = stroke.start + stroke.eta + .055
        while now <= end:
            paddle, velocity = data.qpos[7:14], data.qvel[6:12]
            _drive(data.xfrc_applied[2], paddle, velocity, now, dt, stroke, None, None)
            spin = _world_spin(data.qvel[3:6], data.qpos[3:7])
            _air(data.xfrc_applied[1], data.qvel[:3], spin)
            mujoco.mj_step(model, data)
            now += dt
            touching = any(c.efc_address >= 0 for c in data.contact)
            if touched and not touching:
                if data.qvel[1] > -.5:
                    return None
                spin = _world_spin(data.qvel[3:6], data.qpos[3:7])
                return data.qpos[:3].copy(), data.qvel[:3].copy(), spin
            touched |= touching
        return None


def _world_spin(local, quaternion):
    spin = np.empty(3)
    mujoco.mju_rotVecQuat(spin, local, quaternion)
    return spin


def _air(force, velocity, spin):
    vx, vy, vz = velocity
    wx, wy, wz = spin
    drag = -BALL_MASS * DRAG * math.sqrt(vx*vx + vy*vy + vz*vz)
    lift = BALL_MASS * LIFT
    force[:3] = (drag*vx + lift*(wy*vz-wz*vy), drag*vy + lift*(wz*vx-wx*vz), drag*vz + lift*(wx*vy-wy*vx))
    force[3:] = (-BALL_INERTIA * SPIN_DECAY) * spin


def _flight_approx(positions, velocities, spins, target):
    """Rank candidates quickly before verifying their complete contacts in MuJoCo."""
    p, v, w = positions.copy(), velocities.copy(), spins.copy()
    depth_min = min(-1.30, target[1] - .08)
    dt = .010
    n = len(p)
    bounces = np.zeros(n, dtype=int)
    landing = np.full((n, 2), np.nan)
    crossing = np.full(n, np.nan)
    best = np.full(n, np.inf)
    for _ in range(105):
        dv = -DRAG * np.linalg.norm(v, axis=1, keepdims=True) * v + LIFT * np.cross(w, v)
        dv[:, 2] -= 9.81
        new_p = p + dt * (v + .5 * dt * dv)
        v += dt * dv
        through = (p[:, 1] >= 0.) & (new_p[:, 1] < 0.)
        u = p[through, 1] / (p[through, 1] - new_p[through, 1])
        crossing[through] = p[through, 2] + u * (new_p[through, 2] - p[through, 2])
        first = (p[:, 2] >= TABLE_TOP) & (new_p[:, 2] < TABLE_TOP)
        first &= (np.abs(new_p[:, 0]) < .765) & (np.abs(new_p[:, 1]) < 1.37)
        bounces += first
        first &= bounces == 1
        landing[first] = new_p[first, :2]
        slip = v[first, :2] - BALL_RADIUS * np.cross(w[first], [0., 0., 1.])[:, :2]
        size = np.linalg.norm(slip, axis=1)
        impulse = np.minimum(.24, .18 * 1.86 * -v[first, 2] / np.maximum(size, .01))
        v[first, :2] -= impulse[:, None] * slip
        w[first] += 1 / (.4 * BALL_RADIUS) * np.cross([0., 0., -1.], np.c_[-impulse[:, None] * slip, np.zeros(len(size))])
        new_p[first, 2] = 2 * TABLE_TOP - new_p[first, 2]
        v[first, 2] *= -.86
        p = new_p
        w *= 1. - .18 * dt
        usable = (bounces == 1) & (p[:, 1] < -.65) & (p[:, 1] > depth_min)
        usable &= (p[:, 2] > TABLE_TOP + .045) & (p[:, 2] < TABLE_TOP + .42)
        cost = np.abs(p - target[:3]) @ [7., 9., 8.] + .60 * np.abs(v[:, 1] + target[3])
        cost += .002 * np.maximum(0., 40. - w[:, 0])
        best[usable] = np.minimum(best[usable], cost[usable])
    legal = (crossing > NET_TOP + .012) & (landing[:, 1] < -.14)
    best[~legal] = np.inf
    best += np.where(legal, 4. * np.maximum(0., NET_TOP + .055 - crossing), 0.)
    return best


@dataclass
class Stroke:
    start: float
    eta: float
    coefficients: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    target: np.ndarray

    def translation(self, now):
        elapsed = now - self.start
        if elapsed >= self.eta:
            offset = elapsed - self.eta
            follow = min(offset, .028)
            ramp = min(1., (offset - follow) / .11)
            advance = follow + .11 * (ramp - .5 * ramp**2)
            return self.center + advance * self.velocity, (1. - ramp) * self.velocity, (
                -self.velocity / .11 if ramp < 1. else np.zeros(3))
        u = elapsed / self.eta
        position = np.array([1, u, u*u, u**3, u**4, u**5]) @ self.coefficients
        velocity = np.array([0, 1, 2*u, 3*u*u, 4*u**3, 5*u**4]) @ self.coefficients / self.eta
        acceleration = np.array([0, 0, 2, 6*u, 12*u*u, 20*u**3]) @ self.coefficients / self.eta**2
        return position, velocity, acceleration


def _trajectory(start, eta, position, velocity, center, end_velocity, quat, target):
    v0, v1 = velocity * eta, end_velocity * eta
    displacement = center - position
    coefficients = np.stack((position, v0, np.zeros(3), 10*displacement - 6*v0 - 4*v1,
                             -15*displacement + 8*v0 + 7*v1, 6*displacement - 3*v0 - 3*v1))
    stroke = Stroke(start, eta, coefficients, center, end_velocity, quat, target)
    u = np.linspace(0, 1, 15)
    vel = np.column_stack((0*u, 0*u+1, 2*u, 3*u*u, 4*u**3, 5*u**4)) @ coefficients / eta
    accel = np.column_stack((0*u, 0*u, 0*u+2, 6*u, 12*u*u, 20*u**3)) @ coefficients / eta**2
    if (np.linalg.norm(vel, axis=1).max() <= .94*LINEAR_SPEED
            and np.linalg.norm(accel, axis=1).max() <= .88*LINEAR_ACCELERATION):
        return stroke
    return None


def _clears_table(stroke, depth, half_width):
    u = np.linspace(0, 1, 31)
    approach = np.column_stack((1 + 0*u, u, u**2, u**3, u**4, u**5)) @ stroke.coefficients
    # Reserve clearance for the rotating rim and the full follow-through.
    release = stroke.center + np.linspace(0., .083, 10)[:, None] * stroke.velocity
    p = np.concatenate((approach, release))
    above_surface = ((np.abs(p[:, 0]) < half_width + PADDLE_RADIUS)
                     & (np.abs(p[:, 1]) < depth + PADDLE_RADIUS))
    return not np.any(p[above_surface, 2] < TABLE_HEIGHT + PADDLE_RADIUS + .009)


def _drive(force, q, v, now, dt, stroke, home, home_quat):
    if stroke is None:
        quat = home_quat
        target_vel = _cap(7.*(home-q[:3]), .80*LINEAR_SPEED)
        linear = 48.*(target_vel-v[:3])
    else:
        target_pos, target_vel, _ = stroke.translation(now)
        next_pos, _, _ = stroke.translation(now + dt)
        if now > stroke.start:
            previous_pos, _, _ = stroke.translation(now - dt)
            target_vel = (target_pos - previous_pos) / dt
        acceleration = ((next_pos - target_pos) / dt - target_vel) / dt
        quat = stroke.orientation
        linear = acceleration + 1600.*(target_pos-q[:3]) + 76.*(target_vel-v[:3])
    linear = _cap(linear, LINEAR_ACCELERATION)
    linear = (_cap(v[:3] + linear*dt, LINEAR_SPEED)-v[:3]) / dt
    linear = _cap(linear, LINEAR_ACCELERATION)
    local_rotation = np.empty(3)
    mujoco.mju_subQuat(local_rotation, quat, q[3:7])
    angular = _cap(550.*local_rotation-43.*v[3:], ANGULAR_ACCELERATION)
    angular = (_cap(v[3:]+angular*dt, ANGULAR_SPEED)-v[3:]) / dt
    angular = _cap(angular, ANGULAR_ACCELERATION)
    force[:3] = MASS * linear
    force[3:] = INERTIA * _world_spin(angular, q[3:7])


class PracticePartner:
    """Read privileged state only here and drive a finite-mass six-DOF blade."""

    def __init__(self, arena, *, targets=LANES, guard_table=False, verify_approach=False):
        self.arena = arena
        self.guard_table = guard_table
        self.verify_approach = verify_approach
        self.targets = np.array(targets, dtype=float).reshape(-1, 4)
        if (not len(self.targets) or not np.isfinite(self.targets).all()
                or np.any(self.targets[:, 1] >= 0) or np.any(self.targets[:, 3] <= .60)):
            raise ValueError("provide finite x, negative y, height and incoming pace above 0.60 m/s")
        self.flight = BallFlight(arena)
        self.impact = IsolatedImpact(arena)
        joint = arena.model.joint(CARRIAGE)
        self.qpos, self.qvel = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.body = arena.model.body(CARRIAGE).id
        self.home = arena.model.body(CARRIAGE).pos.copy()
        self.home_quat = arena.model.body(CARRIAGE).quat.copy()
        self.reset(seed=arena.seed, point=arena._point)

    def reset(self, *, seed, point=0):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, max(0, point), 29037]))
        self.stroke = None
        self.reception = None
        self.plan_stage = 0
        self.reports = []
        self.max_speed = 0.
        self.max_angular_speed = 0.
        self.max_force = 0.

    def _plan(self, target, position, velocity, now):
        a = self.arena
        path = self.flight.path(a.ball, a.velocity, a.spin, -1, bounced=a._received_bounce, horizon=.95)
        feasible = (path[:, 0] >= .12) & (path[:, 2] > .51) & (path[:, 2] < a.table_half_length + .35)
        feasible &= (np.abs(path[:, 1]) < .81) & (path[:, 3] > .86) & (path[:, 3] < 1.58)
        options = path[feasible]
        if not len(options):
            return None, dict(windows=0, verified=0)
        chosen = np.unique(np.linspace(0, len(options)-1, min(7, len(options))).round().astype(int))
        settings = np.array(list(product((-.60, .20, .95), (-.055, .018, .092), (-.060, 0., .060),
                                         (-.44, .21, .68), (.0, .9, 1.6), (.0, .21))))
        nsettings = len(settings)
        rows = np.repeat(options[chosen], nsettings, axis=0)
        p, v = rows[:, 1:4], rows[:, 4:7]
        pace_diff, loft_diff, bearing_diff, drive_diff, rise, short = np.tile(settings, (len(chosen), 1)).T
        length = p[:, 1] - target[1]
        pace = (target[3] + pace_diff) * np.exp(DRAG * length) / .93
        landing_y = target[1] + .51 + short + .54 * (target[2] - 1.01)
        dist = p[:, 1] - landing_y
        t = np.expm1(DRAG * dist) / (DRAG * pace)
        drag_rate = DRAG * pace
        dr = -np.expm1(-drag_rate*t)
        launch_z = (TABLE_TOP - p[:, 2] + 9.81 * t / drag_rate) * drag_rate / dr - 9.81 / drag_rate
        launch = np.c_[(target[0]-p[:, 0])*pace/length, -pace, launch_z]
        delta = launch - v
        normal = delta / np.linalg.norm(delta, axis=1, keepdims=True)
        yaw = np.arctan2(normal[:, 0], -normal[:, 1]) + bearing_diff
        pitch = np.arcsin(normal[:, 2]) + loft_diff
        normal = np.c_[np.sin(yaw)*np.cos(pitch), -np.cos(yaw)*np.cos(pitch), np.sin(pitch)]
        speed = np.clip((v*normal).sum(axis=1) + np.linalg.norm(delta, axis=1)/1.82 + drive_diff, -1.85, 2.76)
        tangent = np.c_[-np.sin(yaw)*np.sin(pitch), np.cos(yaw)*np.sin(pitch), np.cos(pitch)]
        drive = speed[:, None] * normal + rise[:, None] * tangent
        out_p, out_v, out_w, centers, quats = self.impact.solve(rows, normal, drive)
        scores = _flight_approx(out_p, out_v, out_w, target)
        best, best_cost, verified = None, np.inf, 0
        for i in np.argsort(scores):
            if not np.isfinite(scores[i]) or verified >= 18:
                break
            stroke = _trajectory(now, rows[i, 0], position, velocity, centers[i], drive[i], quats[i], target)
            if stroke is None or (self.guard_table and not _clears_table(
                stroke, a.table_half_length, a.table_half_width
            )):
                continue
            verified += 1
            departure = out_p[i], out_v[i], out_w[i]
            if self.verify_approach:
                start = np.searchsorted(path[:, 0], stroke.eta - .035)
                departure = self.impact.verify_approach(path[start], stroke)
                if departure is None:
                    continue
            solved = self._outgoing(*departure, min(-1.34, target[1] - .08))
            if not len(solved):
                continue
            practical = ((solved[:, 2] < -.68) & (solved[:, 2] > min(-1.32, target[1] - .06))
                         & (solved[:, 3] > TABLE_TOP + .045))
            if not practical.any():
                continue
            region = solved[practical]
            loss = np.abs(region[:, 1:4] - target[:3]) @ [7., 9., 8.] + .60*np.abs(region[:, 5] + target[3])
            index = np.argmin(loss)
            cost = loss[index] + .32 * scores[i] + .025*np.linalg.norm(drive[i]) + .002*max(0., 40. - departure[2][0])
            if cost < best_cost:
                best, best_cost = stroke, cost
                selected = dict(eta_s=float(stroke.eta), strike_xyz=rows[i, 1:4].tolist(),
                                blade_velocity=stroke.velocity.tolist(), expected_receive=region[index, 1:7].tolist(),
                                predicted_departure=departure[1].tolist(), predicted_departure_spin=departure[2].tolist())
        summary = dict(windows=int(len(options)), verified=verified, target=target.tolist())
        if best is not None:
            summary.update(selected)
        return best, summary

    def _outgoing(self, position, velocity, spin, min_y):
        """Measure the near-side receiver window with the isolated native surfaces."""
        model, d = self.flight.model, self.flight.data
        table = self.flight.surface
        net = model.geom("net").id
        mujoco.mj_resetData(model, d)
        d.qpos[:3] = position
        d.qvel[:3] = velocity
        d.qvel[3:] = spin
        mujoco.mj_forward(model, d)
        touching = bounced = False
        rows = []
        w = np.empty(3)
        for _ in range(round(.9 / self.flight.dt)):
            mujoco.mju_rotVecQuat(w, d.qvel[3:], d.qpos[3:])
            _air(d.xfrc_applied[1], d.qvel[:3], w)
            mujoco.mj_step(model, d)
            contact = False
            for c in d.contact:
                if c.efc_address < 0:
                    continue
                if net in c.geom:
                    return np.empty((0, 7))
                contact |= (c.geom[0] in table or c.geom[1] in table) and abs(c.frame[2]) > .55
            x, y, z = d.qpos[:3]
            if contact and not touching:
                if bounced:
                    break
                if y >= -.12:
                    return np.empty((0, 7))
                bounced = True
            touching = contact
            if y < min_y or d.qvel[1] > -.25 or z < TABLE_HEIGHT - .04:
                break
            if bounced and not contact and y < -.65 and z > TABLE_TOP + .035:
                rows.append((d.time, x, y, z, *d.qvel[:3]))
        return np.array(rows).reshape(-1, 7)

    def advance(self):
        a = self.arena
        data = a.data
        q = data.qpos[self.qpos:self.qpos+7]
        v = data.qvel[self.qvel:self.qvel+6]
        now = float(data.time)
        if a._last_hitter == 1 and a.velocity[1] > .32:
            if self.reception != a.contacts:
                self.reception = a.contacts
                self.target = self.targets[self.rng.integers(len(self.targets))].copy()
                self.plan_stage = 0
                self.stroke = None
            near_touch = a._racket_ids[1] in a._previous_touches
            far_touch = a._geom_ids['table'] in a._previous_touches
            stage = 1 + int(a.ball[1] > -.20) + int(a.ball[1] > .40)
            if not near_touch and not far_touch and self.stroke is None and stage > self.plan_stage:
                self.plan_stage = stage
                self.stroke, record = self._plan(self.target, q[:3], v[:3], now)
                self.reports.append(dict(time_s=now, stage=stage, **record))
        elif self.stroke is not None and now > self.stroke.start + self.stroke.eta + .15:
            self.stroke = None

        _drive(data.xfrc_applied[self.body], q, v, now, self.impact.dt,
               self.stroke, self.home, self.home_quat)
        self.max_speed = max(self.max_speed, float(np.linalg.norm(v[:3])))
        self.max_angular_speed = max(self.max_angular_speed, float(np.linalg.norm(v[3:])))
        self.max_force = max(self.max_force, float(np.linalg.norm(data.xfrc_applied[self.body, :3])))


class PartnerArena(RallyArena):
    """Native near arm and referee with an independent far carriage."""

    def __init__(self, *, targets=LANES, guard_table=False, verify_approach=False, **kwargs):
        self.partner = None
        super().__init__(**kwargs)
        self.partner = PracticePartner(self, targets=targets, guard_table=guard_table, verify_approach=verify_approach)

    def _model_document(self):
        return add_carriage(super()._model_document())

    def _aerodynamics(self):
        if self.partner is not None:
            self.partner.advance()
        super()._aerodynamics()

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if self.partner is not None:
            self.partner.reset(seed=self.seed, point=self._point)
        return result

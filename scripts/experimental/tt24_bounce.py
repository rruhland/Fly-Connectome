"""Wrist-camera paddle bouncing on the table-end-clamped SO-101."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import AIR_DRAG, BALL_MASS, BALL_RADIUS, MODEL_PATH, mujoco
from .tt18_plant import _resize_table
from .tt19_so101 import (AXES, BLADE, CAMERA_BODY, GRIP_BODY, MINI_TABLE, RATES,
                        SOURCE_MODEL, WRIST_CAMERA, _GRASP_POS, _paddle, _robot)
from .tt20_fixture import (ARCHIVED_FIXTURE, BALL_ORANGE, BLADE_RED, CLAMP_FORWARD, CLAMP_RISE,
                           EDGE_BASE_HEIGHT, EDGE_BASE_INSET, TABLE_EDGE, WRIST_APERTURE,
                           _board_camera, _clamp, _edge_clamp, _outboard, _vector, default_fixture)


CONTROL_INTERVAL = 1 / 60
SUBSTEPS = 8
MIN_RISE = .045
MIN_AIRBORNE = .065
MIN_IMPACT_SPEED = .25
MIN_LAUNCH_SPEED = .25
MAX_CONTACT_TIME = .032
_HOME = np.array((.01689, -.49254, -.48437, .97690, -.73672, -.10))
_ARCHIVED_BASE_DEPTH = MINI_TABLE.table_half_length + .375 - CLAMP_FORWARD
_ARCHIVED_BASE_HEIGHT = .49 + CLAMP_RISE
_EXTERNAL_CAMERA = 'tt24_external'
_CHANGED_DRIVE = np.array((.81, .55, .68, .62, .75, 1.))
_CHANGED_FORCE = np.array((.91, .75, .83, .80, .85, 1.))
_ALTERNATE_APERTURE = (-.018, .218, -.234)
_UPPER_ARM_STRETCH = 1.15


def _camera_axes(eye, target):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def _stretch_x(element):
    position = np.fromstring(element.get('pos'), sep=' ')
    position[0] *= _UPPER_ARM_STRETCH
    element.set('pos', _vector(position))


def _alternate_arm(document, arm, camera):
    upper = arm.find(".//body[@name='learner_upper_arm']")
    elbow = upper.find("body[@name='learner_elbow']")
    _stretch_x(elbow)
    for geom in upper.findall('geom'):
        _stretch_x(geom)
    inertial = upper.find('inertial')
    _stretch_x(inertial)
    inertia = np.fromstring(inertial.get('fullinertia'), sep=' ')
    inertia[1:3] += (_UPPER_ARM_STRETCH**2 - 1.) * (inertia[1] + inertia[2] - inertia[0]) / 2
    inertia[3:5] *= _UPPER_ARM_STRETCH
    inertial.set('fullinertia', _vector(inertia))
    mesh = next(mesh for mesh in document.findall('asset/mesh')
                if Path(mesh.get('file')).stem == 'upper_arm_so101_v1')
    mesh.set('scale', _vector((_UPPER_ARM_STRETCH, 1., 1.)))
    elbow.find('joint').set('axis', _vector((.10, -.08, math.sqrt(1. - .10**2 - .08**2))))
    payload = ET.SubElement(camera, 'body', name='tt24_wrist_payload', pos='0 0 .037')
    ET.SubElement(payload, 'geom', name='tt24_payload_housing', type='box', size='.018 .017 .012',
                  mass='.042', rgba='.18 .21 .24 1', contype='0', conaffinity='1')


def _model_document(width, height, detail, arm_design, fixture=TABLE_EDGE):
    document = ET.parse(MODEL_PATH).getroot()
    document.set('model', 'TT24 end-clamped SO-101 paddle bouncing')
    world = document.find('worldbody')
    for tag, name in (('body', 'rival_base'), ('body', 'decoy'), ('geom', 'net'),
                      ('camera', 'eye_left'), ('camera', 'eye_right')):
        world.remove(world.find(f"{tag}[@name='{name}']"))
    for geom in world.findall('geom'):
        if float(geom.get('pos', '0 0 0').split()[2]) > .80:
            world.remove(geom)
    for section in ('tendon', 'equality', 'actuator'):
        parent = document.find(section)
        for item in list(parent):
            reference = item.get('name', item.get('tendon1', ''))
            if reference.startswith('rival_'):
                parent.remove(item)

    _resize_table(world, MINI_TABLE)
    edge = fixture == TABLE_EDGE
    base_depth = MINI_TABLE.table_half_length - EDGE_BASE_INSET if edge else _ARCHIVED_BASE_DEPTH
    base_height = EDGE_BASE_HEIGHT if edge else _ARCHIVED_BASE_HEIGHT
    _robot(document, base_depth, base_height, detail)
    _paddle(document)
    if edge:
        _edge_clamp(document, MINI_TABLE.table_half_length)
        _board_camera(document, alternate=arm_design == 'alternate')
    else:
        _clamp(document, base_depth, base_height, MINI_TABLE.table_half_length, MINI_TABLE.table_half_width)
        aperture = _ALTERNATE_APERTURE if arm_design == 'alternate' else WRIST_APERTURE
        _outboard(document, 108., aperture)
        boom = world.find(".//geom[@name='tt20_camera_boom']")
        # Put the archived support behind the reversed lens on the solo arm.
        boom_tip = np.add(aperture, (-.008, .022, .011))
        boom.set('fromto', _vector((-.014, .036, -.040, *boom_tip)))

    # Recover the upstream arm collision meshes for ball-only collision. The
    # clamp and tabletop still collide with the blade through the TT19 masks.
    arm = world.find("body[@name='learner_base']")
    for source in ET.parse(SOURCE_MODEL).getroot().find('worldbody/body').iter('body'):
        name = source.get('name')
        if name == 'wrist_camera' or (edge and name == 'wrist_camera_mount'):
            continue
        name = 'learner_elbow' if name == 'lower_arm' else f'learner_{name}'
        target = arm if name == 'learner_base' else arm.find(f".//body[@name='{name}']")
        for index, collision in enumerate(source.findall("geom[@class='collision']")):
            geom = deepcopy(collision)
            geom.attrib.update(name=f'{name}_ball_contact_{index}', contype='0', conaffinity='1')
            target.append(geom)
    camera = arm.find(f".//body[@name='{CAMERA_BODY}']")
    for geom in camera.findall('geom'):
        if geom.get('name') in ('tt20_rgb_housing', 'tt20_rgb_barrel'):
            geom.attrib.update(contype='0', conaffinity='1')
    if arm_design == 'alternate':
        _alternate_arm(document, arm, camera)

    ball = world.find("body[@name='ball']/geom[@name='ball']")
    ball.set('rgba', _vector((*BALL_ORANGE, 1.)))
    blade = world.find(f".//geom[@name='{BLADE}']")
    blade.set('rgba', _vector((*BLADE_RED, 1.)))
    handle = world.find(".//geom[@name='learner_handle']")
    handle.attrib.update(contype='0', conaffinity='1')
    visual = document.find('visual')
    visual.find('global').attrib.update(offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'quality', offsamples='0', shadowsize='0')
    visual.find('map').set('zfar', '4')
    eye = np.array((.93, -1.42, 1.48) if edge else (.88, -1.53, 1.24))
    look = np.array((0., -.65, 1.04) if edge else (0., -.93, .86))
    axes = _camera_axes(eye, look)
    ET.SubElement(world, 'camera', name=_EXTERNAL_CAMERA, pos=_vector(eye), fovy='40',
                  xyaxes=_vector(axes[:, :2].T.ravel()))
    document.find('option').set('timestep', str(CONTROL_INTERVAL / SUBSTEPS))
    return document


@dataclass(frozen=True)
class _Drop:
    pose: tuple[float, ...]
    offset: tuple[float, ...]
    velocity: tuple[float, ...]
    spin: tuple[float, ...]
    drag: float
    damping: float


def _sample_drop(seed, split, regime):
    rng = np.random.default_rng(seed)
    if split == 'practice':
        pan, shoulder, elbow, roll = .08, .045, .065, .010
        xy, lateral, spin = .025, .045, 5.
        lower, upper, drag, damping = .19, .27, (.90, 1.10), (26., 35.)
    elif split == 'held':
        pan, shoulder, elbow, roll = .13, .085, .100, .018
        xy, lateral, spin = .039, .075, 10.
        lower, upper, drag, damping = .16, .32, (.80, 1.20), (23., 39.)
    else:
        raise ValueError("split must be 'practice' or 'held'")
    pan, shoulder, elbow, roll = rng.uniform((-pan, -shoulder, -elbow, -roll), (pan, shoulder, elbow, roll))
    pose = _HOME + (pan, shoulder, elbow, -shoulder - elbow, roll, 0.)
    offset = (*rng.uniform(-xy, xy, 2), rng.uniform(lower, upper))
    velocity = (*rng.uniform(-lateral, lateral, 2), rng.uniform(-.15, .04))
    spin = tuple(rng.uniform(-spin, spin, 3))
    drag_scale = rng.uniform(*drag)
    contact_damping = rng.uniform(*damping)
    if regime == 'changed':
        drag_scale *= 1.25
        contact_damping += 8.
    return _Drop(tuple(pose), offset, velocity, spin, drag_scale, contact_damping)


@dataclass
class _Rebound:
    impact_index: int
    contact_time: float
    contact_position: tuple[float, ...]
    incoming_vz: float
    launch_time: float = 0.
    launch_vz: float = 0.
    apex_z: float = 0.
    apex_time: float = 0.
    next_contact_time: float | None = None


class BounceArena:
    """Expose RGB, optional own joint angles/rates and a scalar rebound reward."""

    action_axes = AXES[:5]
    action_size = len(action_axes)
    dt = CONTROL_INTERVAL
    _joint_axes = AXES
    _rates = (*RATES[:5], 0.)
    _changed_drive = _CHANGED_DRIVE
    _changed_force = _CHANGED_FORCE

    def _document(self, width, height, detail, arm):
        return _model_document(width, height, detail, arm, self.fixture)

    def _actuated_pose(self, pose):
        return pose

    def __init__(self, *, view='wrist', proprioception=True, width=96, height=72,
                 max_steps=300, detail='preview', arm='stock', apex_reward_weight=0., fixture=None):
        if view not in ('wrist', 'external', 'none'):
            raise ValueError("view must be 'wrist', 'external' or 'none'")
        if min(width, height, max_steps) < 1:
            raise ValueError('image dimensions and max_steps must be positive')
        if detail not in ('cad', 'preview'):
            raise ValueError("detail must be 'cad' or 'preview'")
        if arm not in ('stock', 'alternate'):
            raise ValueError("arm must be 'stock' or 'alternate'")
        if fixture is None:
            fixture = default_fixture()
        if fixture not in (TABLE_EDGE, ARCHIVED_FIXTURE):
            raise ValueError("expected fixture='table-edge' or fixture='archived'")
        if not math.isfinite(apex_reward_weight) or apex_reward_weight < 0.:
            raise ValueError('apex_reward_weight must be finite and nonnegative')
        self.view = view
        self.arm = arm
        self.fixture = fixture
        self.apex_reward_weight = apex_reward_weight
        self.proprioception = proprioception
        self.width, self.height, self.max_steps = width, height, max_steps
        doc = self._document(width, height, detail, arm)
        self._model = mujoco.MjModel.from_xml_string(ET.tostring(doc, encoding='unicode'))
        self._data = mujoco.MjData(self._model)
        model = self._model
        joints = [model.joint(f'learner_{axis}') for axis in self._joint_axes]
        self._joint_position = np.array([joint.qposadr[0] for joint in joints])
        self._joint_velocity = np.array([joint.dofadr[0] for joint in joints])
        self._control = self._data.ctrl
        self._control_low, self._control_high = model.actuator_ctrlrange.T
        self._standard_gain = model.actuator_gainprm[:, 0].copy()
        self._standard_bias = model.actuator_biasprm[:, 1:3].copy()
        self._standard_force = model.actuator_forcerange.copy()
        self._increments = np.array(self._rates) * model.opt.timestep
        self._command = np.zeros(len(self._joint_axes))
        self._blade = model.geom(BLADE).id
        self._ball_geom = model.geom('ball').id
        self._ball_body = model.body('ball').id
        ball = model.joint('ball_free')
        self._ball_position = self._data.qpos[ball.qposadr[0]:ball.qposadr[0] + 3]
        self._ball_velocity = self._data.qvel[ball.dofadr[0]:ball.dofadr[0] + 3]
        self._ball_spin = self._data.qvel[ball.dofadr[0] + 3:ball.dofadr[0] + 6]
        self._ball_force = self._data.xfrc_applied[self._ball_body, :3]
        self._paddle = model.joint('learner_paddle_free').qposadr[0]
        self._grip = model.body(GRIP_BODY).id
        base = model.body('learner_base').pos
        self._workspace_shift = (float(base[1] + _ARCHIVED_BASE_DEPTH), float(base[2] - _ARCHIVED_BASE_HEIGHT))
        self._renderer = None
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[3] = 0
        self._prepare_pose(_HOME)
        if fixture == ARCHIVED_FIXTURE:
            blade_position = self._data.geom_xpos[self._blade]
            camera = model.body(CAMERA_BODY)
            eye = self._data.xpos[camera.id].copy()
            gripper = self._data.xmat[self._grip].reshape(3, 3)
            matrix = gripper.T @ _camera_axes(eye, blade_position + (0., 0., .13))
            mujoco.mju_mat2Quat(camera.quat, np.ascontiguousarray(matrix).ravel())
        self._done = True

    def _prepare_pose(self, pose):
        model, data = self._model, self._data
        pose = self._actuated_pose(pose)
        mujoco.mj_resetData(model, data)
        data.qpos[self._joint_position] = pose
        self._control[:] = pose
        mujoco.mj_kinematics(model, data)
        grip_matrix = data.xmat[self._grip].reshape(3, 3)
        data.qpos[self._paddle:self._paddle + 3] = data.xpos[self._grip] + grip_matrix @ _GRASP_POS
        data.qpos[self._paddle + 3:self._paddle + 7] = data.xquat[self._grip]
        self._ball_position[:] = (0., 0., 3.)
        for _ in range(28):
            mujoco.mj_step(model, data)
        data.time = 0.
        mujoco.mj_kinematics(model, data)

    sample_drop = staticmethod(_sample_drop)

    def reset(self, *, seed=None, split='practice', regime='standard', render=True):
        """Reset to a new seeded drop in the practice or held distribution."""
        if regime not in ('standard', 'changed'):
            raise ValueError("regime must be 'standard' or 'changed'")
        self._drop = self.sample_drop(seed, split, regime)
        self._seed, self._split, self._regime = seed, split, regime
        model = self._model
        drive = self._changed_drive if regime == 'changed' else np.ones(len(self._joint_axes))
        force = self._changed_force if regime == 'changed' else np.ones(len(self._joint_axes))
        model.actuator_gainprm[:, 0] = self._standard_gain * drive
        model.actuator_biasprm[:, 1:3] = self._standard_bias * drive[:, None]
        model.actuator_forcerange[:] = self._standard_force * force[:, None]
        model.geom_solref[self._blade] = (-100000., -self._drop.damping)
        self._prepare_pose(self._drop.pose)
        self._drag_force_scale = -BALL_MASS * AIR_DRAG * self._drop.drag
        self._ball_position[:] = self._data.geom_xpos[self._blade] + self._drop.offset
        self._ball_velocity[:] = self._drop.velocity
        self._ball_spin[:] = self._drop.spin
        mujoco.mj_forward(self._model, self._data)
        self._touching = False
        self._contact = None
        self._flight = None
        self._last_airborne = None
        self._rebounds = []
        self._paddle_contacts = 0
        self._top_contacts = 0
        self._steps = 0
        self._done = False
        self._reason = None
        self._collider = None
        self._reward = 0.
        return self._observation(render)

    def step(self, action, *, render=True):
        """Apply normalized rates to the independently actuated position-servo targets."""
        if self._done:
            raise RuntimeError('reset before stepping a finished drop')
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError(f'expected {self.action_size} finite joint commands')
        np.clip(action, -1., 1., out=self._command[:self.action_size])
        increments = self._command * self._increments
        reward = 0.
        for _ in range(SUBSTEPS):
            self._control += increments
            np.clip(self._control, self._control_low, self._control_high, out=self._control)
            incoming = float(self._ball_velocity[2])
            velocity = self._ball_velocity
            speed = math.sqrt(float(velocity @ velocity))
            np.multiply(velocity, self._drag_force_scale * speed, out=self._ball_force)
            mujoco.mj_step(self._model, self._data)
            reward += self._judge_substep(incoming)
            if self._done:
                break
        self._steps += 1
        if self._steps >= self.max_steps and not self._done:
            self._end('timeout')
        self._reward += reward
        return self._observation(render), reward, self._done

    def _judge_substep(self, incoming):
        data = self._data
        t = data.time
        blade_contact = None
        for contact in data.contact:
            if contact.efc_address < 0:
                continue
            first, second = contact.geom
            if first == self._ball_geom:
                other, sign = second, -1
            elif second == self._ball_geom:
                other, sign = first, 1
            else:
                continue
            if other == self._blade:
                blade_contact = (contact, sign)
            else:
                self._record_landing(t)
                self._collider = self._model.geom(other).name
                self._end('non_paddle_contact')
                return 0.

        x, y, z = self._ball_position
        vz = self._ball_velocity[2]
        if blade_contact is not None:
            self._record_landing(t)
            self._flight = None
            if not self._touching:
                self._paddle_contacts += 1
                contact, sign = blade_contact
                normal = data.geom_xmat[self._blade].reshape(3, 3)[:, 2]
                normal_z = sign * contact.frame[2]
                # The stock paddle has two flat faces. Judge whichever face is
                # up, excluding its rim and upward crossings from below.
                face_alignment = abs(float(contact.frame[:3] @ normal))
                if normal_z > .55 and face_alignment > .95:
                    self._top_contacts += 1
                    if incoming < -MIN_IMPACT_SPEED:
                        self._contact = _Rebound(self._paddle_contacts, float(t), (float(x), float(y), float(z)), incoming)
            self._touching = True
        elif self._touching:
            self._touching = False
            flight, self._contact = self._contact, None
            if flight is not None and t - flight.contact_time <= MAX_CONTACT_TIME and vz > MIN_LAUNCH_SPEED:
                flight.launch_time = float(t)
                flight.launch_vz = float(vz)
                flight.apex_z = float(z)
                flight.apex_time = float(t)
                self._flight = flight

        shift_y, shift_z = self._workspace_shift
        local_y, local_z = y - shift_y, z - shift_z
        if local_z < .51 or local_z > 2.1 or abs(x) > .66 or not -1.65 < local_y < -.33:
            self._end('out_of_bounce_area')
            return 0.
        flight = self._flight
        if flight is None:
            return 0.
        if z > flight.apex_z:
            flight.apex_z = float(z)
            flight.apex_time = float(t)
        if vz <= 0. and t - flight.launch_time >= MIN_AIRBORNE:
            self._flight = None
            rise = flight.apex_z - flight.contact_position[2]
            if rise >= MIN_RISE:
                self._rebounds.append(flight)
                self._last_airborne = flight
                return 1. + self.apex_reward_weight * min(rise / .12, 1.)
        return 0.

    def _record_landing(self, time):
        if self._last_airborne is not None:
            self._last_airborne.next_contact_time = float(time)
            self._last_airborne = None

    def _end(self, reason):
        self._reason = reason
        self._done = True

    def _observation(self, render):
        result = {}
        if self.proprioception:
            result['joints'] = self._data.qpos[self._joint_position[:self.action_size]].astype(np.float32)
            result['joint_velocity'] = self._data.qvel[self._joint_velocity[:self.action_size]].astype(np.float32)
        if render and self.view != 'none':
            result['rgb'] = self._render(self.view)
        return result

    def _render(self, view):
        camera = WRIST_CAMERA if view == 'wrist' else _EXTERNAL_CAMERA
        return self._render_camera(camera, self._scene_option)

    def _render_camera(self, camera, options):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, width=self.width, height=self.height)
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_camlight(self._model, self._data)
        self._renderer.update_scene(self._data, camera=camera, scene_option=options)
        return self._renderer.render()

    def render_wrist(self):
        """Read native RGB from the rigid camera carried by the arm."""
        return self._render('wrist')

    def render_external(self):
        """Read native RGB from the optional spectator camera."""
        return self._render('external')

    def judge_metrics(self):
        """Read hidden physical outcome evidence for evaluation after a drop."""
        events = []
        previous_impact = longest_run = run = 0
        for flight in self._rebounds:
            airborne_duration = flight.next_contact_time - flight.launch_time if flight.next_contact_time else None
            events.append(dict(asdict(flight), rise_m=flight.apex_z - flight.contact_position[2],
                               time_to_apex_s=flight.apex_time - flight.launch_time,
                               airborne_s=airborne_duration))
            run = run + 1 if flight.impact_index == previous_impact + 1 else 1
            longest_run = max(longest_run, run)
            previous_impact = flight.impact_index
        return dict(seed=self._seed, split=self._split, regime=self._regime, arm=self.arm,
                    steps=self._steps, elapsed_s=float(self._data.time),
                    done=self._done, termination=self._reason, other_collider=self._collider,
                    paddle_contacts=self._paddle_contacts, top_face_contacts=self._top_contacts,
                    rebounds=len(events), longest_consecutive_rebounds=longest_run,
                    reward=self._reward, apex_reward_weight=self.apex_reward_weight, rebound_events=events,
                    drop=asdict(self._drop))

    def close(self):
        """Release the graphics context when the caller is done collecting frames."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

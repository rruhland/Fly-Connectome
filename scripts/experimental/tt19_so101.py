"""Stock-scale SO-101 follower in a compact, scored MuJoCo table-tennis fixture."""

from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import (AIR_DRAG, BALL_RADIUS, BASE_DEPTH, GRAVITY, HOME, MODEL_PATH,
                        TABLE_BOUNCE, TABLE_FORWARD, TABLE_HALF_LENGTH, TABLE_HEIGHT, _arm_angles,
                        _flight_to_depth, _racket_loft, _vertical, mujoco)
from .tt4_camera import PerspectiveCamera
from .tt18_plant import PlantSettings, TT18PlantArena, _resize_table, _vector


ASSET_DIR = Path(__file__).with_name('tt19_so101_assets')
SOURCE_MODEL = ASSET_DIR / 'so101_new_calib_camera.xml'
SOURCE_REVISION = '141aea7bcac2744d92998e30268c413202a56f96'
AXES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
RATES = (5.2, 5.2, 5.2, 5.2, 5.2, 4.0)
_BLADE_TWIST = math.pi / 4
SO101_HOME = (0., .55, -.77, -1.42, -_BLADE_TWIST, -.1)
MINI_TABLE = PlantSettings(table_half_width=.424, table_half_length=.80)
WRIST_CAMERA = 'tt19_wrist'
TIP = 'learner_tip'
BLADE = 'learner_racket'
PADDLE_BODY = 'learner_paddle'
GRIP_BODY = 'learner_gripper'
CAMERA_BODY = 'learner_wrist_camera'
_GRASP_POS = np.array((-.0079, -.000218121, -.085))
_BLADE_DEPTH = .092
_BASE_QUAT = '.7071067811865476 0 0 .7071067811865476'
_RELEASE_ANGLE = .44


def _robot(document, base_depth, base_height, detail):
    upstream = ET.parse(SOURCE_MODEL).getroot()
    world = document.find('worldbody')
    old = world.find("body[@name='learner_base']")
    follower = deepcopy(upstream.find('worldbody/body'))
    world.insert(list(world).index(old), follower)
    world.remove(old)
    follower.set('pos', _vector((0., -base_depth, base_height)))
    follower.set('quat', _BASE_QUAT)
    for body in follower.iter('body'):
        name = body.get('name')
        body.set('name', 'learner_elbow' if name == 'lower_arm' else f'learner_{name}')
        for collision in body.findall("geom[@class='collision']"):
            body.remove(collision)
    for joint in follower.iter('joint'):
        joint.set('name', f"learner_{joint.get('name')}")
    gripper = follower.find(f".//body[@name='{GRIP_BODY}']")
    tip = gripper.find("site[@name='gripperframe']")
    tip.set('name', TIP)
    tip.set('rgba', '1 1 1 0')
    camera = gripper.find(f".//body[@name='{CAMERA_BODY}']")
    ET.SubElement(camera, 'camera', name=WRIST_CAMERA, pos='0 0 .021', fovy='77')
    for node in upstream.findall('default/default'):
        document.find('default').append(deepcopy(node))
    for asset in upstream.findall('asset/*'):
        asset = deepcopy(asset)
        if asset.tag == 'mesh':
            directory = 'assets_preview' if detail == 'preview' else 'assets'
            asset.set('file', str(ASSET_DIR / directory / asset.get('file')))
        document.find('asset').append(asset)
    for tag in ('tendon', 'equality', 'actuator'):
        section = document.find(tag)
        for node in list(section):
            if node.get('name', node.get('tendon1', '')).startswith('learner_'):
                section.remove(node)
    for drive in upstream.findall('actuator/position'):
        drive = deepcopy(drive)
        name = f"learner_{drive.get('name')}"
        drive.set('name', name)
        drive.set('joint', name)
        drive.set('ctrlrange', follower.find(f".//joint[@name='{name}']").get('range'))
        document.find('actuator').append(drive)
    ET.SubElement(world, 'geom', name='tt19_pedestal', type='box', pos=_vector((0., -base_depth, base_height / 2)),
                  size=_vector((.105, .101, base_height / 2)), rgba='.21 .25 .29 1')


def _paddle(document):
    paddle = ET.SubElement(document.find('worldbody'), 'body', name=PADDLE_BODY, pos='0 -1 1')
    ET.SubElement(paddle, 'freejoint', name='learner_paddle_free')
    ET.SubElement(paddle, 'geom', name='learner_handle', type='capsule', fromto='0 0 .022 0 0 -.043',
                  size='.0085', mass='.015', rgba='.61 .48 .32 1', contype='8', conaffinity='0')
    twist, pitch = math.cos(_BLADE_TWIST / 2), math.sin(_BLADE_TWIST / 2)
    quat = np.array((twist, -pitch, twist, pitch)) * math.sqrt(.5)
    ET.SubElement(paddle, 'geom', name=BLADE, **{'class': 'racket_collision'}, type='cylinder',
                  pos=_vector((0., 0., -_BLADE_DEPTH)), size='.076 .006',
                  quat=_vector(quat), mass='.10', rgba='.89 .25 .19 1')
    ET.SubElement(document.find('equality'), 'weld', name='tt19_grasp', body1=GRIP_BODY, body2=PADDLE_BODY,
                  relpose=_vector((*_GRASP_POS, 1., 0., 0., 0.)), torquescale='.075', solref='.0042 1',
                  solimp='.98 .999 .0005 .5 2')


class SO101Arena(TT18PlantArena):
    """Use the six physical follower joints and ordinary TT8 public point interface."""

    def __init__(self, *args, plant=None, strict_endline=False, base_height=.49, wrist_size=None, wrist_near=None, detail='cad', **kwargs):
        self.strict_endline = strict_endline
        settings = MINI_TABLE if plant is None else plant
        if isinstance(settings, dict):
            settings = PlantSettings(**settings)
        if settings.payload_mount is not None or settings.link_mass_scale != 1 or settings.link_mass_scales or settings.travel_scale != 1:
            raise ValueError('SO-101 uses its upstream links, limits and wrist camera assembly')
        if detail not in ('cad', 'preview'):
            raise ValueError('expected full CAD or preview visual detail')
        if wrist_near is not None and (not math.isfinite(wrist_near) or wrist_near <= 0):
            raise ValueError('expected a positive near clip distance in metres')
        self.detail = detail
        self.base_height = base_height
        self.base_depth = settings.table_half_length + .375
        self.rival_base_depth = settings.table_half_length + .635
        self.wrist_size = wrist_size
        self.wrist_near = wrist_near
        super().__init__(*args, plant=settings, **kwargs)
        self._orient_wrist()

    def _model_document(self):
        document = ET.parse(MODEL_PATH).getroot()
        world = document.find('worldbody')
        _resize_table(world, self.plant)
        ball = world.find("body[@name='ball']/geom[@name='ball']")
        ball.set('mass', str(float(ball.get('mass')) * self.plant.ball_mass_scale))
        world.find("body[@name='rival_base']").set('pos', _vector((0., self.rival_base_depth, .96)))
        _robot(document, self.base_depth, self.base_height, self.detail)
        _paddle(document)
        return document

    def _arm_axes(self, side):
        if side == 1:
            return AXES, (), np.array(RATES)
        return super()._arm_axes(side)

    def _arm_home(self, side):
        if side == 1:
            return SO101_HOME, ()
        return super()._arm_home(side)

    def reset(self, *, seed=None, render=True, server=-1):
        super().reset(seed=seed, render=False, server=server)
        gripper = self.model.body(GRIP_BODY).id
        paddle = self.model.joint('learner_paddle_free').qposadr[0]
        position = self.data.xpos[gripper] + self.data.xmat[gripper].reshape(3, 3) @ _GRASP_POS
        self.data.qpos[paddle:paddle + 7] = np.r_[position, self.data.xquat[gripper]]
        self._grasp_id = self.model.equality('tt19_grasp').id
        self._jaw_position = self.model.joint('learner_gripper').qposadr[0]
        self.data.eq_active[self._grasp_id] = 1
        mujoco.mj_forward(self.model, self.data)
        return self.render_cameras() if render else None

    @property
    def grasped(self):
        """Whether the simulated no-slip jaw contact is holding the handle."""
        return bool(self.data.eq_active[self._grasp_id])

    def _aerodynamics(self):
        if self.grasped and self.data.qpos[self._jaw_position] > _RELEASE_ANGLE:
            self.data.eq_active[self._grasp_id] = 0
            self._event('grip_release', side=1)
        super()._aerodynamics()

    def _contacts(self, incoming):
        # Once a shot has passed the receiving end line without a bounce,
        # a later touch by that receiver cannot turn the long shot into a win.
        # A ball that bounced in remains live behind the end line.
        towards_receiver = self._last_hitter
        if (self.strict_endline and not self._waiting_for_strike and not self._received_bounce
                and towards_receiver * self.ball[1] > self.table_half_length + BALL_RADIUS
                and towards_receiver * incoming[1] > 0.):
            self._end(-self._last_hitter, 'long_past_endline')
            return
        super()._contacts(incoming)

    def _opening(self):
        rng = self._rng
        depth = self.table_half_length
        if self.server == 1:
            self.ball[:] = rng.uniform((-.064, -depth - .119, 1.096), (.064, -depth - .080, 1.118))
            self.velocity[:] = rng.uniform((-.05, -.015, 2.77), (.05, .015, 2.94))
            return
        start_y = depth + .060
        first_y = depth * rng.uniform(.58, .72)
        receive_y = -depth * rng.uniform(.69, .89)
        speed = rng.uniform(2.58, 2.88) * math.sqrt(depth / MINI_TABLE.table_half_length)
        start_x = rng.uniform(-.30, .30) * self.table_half_width
        receive_x = .63 * start_x + rng.uniform(-.28, .28) * self.table_half_width
        vx = speed * (receive_x - start_x) / (start_y - receive_y)
        first_time = math.expm1(AIR_DRAG * (start_y - first_y)) / (AIR_DRAG * speed)
        first_speed = speed * math.exp(-AIR_DRAG * speed * first_time) * TABLE_FORWARD
        flight = math.expm1(AIR_DRAG * (first_y - receive_y)) / (AIR_DRAG * first_speed)
        drag = AIR_DRAG * first_speed
        rise = GRAVITY * flight / -math.expm1(-drag * flight) - GRAVITY / drag
        impact = -rise / TABLE_BOUNCE
        before_drag = AIR_DRAG * speed
        vz = (impact + GRAVITY / before_drag) * math.exp(before_drag * first_time) - GRAVITY / before_drag
        displacement, _ = _vertical(0., vz, first_time, before_drag)
        self.ball[:] = (start_x, start_y, TABLE_HEIGHT + BALL_RADIUS - displacement)
        self.velocity[:] = (vx, -speed, vz)
        self.data.qvel[self._ball_dof + 3:self._ball_dof + 6] = rng.uniform((55., -20., -28.), (115., 20., 28.))

    def _choose_rival_response(self):
        super()._choose_rival_response()
        self._rival_target *= self.table_half_length / TABLE_HALF_LENGTH

    def _aim(self, side, bias=0., target_depth=.96, scale=1.):
        if side == 1:
            raise ValueError('the stock TT4 oracle cannot control the SO-101 mechanism')
        incoming = -self.velocity[1]
        if self._last_hitter == -1 or incoming > -.30:
            return self._command_to(-1, HOME, scale)
        depth = self.table_half_length + .045
        x, _, _, _ = _flight_to_depth(self.ball, self.velocity, depth)
        depth += .10 * np.clip((abs(x) - .15) / .20, 0., 1.)
        eta = (depth - self.ball[1]) / -incoming
        if eta < -.075:
            return np.zeros(4)
        if eta >= 0.:
            x, z, vy, vz = _flight_to_depth(self.ball, self.velocity, depth)
        else:
            x, _, z = self.ball + self.velocity * eta
            _, vy, vz = self.velocity
        z = np.clip(z, TABLE_HEIGHT + .071, 1.43)
        speed = 1.68 * scale
        pitch = _racket_loft(-vy, vz, z, depth + target_depth, speed)
        swing = np.clip(speed * (.067 - eta), -.20, .10)
        blade_y = depth + (BALL_RADIUS + .006) * math.cos(pitch) - swing + BASE_DEPTH - self.rival_base_depth
        blade_z = z - (BALL_RADIUS + .006) * math.sin(pitch)
        return self._command_to(-1, _arm_angles(x + bias, blade_y, blade_z, -1, pitch), scale)

    def _orient_wrist(self):
        camera = self.model.camera(WRIST_CAMERA)
        parent = int(camera.bodyid[0])
        rotation = self.data.xmat[parent].reshape(3, 3)
        eye = self.data.xpos[parent] + rotation @ camera.pos
        target = .44 * self.data.site(TIP).xpos + .56 * self.paddle
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.data.xmat[self.model.body(GRIP_BODY).id].reshape(3, 3)[:, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        matrix = rotation.T @ np.column_stack((right, up, -forward))
        mujoco.mju_mat2Quat(camera.quat, np.ascontiguousarray(matrix).reshape(-1))

    def render_wrist(self):
        """Read the native RGB sensor rigidly attached to the follower's camera assembly."""
        if self._mounted_renderer is None:
            width, height = self.wrist_size or (self.width, self.height)
            self._mounted_renderer = mujoco.Renderer(self.model, width=width, height=height)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self._update_wrist_scene()
        return self._mounted_renderer.render()

    def _update_wrist_scene(self):
        if self.wrist_near is None:
            self._mounted_renderer.update_scene(self.data, camera=WRIST_CAMERA)
            return
        mapping = self.model.vis.map
        near = mapping.znear
        try:
            mapping.znear = self.wrist_near / self.model.stat.extent
            self._mounted_renderer.update_scene(self.data, camera=WRIST_CAMERA)
        finally:
            mapping.znear = near

    def render_mounted_camera(self):
        return self.render_wrist()


def make_arena(spec=None, *, arena_type=SO101Arena):
    """Instantiate TT19 with the TT17-style arena and optional stereo calibration fields."""
    spec = {} if spec is None else spec
    arena = arena_type(**spec.get('arena', {}), plant=spec.get('plant'))
    if 'camera' in spec:
        arena._rig = tuple(PerspectiveCamera(**eye) for eye in spec['camera'])
    return arena

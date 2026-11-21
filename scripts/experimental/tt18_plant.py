"""Optional physical TT8/TT17 tables with independently adjustable materials and arms."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import (BALL_MASS, MODEL_PATH, TABLE_HALF_LENGTH, TABLE_HALF_WIDTH,
                        TABLE_HEIGHT, mujoco)
from .tt4_camera import PerspectiveCamera
from .tt4_hall import hall_model
from .tt6_body_arena import body_xml
from .tt8_service_arena import MIN_TOSS_RISE, TT8ServiceArena
from .tt16_transfer_protocol import make_arena as make_tt17_arena


@dataclass(frozen=True)
class PlantSettings:
    ball_mass_scale: float = 1.
    link_mass_scale: float = 1.
    link_mass_scales: Mapping[str, float] = field(default_factory=dict)
    travel_scale: float = 1.
    table_half_width: float = TABLE_HALF_WIDTH
    table_half_length: float = TABLE_HALF_LENGTH
    left_damping: float = 24.
    right_damping: float = 24.
    payload_mount: str | None = None
    payload_mass: float = .05
    camera_fovy: float = 78.

    def __post_init__(self):
        scales = (self.ball_mass_scale, self.link_mass_scale, self.travel_scale, *self.link_mass_scales.values())
        dimensions = (self.table_half_width, self.table_half_length)
        damping = (self.left_damping, self.right_damping)
        if not all(math.isfinite(v) and v > 0 for v in (*scales, *dimensions, *damping)):
            raise ValueError('plant scales, table dimensions and damping must be finite and positive')
        if self.payload_mount not in (None, 'wrist', 'proximal'):
            raise ValueError('expected a wrist or proximal camera mount')
        if not math.isfinite(self.payload_mass) or self.payload_mass <= 0 or not 0 < self.camera_fovy < 180:
            raise ValueError('expected a positive payload mass and valid camera field of view')


PLANTS = {
    'compact': PlantSettings(ball_mass_scale=1.25, link_mass_scale=1.22, travel_scale=.86,
                             table_half_width=.704, table_half_length=1.32, left_damping=43.,
                             right_damping=17., payload_mount='wrist', payload_mass=.045),
    'extended': PlantSettings(ball_mass_scale=1.34, link_mass_scale=.86, travel_scale=.94,
                              table_half_width=.813, table_half_length=1.43, left_damping=17.,
                              right_damping=40., payload_mount='proximal', payload_mass=.065),
}

MOUNTED_CAMERA = 'tt18_mount_view'
PAYLOAD = 'tt18_camera_housing'
_PROXIMAL_BODIES = {'gantry': 'bridge', 'offset': 'carriage', 'rotary': 'shoulder', 'stock': 'shoulder'}


def _vector(values):
    return ' '.join(str(value) for value in values)


def _resize_table(world, settings):
    table = world.find("geom[@name='table']")
    table_size = np.fromstring(table.get('size'), sep=' ')
    width, depth, _ = table_size
    sx, sy = settings.table_half_width / width, settings.table_half_length / depth
    # These world geoms are the tabletop, markings, supports, net and posts.
    for geom in world.findall('geom'):
        if geom.get('name') == 'floor':
            continue
        pos = np.fromstring(geom.get('pos'), sep=' ')
        pos[:2] *= (sx, sy)
        geom.set('pos', _vector(pos))
        if geom.get('type') == 'box':
            size = np.fromstring(geom.get('size'), sep=' ')
            size[:2] *= (sx, sy)
            geom.set('size', _vector(size))
    table.set('solref', f'-100000 {-settings.left_damping}')
    if settings.left_damping == settings.right_damping:
        return

    right = deepcopy(table)
    right.set('name', 'table_right')
    right.set('solref', f'-100000 {-settings.right_damping}')
    world.insert(list(world).index(table) + 1, right)
    panel_size = (settings.table_half_width / 2, settings.table_half_length, table_size[2])
    for side, panel in ((-1, table), (1, right)):
        panel.set('size', _vector(panel_size))
        panel.set('pos', _vector((side * panel_size[0], 0., TABLE_HEIGHT - panel_size[2])))


def _scale_arm(arm, axes, home, settings):
    for axis, center in zip(axes, home, strict=True):
        joint = arm.find(f".//joint[@name='learner_{axis}']")
        bounds = np.fromstring(joint.get('range'), sep=' ')
        joint.set('range', _vector(center + settings.travel_scale * (bounds - center)))
    links = {body.get('name').removeprefix('learner_'): body for body in arm.iter('body')
             if body.find('joint') is not None}
    invalid = settings.link_mass_scales.keys() - links.keys()
    if invalid:
        raise ValueError(f'unknown moving learner links {sorted(invalid)}')
    for name, body in links.items():
        multiplier = settings.link_mass_scales.get(name, 1.)
        scale = settings.link_mass_scale * multiplier
        inertial = body.find('inertial')
        if inertial is not None:
            inertial.set('mass', str(float(inertial.get('mass')) * scale))
            inertia = np.fromstring(inertial.get('diaginertia'), sep=' ')
            inertial.set('diaginertia', _vector(inertia * scale))
        for geom in body.findall('geom'):
            geom_scale = multiplier if geom.get('name') == 'learner_racket' else scale
            mass = geom.get('mass')
            if mass is not None:
                geom.set('mass', str(float(mass) * geom_scale))
            else:
                density = float(geom.get('density', {'link': '470', 'pivot': '360'}[geom.get('class')]))
                geom.set('density', str(density * geom_scale))


def _add_payload(arm, body, settings):
    if settings.payload_mount == 'wrist':
        link = 'wrist_roll' if body == 'rotary' else 'wrist'
        pos = (.091, -.025, .099)
    else:
        link = _PROXIMAL_BODIES[body]
        pos = (-.105, -.045, .31)
    parent = arm.find(f".//body[@name='learner_{link}']")
    ET.SubElement(parent, 'geom', name=PAYLOAD, type='box', pos=_vector(pos), size='.027 .021 .019',
                  mass=str(settings.payload_mass), rgba='.18 .29 .35 1')
    ET.SubElement(parent, 'camera', name=MOUNTED_CAMERA, pos=_vector((pos[0], pos[1] + .023, pos[2])),
                  fovy=str(settings.camera_fovy))


class TT18PlantArena(TT8ServiceArena):
    """Keep the TT8 score and public stereo while varying the compiled MuJoCo plant."""

    def __init__(self, *args, plant=None, render_mode='mujoco', **kwargs):
        if render_mode != 'mujoco':
            raise ValueError('TT18 physical tables require native MuJoCo rendering')
        self.plant = PlantSettings() if plant is None else plant
        self.table_half_width = self.plant.table_half_width
        self.table_half_length = self.plant.table_half_length
        self._mounted_renderer = None
        super().__init__(*args, render_mode=render_mode, **kwargs)
        self.table_panels = ('table',)
        if self.plant.left_damping != self.plant.right_damping:
            self.table_panels += ('table_right',)
        if self.plant.payload_mount is not None:
            self._aim_mounted_camera()

    def _model_document(self):
        model = ET.parse(MODEL_PATH).getroot() if self.body == 'stock' else body_xml(self.body)
        world = model.find('worldbody')
        ball = world.find("body[@name='ball']/geom[@name='ball']")
        ball.set('mass', str(BALL_MASS * self.plant.ball_mass_scale))
        _resize_table(world, self.plant)
        arm = world.find("body[@name='learner_base']")
        axes, _, _ = self._arm_axes(1)
        home, _ = self._arm_home(1)
        _scale_arm(arm, axes, home, self.plant)
        if self.plant.payload_mount is not None:
            _add_payload(arm, self.body, self.plant)
        return model

    def _load_model(self):
        model = self._model_document()
        if self.venue == 'hall':
            return hall_model(model)
        return mujoco.MjModel.from_xml_string(ET.tostring(model, encoding='unicode'))

    def _aim_mounted_camera(self):
        camera = self.model.camera(MOUNTED_CAMERA)
        parent = int(camera.bodyid[0])
        world_rotation = self.data.xmat[parent].reshape(3, 3)
        eye = self.data.xpos[parent] + world_rotation @ camera.pos
        forward = np.array((0., .14, TABLE_HEIGHT + .14)) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0., 0., 1.))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = world_rotation.T @ np.column_stack((right, up, -forward))
        mujoco.mju_mat2Quat(camera.quat, np.ascontiguousarray(rotation).reshape(-1))

    def render_mounted_camera(self):
        """Capture one native RGB image from the camera attached to the moving arm."""
        if self.plant.payload_mount is None:
            raise ValueError('this plant has no mounted camera')
        if self._mounted_renderer is None:
            self._mounted_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self._mounted_renderer.update_scene(self.data, camera=MOUNTED_CAMERA)
        return self._mounted_renderer.render()

    def render_cameras(self, mode=None):
        if mode not in (None, 'mujoco'):
            raise ValueError('TT18 physical tables require native MuJoCo rendering')
        return super().render_cameras(mode)

    def _opening(self):
        super()._opening()
        # Preserve the toss's distance behind the actual end line. Scale the rival
        # serve's longitudinal geometry while keeping the stock launch generator.
        if self.server == 1:
            self.ball[1] += TABLE_HALF_LENGTH - self.table_half_length
        else:
            scale = self.table_half_length / TABLE_HALF_LENGTH
            self.ball[1] *= scale
            self.velocity[1] *= scale

    def _racket_contact(self, side, location, incoming):
        if not self._waiting_for_strike or side != self.server:
            super()._racket_contact(side, location, incoming)
            return
        self.contacts += 1
        self.service_contact = True
        self._event('contact', side=side, location=location, incoming=incoming,
                    racket=tuple(self.paddle.tolist()))
        if incoming[2] >= 0 or self._toss_apex - self._toss_height < MIN_TOSS_RISE:
            self._end(-side, 'serve_illegal_toss_strike')
        elif self.ball[1] >= -self.table_half_length or self.ball[2] < TABLE_HEIGHT:
            self._end(-side, 'serve_illegal_strike_position')
        else:
            self._waiting_for_strike = False
            self._event('serve_strike', side=side, location=location, incoming=incoming)
            self._choose_rival_response()

    def _contacts(self, incoming):
        if len(self.table_panels) == 1:
            super()._contacts(incoming)
            return
        if self._waiting_for_strike:
            self._toss_apex = max(self._toss_apex, self.ball[2])
        if not (self.data.ncon or self._previous_touches):
            return
        ball = self._geom_ids['ball']
        table = self._geom_ids['table']
        right = self.model.geom('table_right').id
        touches = {}
        for contact in self.data.contact:
            if contact.efc_address < 0 or ball not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == ball else contact.geom[0])
            if other == right:
                other = table
            location, vertical = tuple(contact.pos.tolist()), abs(float(contact.frame[2]))
            # Two adjacent panels can share a contact at the seam. Score one
            # tabletop bounce and prefer its top face over either interior edge.
            if other == table and other in touches and vertical < touches[other][1]:
                continue
            touches[other] = (location, vertical)
        for other, (location, vertical) in touches.items():
            if other in self._previous_touches:
                continue
            if other == table:
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
                self._event('separation', collider=self._collider_names[other], location=location)
        self._previous_touches = touches

    def close(self):
        if self._mounted_renderer is not None:
            self._mounted_renderer.close()
            self._mounted_renderer = None
        super().close()


def make_arena(spec, *, arena_type=TT18PlantArena):
    """Construct a TT17-compatible arena from a named or serialized evaluator plant."""
    plant = spec.get('plant')
    if plant is None and arena_type is TT18PlantArena:
        return make_tt17_arena(spec)
    if isinstance(plant, str):
        settings = PLANTS[plant]
    elif isinstance(plant, PlantSettings):
        settings = plant
    else:
        settings = PlantSettings(**(plant or {}))
    arena = arena_type(**spec['arena'], plant=settings)
    arena._rig = tuple(PerspectiveCamera(**eye) for eye in spec['camera'])
    return arena

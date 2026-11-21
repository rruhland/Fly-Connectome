"""An end-clamped SO-101 station with an external, physically opaque wrist camera."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt18_view_trial import configure_view_colors
from .tt19_so101 import CAMERA_BODY, GRIP_BODY, TIP, WRIST_CAMERA
from .tt19_view import STREAMS as TT19_STREAMS, WRIST_FOV
from .tt19_view_trial import CAMERAS, SO101ViewArena


STREAMS = dict(TT19_STREAMS, outside=('left',), left=('left',), right=('right',), side=('side',))
CLAMP_FORWARD = .035
CLAMP_RISE = .022
WRIST_APERTURE = (0., .190, -.210)  # metres, in the gripper body frame
RAISED_APERTURE = (0., .240, -.290)
OUTBOARD_FOV = 84.
ULTRAWIDE_FOV = 126.
ULTRAWIDE_PITCH = 20.
NEAR_M = .003
BALL_ORANGE = (.97, .42, .05)
BLADE_RED = (.79, .085, .105)
RIVAL_RUBBER = (.19, .20, .22)


def _vector(values):
    return ' '.join(f'{value:.10g}' for value in values)


def _box(parent, name, position, half_size, rgba):
    return ET.SubElement(parent, 'geom', name=name, type='box', pos=_vector(position),
                         size=_vector(half_size), rgba=_vector((*rgba, 1.)), contype='2', conaffinity='5')


def _clamp(document, depth, flange_height, table_half_length, table_half_width):
    world = document.find('worldbody')
    world.remove(world.find("geom[@name='tt19_pedestal']"))
    support = ET.SubElement(world, 'body', name='tt20_table_end_clamp')
    table = world.find("geom[@name='table']")
    underside = float(table.get('pos').split()[2]) - float(table.get('size').split()[2])
    base_y, end = -depth, -table_half_length
    arm_x = .67 * table_half_width
    span = end + .067 - base_y
    steel, zinc, rubber = (.16, .22, .26), (.43, .48, .50), (.13, .14, .15)
    # The dedicated underside adapter rail is bolted inboard of the table edge.
    # The paired jaws grip this rail below the tabletop, leaving the top clear.
    _box(support, 'tt20_clamp_apron', (0., end + .071, underside - .042),
         (arm_x + .043, .033, .027), steel)
    _box(support, 'tt20_clamp_apron_pad', (0., end + .071, underside - .0075),
         (arm_x + .050, .038, .0075), zinc)
    _box(support, 'tt20_clamp_flange', (0., base_y, flange_height - .009), (.108, .092, .009), steel)
    _box(support, 'tt20_clamp_crossbar', (0., base_y, flange_height - .036),
         (arm_x + .035, .035, .018), steel)
    low = flange_height - .036
    for index, x in enumerate((-arm_x, arm_x)):
        prefix = f'tt20_clamp_{index}'
        _box(support, prefix + '_beam', (x, base_y + span / 2, low), (.024, span / 2, .018), steel)
        top = underside - .070
        _box(support, prefix + '_riser', (x, end + .047, (low + top) / 2),
             (.025, .025, (top - low) / 2), steel)
        _box(support, prefix + '_lower_jaw', (x, end + .071, underside - .090), (.037, .049, .011), zinc)
        _box(support, prefix + '_nose', (x, end + .018, underside - .048), (.035, .013, .054), steel)
        _box(support, prefix + '_upper_jaw', (x, end + .049, underside - .018), (.037, .042, .008), zinc)
        _box(support, prefix + '_shoe', (x, end + .078, underside - .076), (.026, .024, .007), rubber)
        ET.SubElement(support, 'geom', name=prefix + '_screw', type='cylinder',
                      pos=_vector((x, end + .078, underside - .114)), size='.008 .023',
                      rgba='.50 .54 .55 1', contype='2', conaffinity='5')
        ET.SubElement(support, 'geom', name=prefix + '_handle', type='capsule',
                      fromto=_vector((x - .024, end + .078, underside - .127,
                                      x + .024, end + .078, underside - .127)),
                      size='.0055', rgba='.58 .61 .63 1', contype='2', conaffinity='5')
        for dx in (-.021, .021):
            position = 'left' if dx < 0 else 'right'
            ET.SubElement(support, 'geom', name=f'{prefix}_apron_bolt_{position}',
                          type='cylinder', pos=_vector((x + dx, end + .090, underside - .014)),
                          size='.006 .005', rgba='.55 .59 .61 1', contype='0', conaffinity='0')


def _outboard(document, fovy, aperture):
    gripper = document.find(f".//body[@name='{GRIP_BODY}']")
    dock = gripper.find("body[@name='learner_wrist_camera_mount']")
    camera_body = dock.find(f"body[@name='{CAMERA_BODY}']")
    dock.remove(camera_body)
    gripper.append(camera_body)
    camera_body.set('pos', _vector(aperture))
    camera_body.set('quat', '1 0 0 0')
    camera_body.find('inertial').set('pos', '0 0 .014')
    for geom in camera_body.findall('geom'):
        camera_body.remove(geom)
    lens = camera_body.find(f"camera[@name='{WRIST_CAMERA}']")
    lens.set('pos', '0 0 0')
    lens.set('fovy', str(fovy))
    # The optical origin sits just beyond the lens. The opaque module and
    # aperture barrel are behind it along the camera's +Z axis.
    _box(camera_body, 'tt20_rgb_housing', (0., 0., .015), (.018, .017, .011), (.12, .15, .17))
    ET.SubElement(camera_body, 'geom', name='tt20_rgb_barrel', type='cylinder', pos='0 0 .0065',
                  size='.009 .005', rgba='.20 .25 .28 1', contype='2', conaffinity='5')
    ET.SubElement(camera_body, 'geom', name='tt20_rgb_lens', type='cylinder', pos='0 0 .0028',
                  size='.0068 .0016', rgba='.045 .10 .15 1', contype='0', conaffinity='0')
    boom_end = np.add(aperture, (.016, .007, .009))
    ET.SubElement(gripper, 'geom', name='tt20_camera_boom', type='capsule', size='.006', density='900',
                  fromto=_vector((-.014, .036, -.040, *boom_end)), rgba='.17 .21 .23 1', contype='0', conaffinity='0')
    visual = document.find('visual')
    extent = float(document.find('statistic').get('extent', '8'))
    visual.find('map').set('znear', str(NEAR_M / extent))


def _unmarked(document):
    # Remove only the synthetic survey and role tags. Keep the white playing
    # lines, net tape, original jaw and the actual paddle contact geometry.
    for parent in document.iter():
        for item in list(parent):
            name = item.get('name', '')
            if name.startswith(('tt18_landmark_', 'tt19_rim_', 'tt19_tip_mark')):
                parent.remove(item)


class TT20Arena(SO101ViewArena):
    """Select a physically moved clamp, an externally mounted lens, and public RGB streams."""

    def __init__(self, *args, mount='clamp', wrist_optics='outboard', marked=False,
                 view_mode='wrist+fixed', width=None, height=None, wrist_fovy=None,
                 role_palette='natural', base_height=None, venue='hall', **kwargs):
        if mount not in ('clamp', 'pedestal'):
            raise ValueError('expected clamp or pedestal')
        if wrist_optics not in ('outboard', 'raised', 'ultrawide', 'legacy') or view_mode not in STREAMS:
            raise ValueError('unknown wrist optic or stream selection')
        if role_palette not in ('natural', 'legacy'):
            raise ValueError('expected natural or legacy role colors')
        if wrist_fovy is not None and (not math.isfinite(wrist_fovy) or not 45. <= wrist_fovy <= 135.):
            raise ValueError('expected a vertical wrist field of view between 45 and 135 degrees')
        fovy = {'legacy': WRIST_FOV, 'ultrawide': ULTRAWIDE_FOV}.get(wrist_optics, OUTBOARD_FOV)
        dimensions = (960, 720) if wrist_optics == 'ultrawide' else (480, 360)
        width = dimensions[0] if width is None else width
        height = dimensions[1] if height is None else height
        self.mount = mount
        self.wrist_optics = wrist_optics
        self.marked = marked
        self.role_palette = role_palette
        self.wrist_fovy = fovy if wrist_fovy is None else wrist_fovy
        base_height = .49 + (CLAMP_RISE if mount == 'clamp' else 0.) if base_height is None else base_height
        inherited_mode = view_mode if view_mode in TT19_STREAMS else 'fixed'
        self.streams = STREAMS[view_mode]
        super().__init__(*args, view_mode=inherited_mode, width=width, height=height, base_height=base_height,
                         venue=venue, **kwargs)
        self.view_mode = view_mode
        self._playing_colliders = set(self._collider_names)
        self._playing_colliders.update(self.model.geom(name).id for name in self.table_panels)
        self._collider_names.update({index: self.model.geom(index).name or f'geom_{index}'
                                     for index in range(self.model.ngeom) if index not in self._collider_names})

    def _model_document(self):
        if self.mount == 'clamp':
            self.base_depth = self.plant.table_half_length + .375 - CLAMP_FORWARD
        document = super()._model_document()
        if self.mount == 'clamp':
            _clamp(document, self.base_depth, self.base_height, self.plant.table_half_length, self.plant.table_half_width)
        if self.wrist_optics != 'legacy':
            aperture = RAISED_APERTURE if self.wrist_optics == 'raised' else WRIST_APERTURE
            _outboard(document, self.wrist_fovy, aperture)
        if not self.marked:
            _unmarked(document)
        return document

    def _orient_wrist(self):
        if self.wrist_optics == 'legacy':
            super()._orient_wrist()
            self.model.camera(WRIST_CAMERA).fovy[0] = self.wrist_fovy
        else:
            camera = self.model.camera(WRIST_CAMERA)
            body = int(camera.bodyid[0])
            gripper = self.data.body(GRIP_BODY)
            eye = self.data.xpos[body].copy()
            jaw_and_blade = (.52 * self.data.site(TIP).xpos + .48 * self.paddle) - eye
            court = np.array((0., .025, TABLE_HEIGHT + .11)) - eye
            jaw_and_blade /= np.linalg.norm(jaw_and_blade)
            court /= np.linalg.norm(court)
            forward = .31 * jaw_and_blade + .69 * court
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, (0., 0., 1.))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            axes = gripper.xmat.reshape(3, 3).T @ np.column_stack((right, up, -forward))
            if self.wrist_optics == 'ultrawide':
                angle = math.radians(ULTRAWIDE_PITCH)
                cosine, sine = math.cos(angle), math.sin(angle)
                axes = axes @ np.array(((1., 0., 0.), (0., cosine, -sine), (0., sine, cosine)))
            mujoco.mju_mat2Quat(self.model.body(body).quat, np.ascontiguousarray(axes).ravel())
        mujoco.mj_forward(self.model, self.data)

    def _contacts(self, incoming):
        support = None
        if self.data.ncon:
            ball = self._geom_ids['ball']
            for contact in self.data.contact:
                if contact.efc_address < 0 or ball not in contact.geom:
                    continue
                other = int(contact.geom[1] if contact.geom[0] == ball else contact.geom[0])
                if other not in self._playing_colliders:
                    support = self._collider_names[other], tuple(contact.pos.tolist())
                    break
        super()._contacts(incoming)
        if support is not None and not (self.terminated or self.truncated):
            collider, location = support
            self._event('fixture_contact', collider=collider, location=location, incoming=incoming)
            if self._waiting_for_strike:
                self._end(-self.server, 'serve_missed_toss')
            else:
                self._end(self._last_hitter if self._received_bounce else -self._last_hitter, 'fixture_contact')

    def lens_intrinsics(self):
        """Public native image dimensions and lens values for each available stream."""
        result = {}
        for key in ('wrist', 'left', 'right', 'side', 'broadcast'):
            fovy = self.wrist_fovy if key == 'wrist' else CAMERAS[key][-1]
            result[key] = dict(width=self.width, height=self.height, fovy=fovy,
                               center=((self.width - 1) / 2, (self.height - 1) / 2),
                               focal=self.height / (2 * math.tan(math.radians(fovy) / 2)))
        return result

    def _color_roles(self):
        if self.role_palette == 'legacy':
            configure_view_colors(self)
        else:
            for name, rgb in (('ball', BALL_ORANGE), ('learner_racket', BLADE_RED), ('rival_racket', RIVAL_RUBBER)):
                self.model.geom(name).rgba[:3] = rgb

    def render_cameras(self, mode=None):
        self._color_roles()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        renderer = self._view_renderer()
        return {stream: self._render_view(renderer, self.data, WRIST_CAMERA if stream == 'wrist' else CAMERAS[stream][0])
                for stream in self.streams}

    def render_wrist(self):
        self._color_roles()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return self._render_view(self._view_renderer(), self.data, WRIST_CAMERA)

"""Native SO-101 RGB views and visible, passive calibration marks."""
from colorsys import hsv_to_rgb
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_camera import PerspectiveCamera
from .tt18_view_trial import add_view_landmarks, configure_view_colors
from .tt19_so101 import SO101Arena, TIP, WRIST_CAMERA
from .tt19_view import LANDMARKS, MODES, RIM, RIM_HUES, STREAMS, TIP_HUE, WRIST, WRIST_FOV


SIZE = (320, 240)
BROADCAST_CAMERA = 'tt19_broadcast'
CAMERAS = {
    'left': ('tt19_left', (-.70, -2.58, 1.12), (0., -.19, .15), 57.),
    'right': ('tt19_right', (.75, -2.62, 1.16), (0., -.19, .15), 57.),
    'side': ('tt19_side', (1.63, -1.73, 1.31), (0., -.27, .18), 62.),
    'broadcast': (BROADCAST_CAMERA, (1.68, -2.13, 1.50), (0., -.24, .04), 55.),
}
def _vector(values):
    return ' '.join(f'{value:.8g}' for value in values)


def _camera(world, name, eye, target, fovy):
    camera = PerspectiveCamera(eye, target, *SIZE, fovy)
    right, up, forward = camera.basis
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
    ET.SubElement(world, 'camera', name=name, pos=_vector(np.array(eye) + (0., 0., TABLE_HEIGHT)),
                  quat=_vector(quaternion), fovy=str(fovy))


def _paint(asset, parent, name, position, hue, radius):
    material = f'{name}_mat'
    rgb = hsv_to_rgb(hue, .99, .66)
    ET.SubElement(asset, 'material', name=material, rgba=_vector((*rgb, 1.)),
                  shininess='0', emission='.5', specular='0')
    ET.SubElement(parent, 'geom', name=name, type='sphere', size=str(radius), pos=_vector(position),
                  contype='0', conaffinity='0', mass='0', material=material)


def add_so_view_geometry(document):
    """Install RGB fixtures and preserve the plant's collision and inertial model."""
    add_view_landmarks(document, LANDMARKS)
    world, asset = document.find('worldbody'), document.find('asset')
    for values in CAMERAS.values():
        _camera(world, *values)
    paddle = next(parent for parent in world.iter('body') if parent.find("geom[@name='learner_racket']") is not None)
    racket = paddle.find("geom[@name='learner_racket']")
    center = np.fromstring(racket.get('pos', '0 0 0'), sep=' ')
    quaternion = np.fromstring(racket.get('quat', '1 0 0 0'), sep=' ')
    rotation = Rotation.from_quat(np.roll(quaternion, -1))
    for index, (position, hue) in enumerate(zip(RIM, RIM_HUES, strict=True)):
        _paint(asset, paddle, f'tt19_rim_{index}', center + rotation.apply(position), hue, .0095)
    gripper = next(parent for parent in world.iter('body') if parent.find("site[@name='learner_tip']") is not None)
    tip = gripper.find("site[@name='learner_tip']")
    position = np.fromstring(tip.get('pos', '0 0 0'), sep=' ')
    _paint(asset, gripper, 'tt19_tip_mark', position, TIP_HUE, .009)
    return document


class SO101ViewArena(SO101Arena):
    """Return synchronized RGB under stable stream IDs for the chosen camera rig."""

    def __init__(self, *args, view_mode='wrist+fixed', width=SIZE[0], height=SIZE[1],
                 detail='preview', **kwargs):
        if view_mode not in MODES:
            raise ValueError(f'unknown SO-101 camera mode {view_mode!r}')
        self.view_mode = view_mode
        super().__init__(*args, width=width, height=height, detail=detail, **kwargs)

    def _model_document(self):
        return add_so_view_geometry(super()._model_document())

    def _orient_wrist(self):
        camera = self.model.camera(WRIST_CAMERA)
        parent = int(camera.bodyid[0])
        mount = self.data.xmat[parent].reshape(3, 3)
        eye = self.data.xpos[parent] + mount @ camera.pos
        nearby = .48 * self.data.site(TIP).xpos + .52 * self.paddle - eye
        court = np.array((0., -.04, TABLE_HEIGHT + .12)) - eye
        nearby /= np.linalg.norm(nearby)
        court /= np.linalg.norm(court)
        forward = .45 * nearby + .55 * court
        forward /= np.linalg.norm(forward)
        right = court - (court @ forward) * forward
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rotation = mount.T @ np.column_stack((right, up, -forward))
        mujoco.mju_mat2Quat(camera.quat, np.ascontiguousarray(rotation).ravel())
        camera.fovy[0] = WRIST_FOV

    def _view_renderer(self):
        if self._renderer is None:
            size = self.model.vis.global_
            width, height = size.offwidth, size.offheight
            quality = self.model.vis.quality
            samples = quality.offsamples
            try:
                size.offwidth, size.offheight = self.width, self.height
                if self.detail == 'preview':
                    quality.offsamples = 0
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            finally:
                size.offwidth, size.offheight = width, height
                quality.offsamples = samples
        return self._renderer

    @staticmethod
    def _render_view(renderer, data, camera):
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return renderer.render()

    def render_cameras(self, mode=None):
        configure_view_colors(self)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        renderer = self._view_renderer()
        frames = {}
        for stream in STREAMS[self.view_mode]:
            camera = WRIST_CAMERA if stream == WRIST else CAMERAS[stream][0]
            frames[stream] = self._render_view(renderer, self.data, camera)
        return frames

    def render_broadcast(self, renderer=None):
        renderer = self._view_renderer() if renderer is None else renderer
        return self._render_view(renderer, self.data, BROADCAST_CAMERA)

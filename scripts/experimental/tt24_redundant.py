"""An SO-101 variant with a separately actuated distal pitch bracket."""

import math
import xml.etree.ElementTree as ET

import numpy as np

from .tt19_so101 import AXES, BLADE, GRIP_BODY, PADDLE_BODY, RATES
from .tt24_bounce import BounceArena, _camera_axes, mujoco
from .tt20_fixture import _vector


EXTRA = 'distal_pitch'
_LOCK = 'tt24_distal_lock'
_JAM = 'tt24_original_pitch_lock'
_CALIBRATION_CAMERAS = ('tt24_pose_a', 'tt24_pose_b')
_MARKER_GROUP = 4


def _marks(document):
    world = document.find('worldbody')
    paddle = world.find(f"body[@name='{PADDLE_BODY}']")
    blade = paddle.find(f"geom[@name='{BLADE}']")
    move = np.fromstring(blade.get('pos'), sep=' ')
    quaternion = np.fromstring(blade.get('quat'), sep=' ')
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    orient = matrix.reshape(3, 3)
    palette = ('0.02 1 .99 1', '1 .02 .99 1', '.03 .99 .02 1')
    for idx, (angle, rgba) in enumerate(zip((.13, 2.16, 4.15), palette, strict=True)):
        place = .088 * np.array((math.cos(angle), math.sin(angle), 0.))
        ET.SubElement(paddle, 'site', name=f'tt24_probe_marker_{idx}', type='sphere', size='.0095',
                      pos=_vector(move + orient @ place), rgba=rgba, group=str(_MARKER_GROUP))
    target = np.array((0., -.96, .91))
    for name, position in zip(_CALIBRATION_CAMERAS, ((.61, -1.74, 1.64), (-.68, -1.27, 1.54)), strict=True):
        eye = np.asarray(position)
        matrix = _camera_axes(eye, target)
        ET.SubElement(world, 'camera', name=name, pos=_vector(eye), fovy='31',
                      xyaxes=_vector(matrix[:, :2].T.ravel()))


def _bracket(document, pinned, jammed):
    wrist = document.find(".//body[@name='learner_wrist']")
    hand = wrist.find(f"body[@name='{GRIP_BODY}']")
    old = np.fromstring(hand.get('pos'), sep=' ')
    pivot = np.array((0., -.032, .009))
    wrist.remove(hand)
    link = ET.SubElement(wrist, 'body', name='learner_distal_bracket', pos=_vector(pivot))
    ET.SubElement(link, 'joint', name=f'learner_{EXTRA}', type='hinge', axis='.11 0 .99393', range='-.48 .48',
                  armature='.013', damping='.43', frictionloss='.034')
    ET.SubElement(link, 'geom', name='learner_distal_shell', type='box', size='.015 .019 .013',
                  pos='0 -.012 0', mass='.038', contype='0', conaffinity='1', rgba='.30 .36 .40 1')
    hand.set('pos', _vector(old - pivot))
    link.append(hand)
    actuators = document.find('actuator')
    jaw = actuators.find("position[@name='learner_gripper']")
    motor = ET.Element('position', name=f'learner_{EXTRA}', joint=f'learner_{EXTRA}', kp='780', kv='2.3',
                       ctrlrange='-.48 .48', forcerange='-3.2 3.2')
    actuators.insert(list(actuators).index(jaw), motor)
    ET.SubElement(document.find('equality'), 'joint', name=_LOCK, joint1=f'learner_{EXTRA}',
                  polycoef='0 0 0 0 0', active='true' if pinned else 'false', solref='.003 1',
                  solimp='.999 .9999 .0005')
    if jammed:
        ET.SubElement(document.find('equality'), 'joint', name=_JAM, joint1='learner_wrist_flex',
                      polycoef='0 0 0 0 0', solref='.003 1', solimp='.999 .9999 .0005')
    document.set('model', 'TT24 six-actuated-joint arm')


class MarkerBounce(BounceArena):
    """Stock body carrying optical calibration marks hidden in the game view."""

    def __init__(self, *, calibration=False, **kwargs):
        self.calibration = calibration
        super().__init__(**kwargs)
        self._scene_option.sitegroup[_MARKER_GROUP] = 0
        self._probe_option = mujoco.MjvOption()
        self._probe_option.sitegroup[:] = 0
        self._probe_option.sitegroup[_MARKER_GROUP] = 1
        self._probe_option.geomgroup[3] = 0
        if calibration:
            self._model.geom_contype[self._ball_geom] = 0
            self._model.geom_conaffinity[self._ball_geom] = 0

    def _document(self, width, height, detail, arm):
        document = super()._document(width, height, detail, arm)
        _marks(document)
        return document

    def _judge_substep(self, incoming):
        if self.calibration:
            return 0.
        return super()._judge_substep(incoming)

    def probe_images(self):
        return tuple(self._render_camera(cam, self._probe_option) for cam in _CALIBRATION_CAMERAS)


class RedundantBounce(MarkerBounce):
    """Five legacy encoders and a sixth actuator on a moving distal bracket."""

    action_axes = (*AXES[:5], EXTRA)
    action_size = len(action_axes)
    _joint_axes = (*action_axes, AXES[-1])
    _rates = (*RATES[:3], 1.72, RATES[4], 5.85, 0.)
    _changed_drive = np.insert(BounceArena._changed_drive, 5, .79)
    _changed_force = np.insert(BounceArena._changed_force, 5, .86)

    def __init__(self, *, pinned=False, jammed=False, **kwargs):
        self.pinned = pinned
        self.jammed = jammed
        super().__init__(**kwargs)
        self.arm = 'redundant'
        if jammed:
            self.arm += '_jammed'
        if pinned:
            self.arm += '_pinned'

    def _document(self, width, height, detail, arm):
        document = super()._document(width, height, detail, arm)
        _bracket(document, self.pinned, self.jammed)
        return document

    def _actuated_pose(self, pose):
        return np.insert(pose, 5, 0.)

    def _prepare_pose(self, pose):
        if self.jammed:
            model = self._model
            index = self.action_axes.index('wrist_flex')
            angle = self._actuated_pose(pose)[index]
            reference = model.qpos0[self._joint_position[index]]
            model.eq_data[model.equality(_JAM).id, 0] = angle - reference
        super()._prepare_pose(pose)

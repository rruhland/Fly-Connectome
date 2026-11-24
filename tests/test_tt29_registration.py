"""Check that ordinary court RGB recovers a metric external camera remount."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
from mujoco.rendering.classic.renderer import GLContext

from scripts.experimental.tt19_view_trial import CAMERAS
from scripts.experimental.tt25_reference_receiver import ReferenceReceiver
from scripts.experimental.tt27_world_camera import CameraMounts
from scripts.experimental.tt29_registration import _accuracy, register
from scripts.experimental.tt29_vision import Rig


def test_registration_requires_visible_court_lines():
    arena = ReferenceReceiver(width=384, height=288)
    try:
        nominal = Rig.from_arena(arena).fixed['left']
        with pytest.raises(ValueError, match='boundary pixels'):
            register(np.zeros((288, 384, 3), dtype=np.uint8), nominal)
    finally:
        arena.close()


@pytest.mark.skipif(GLContext is None or GLContext.__module__ != 'mujoco.egl',
                    reason='run with MUJOCO_GL=egl to test native registration RGB')
def test_rgb_only_registration_recovers_independent_metric_remounts():
    arena = ReferenceReceiver(width=384, height=288)
    try:
        nominal = Rig.from_arena(arena)
        CameraMounts(arena).move(((3.8, -3.4, 2.5), (-3.0, 4.1, -3.8)))
        for eye, delta in zip(('left', 'right'), ((.09, -.08, .04), (-.05, .12, .03)), strict=True):
            arena.model.camera(CAMERAS[eye][0]).pos[:] += delta
        actual = Rig.from_arena(arena)
        for eye, rgb in arena.render_cameras().items():
            recovered = register(rgb, nominal.fixed[eye])
            errors = _accuracy(recovered.camera, actual.fixed[eye])
            stale = _accuracy(nominal.fixed[eye], actual.fixed[eye])
            assert errors['position_mm'] < 18.
            assert errors['angle_deg'] < .40
            assert errors['volume_reprojection_px'] < .85
            assert errors['volume_reprojection_px'] < stale['volume_reprojection_px'] / 10
    finally:
        arena.close()

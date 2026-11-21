"""The native hall cache preserves shaded stereo and physical state."""

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec('mujoco') is None:
    pytest.skip('table-tennis optional dependencies are required', allow_module_level=True)

from mujoco.rendering.classic.renderer import GLContext

if GLContext is None or GLContext.__module__ != 'mujoco.egl':
    pytest.skip('run with MUJOCO_GL=egl to exercise native hall rendering', allow_module_level=True)

pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_arena import BODIES, TT6Arena
from scripts.experimental.tt6_render import cache_hall_stereo


@pytest.mark.parametrize('body', BODIES)
def test_cached_reflections_preserve_moving_stereo_images(body):
    configuration = dict(body=body, width=192, height=144, baseline=.28,
                         venue='hall', distractor=True)
    native = TT6Arena(69237, **configuration)
    cached = TT6Arena(69237, **configuration)
    try:
        cache_hall_stereo(cached, reflections=True)
        rng = np.random.default_rng(719)
        for seed in (69237, 69238):
            native_start = native.reset(seed=seed)
            np.testing.assert_array_equal(cached.reset(seed=seed), native_start)
            action = rng.uniform(-1., 1., native.action_size)
            native_frame, *native_result = native.step(action)
            cached_frame, *cached_result = cached.step(action)
            assert np.any(native_frame != native_start)
            np.testing.assert_array_equal(cached_frame, native_frame)
            np.testing.assert_array_equal(cached.data.qpos, native.data.qpos)
            np.testing.assert_array_equal(cached.data.qvel, native.data.qvel)
            assert cached_result == native_result
    finally:
        native.close()
        cached.close()

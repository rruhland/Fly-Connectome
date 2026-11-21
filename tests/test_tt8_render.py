import importlib.util
import numpy as np
import pytest

if importlib.util.find_spec('mujoco') is None:
    pytest.skip('MuJoCo is required', allow_module_level=True)
from mujoco.rendering.classic.renderer import GLContext

if GLContext is None or GLContext.__module__ != 'mujoco.egl':
    pytest.skip('requires the native EGL renderer', allow_module_level=True)

from scripts.experimental.tt4_transfer_trial import ROOT, _read
from scripts.experimental.tt6_body_arena import TT6Wiring
from scripts.experimental.tt6_continuous_trial import prepare_point
from scripts.experimental.tt6_render import cache_hall_stereo
from scripts.experimental.tt8_service_arena import TT8ServiceArena
from scripts.experimental.tt8_service_trial import PointInterface


def test_compiled_renderer_matches_both_eyes_during_near_side_toss():
    receipt = _read(ROOT/'artifacts/tt8/service-warm-gantry.json.gz')
    row = receipt['points'][10]
    sequences = []
    for compiled in (False, True):
        arena = TT8ServiceArena(**receipt['arena'])
        try:
            cache_hall_stereo(arena, reflections=True, compiled_geometry=compiled)
            wiring = TT6Wiring(arena, receipt['installation_seed'])
            game = PointInterface(wiring, prepare_point(arena, row['phase']),
                                  server=row['server'], seed=receipt['seed'])
            frames = [game.reset(seed=row['seed'])]
            for action in row['actions'][:12]:
                frames.append(game.step(action)[0])
            sequences.append(frames)
        finally:
            arena.close()
    np.testing.assert_array_equal(*sequences)

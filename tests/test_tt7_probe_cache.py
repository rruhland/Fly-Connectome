from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt6_axes import TiltedAxisArena
from scripts.experimental.tt6_body_arena import TT6Arena, TT6Wiring
from scripts.experimental.tt6_body_probe import probe_commands
from scripts.experimental.tt6_continuous import BodyLibrary
from scripts.experimental.tt6_timing import DelayedWiring
from scripts.experimental.tt7_probe_cache import ProbeCache


SCHEDULE = dict(calibration=[93, 109, 123, 131, 153], player_seed=37)


def setup(tmp_path, *, arena_type=TT6Arena, wiring_seed=41, schedule=SCHEDULE, **overrides):
    geometry = dict(width=32, height=24, max_steps=2, view_seed=7, render_mode='projected')
    arena = arena_type(51, **(geometry | overrides))
    wiring = DelayedWiring(TT6Wiring(arena, wiring_seed))
    return ProbeCache(tmp_path, wiring, arena.cameras, schedule, 5), wiring


def render_calls(monkeypatch):
    calls = []
    original = TT6Arena.render_cameras

    def render_cameras(self, *args, **kwargs):
        calls.append(self.steps)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TT6Arena, 'render_cameras', render_cameras)
    return calls


def test_cached_probe_preserves_rgb_commands_and_final_state(tmp_path, monkeypatch):
    calls = render_calls(monkeypatch)
    cold, before = setup(tmp_path)
    warm, after = setup(tmp_path)
    commands = probe_commands(37, 5, before.action_size)
    expected = cold.collect(before, 93, commands)
    assert calls == [0, 1, 2]
    assert before.arena.truncated and len(expected) == 3
    calls.clear()
    actual = warm.collect(after, 93, commands)
    assert calls == []
    for (first_action, first_rgb), (second_action, second_rgb) in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first_action, second_action)
        np.testing.assert_array_equal(first_rgb, second_rgb)
    np.testing.assert_array_equal(before.arena.data.qpos, after.arena.data.qpos)
    np.testing.assert_array_equal(before.arena.data.qvel, after.arena.data.qvel)
    assert before.events == after.events
    assert before.point_result() == after.point_result()
    assert before.applied == after.applied
    np.testing.assert_array_equal(before.arena.render_cameras(), after.arena.render_cameras())

    # An incompatible file must be recaptured in full.
    artifact = next(tmp_path.glob('*.npz'))
    artifact.write_bytes(b'interrupted write')
    calls.clear()
    repaired = warm.collect(after, 93, commands)
    assert calls == [0, 1, 2]
    np.testing.assert_array_equal(repaired[-1][1], expected[-1][1])
    assert not list(tmp_path.glob('*.tmp'))


def test_probe_inputs_and_actual_model_separate_cache_keys(tmp_path):
    first, wiring = setup(tmp_path)
    assert first.key == setup(tmp_path, schedule=SCHEDULE | {'tests': [143]})[0].key
    for changes in (dict(view_seed=8), dict(wiring_seed=42), dict(baseline=.28),
                    dict(max_steps=3), dict(frameskip=4), dict(body='gantry'),
                    dict(schedule=SCHEDULE | {'calibration': [94, 109, 123, 131, 153]}),
                    dict(schedule=SCHEDULE | {'player_seed': 38})):
        assert first.key != setup(tmp_path, **changes)[0].key
    standard, _ = setup(tmp_path, arena_type=TiltedAxisArena, layout='standard')
    tilted, _ = setup(tmp_path, arena_type=TiltedAxisArena, layout='held-a')
    assert standard.key != tilted.key

    commands = probe_commands(37, 5, wiring.action_size)
    first.collect(wiring, 93, commands)
    commands[0, 0] = .7
    first.collect(wiring, 93, commands)
    assert len(list(tmp_path.glob('*.npz'))) == 2


def test_same_probe_is_captured_once_by_concurrent_writers(tmp_path, monkeypatch):
    calls = render_calls(monkeypatch)
    barrier = Barrier(2)

    def collect():
        cache, wiring = setup(tmp_path)
        commands = probe_commands(37, 5, wiring.action_size)
        barrier.wait(timeout=10)
        return cache.collect(wiring, 93, commands)

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = tuple(pool.map(lambda _: collect(), range(2)))
    assert calls == [0, 1, 2]
    assert len(list(tmp_path.glob('*.npz'))) == 1
    for (_, first), (_, second) in zip(left, right, strict=True):
        np.testing.assert_array_equal(first, second)


def test_commissioning_only_captures_requested_probes(tmp_path, monkeypatch):
    from scripts.experimental import tt6_continuous as control

    calls = render_calls(monkeypatch)
    monkeypatch.setattr(control, 'body_tracker_factory', lambda rig, histories: None)
    monkeypatch.setattr(control, 'prediction_error', lambda *args: {'error': 0.})
    monkeypatch.setattr(control, 'supports', lambda *args: True)
    monkeypatch.setattr(control, 'fit_camera_body', lambda *args: 'refit')
    model = type('Model', (), {'action_size': 4})()
    _, first = setup(tmp_path)
    _, second = setup(tmp_path)
    recalled, _, probes, attempts = BodyLibrary([model]).commission(
        first, first.arena.cameras, SCHEDULE, horizon=5, probe_cache=tmp_path)
    assert recalled is model and len(probes) == 1 and attempts[-1]['recalled'] == 0
    assert len(list(tmp_path.glob('*.npz'))) == 1
    calls.clear()
    fitted, _, probes, attempts = BodyLibrary([]).commission(
        second, second.arena.cameras, SCHEDULE, horizon=5, probe_cache=tmp_path)
    assert fitted == 'refit' and len(probes) == 3 and attempts[-1]['learned'] == 0
    assert len(list(tmp_path.glob('*.npz'))) == 3
    assert calls == [0, 1, 2, 0, 1, 2]


def test_native_and_reflecting_renderers_have_separate_cache_keys(tmp_path):
    from mujoco.rendering.classic.renderer import GLContext

    if GLContext is None or GLContext.__module__ != 'mujoco.egl':
        pytest.skip('requires EGL for native hall rendering')
    from scripts.experimental.tt6_render import cache_hall_stereo

    geometry = dict(venue='hall', render_mode='mujoco', width=32, height=24)
    arena = TT6Arena(51, **geometry)
    wiring = DelayedWiring(TT6Wiring(arena, 41))
    try:
        native = ProbeCache(tmp_path, wiring, arena.cameras, SCHEDULE, 5)
        cache_hall_stereo(arena, reflections=True)
        reflecting = ProbeCache(tmp_path, wiring, arena.cameras, SCHEDULE, 5)
        assert native.key != reflecting.key
    finally:
        arena.close()

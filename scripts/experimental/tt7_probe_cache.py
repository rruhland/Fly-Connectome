"""Share complete public RGB trajectories between matched calibration runs."""

from dataclasses import astuple
import fcntl
from functools import lru_cache
from hashlib import sha256
import inspect
import json
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np

from . import tt3_arena, tt4_camera, tt4_hall, tt6_body_probe
from .tt4_arena import mujoco


def _source_versions(*objects):
    paths = sorted({Path(inspect.getfile(obj)) for obj in objects})
    return sorted(sha256(path.read_bytes()).hexdigest() for path in paths)


def _types(instance_type):
    return instance_type.__mro__[:-1]


@lru_cache(maxsize=1)
def _native_driver(context_type):
    context = context_type(1, 1)
    try:
        return _gl_driver(context)
    finally:
        context.free()


def _gl_driver(context):
    import OpenGL
    from OpenGL import GL

    context.make_current()
    strings = [GL.glGetString(token).decode() for token in
               (GL.GL_VENDOR, GL.GL_RENDERER, GL.GL_VERSION)]
    return [*strings, OpenGL.__version__]


def _renderer_version(arena):
    if arena.render_mode == 'projected':
        return dict(mode='projected')
    from mujoco.rendering.classic.renderer import GLContext

    if arena._renderer is None:
        renderer_type = mujoco.Renderer
        context_type = GLContext
        driver = _native_driver(context_type)
    else:
        renderer_type = type(arena._renderer)
        context_type = type(arena._renderer._gl_context)
        driver = _gl_driver(arena._renderer._gl_context)
    return dict(mode=arena.render_mode, renderer=renderer_type.__qualname__, driver=driver,
                context=context_type.__module__, source=_source_versions(context_type, *_types(renderer_type)))


def _commission_key(wiring, rig, schedule, horizon):
    arena = wiring.arena
    if arena.render_mode == 'mujoco':
        arena._apply_camera_rig()
    model = np.empty(mujoco.mj_sizeModel(arena.model), dtype=np.uint8)
    mujoco.mj_saveModel(arena.model, None, model)
    manifest = dict(
        format=1, platform=platform.machine(), mujoco=mujoco.__version__, numpy=np.__version__,
        source=_source_versions(tt3_arena, tt4_camera, tt4_hall, tt6_body_probe,
                                *_types(type(arena)), *_types(type(wiring)), *_types(type(wiring.wiring))),
        model=sha256(model).hexdigest(), body=arena.body, rig=[astuple(eye) for eye in rig],
        motor_rates=arena._joint_increments.tolist(), distractor=arena.distractor, view_seed=arena.view_seed,
        frameskip=arena.frameskip, max_steps=arena.max_steps, renderer=_renderer_version(arena),
        wiring=wiring.wiring.state(), delays=wiring.disturbance.delays, noise=wiring.disturbance.noise,
        wiring_seed=wiring.seed, calibration=schedule['calibration'], player_seed=schedule['player_seed'],
        horizon=horizon,
    )
    return sha256(json.dumps(manifest, sort_keys=True).encode()).digest()


class ProbeCache:
    """Read and publish one trajectory at a time as the commissioner requests it."""

    def __init__(self, directory, wiring, rig, schedule, horizon):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = _commission_key(wiring, rig, schedule, horizon)
        self.shape = (len(rig), wiring.arena.height, wiring.arena.width, 3)

    def _load(self, path, command_count):
        try:
            with np.load(path, allow_pickle=False) as source:
                if source['key'].item() != path.stem:
                    return None
                frames = source['frames']
        except (OSError, ValueError, TypeError, EOFError, BadZipFile, KeyError):
            return None
        if (frames.dtype == np.uint8 and frames.shape[1:] == self.shape
                and 1 <= len(frames) <= command_count + 1):
            return frames
        return None

    def _replay(self, wiring, seed, commands, frames):
        # Keep the simulator state and command transport as in a live probe.
        replay = tt6_body_probe.collect_probe(wiring, seed, commands, render=False)
        if len(replay) != len(frames):
            return None
        arena = wiring.arena
        mujoco.mj_kinematics(arena.model, arena.data)
        if arena.render_mode == 'mujoco':
            mujoco.mj_camlight(arena.model, arena.data)
        return [(action, pixels) for (action, _), pixels in zip(replay, frames, strict=True)]

    def collect(self, wiring, seed, commands):
        digest = sha256(self.key)
        digest.update(int(seed).to_bytes(8, 'little'))
        digest.update(np.asarray(commands, dtype='<f8').tobytes())
        path = self.directory / f'{digest.hexdigest()}.npz'
        frames = self._load(path, len(commands))
        if frames is not None:
            history = self._replay(wiring, seed, commands, frames)
            if history is not None:
                return history

        with path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if frames is not None:
                return self._capture(path, wiring, seed, commands)
            frames = self._load(path, len(commands))
            if frames is not None:
                history = self._replay(wiring, seed, commands, frames)
                if history is not None:
                    return history
            return self._capture(path, wiring, seed, commands)

    def _capture(self, path, wiring, seed, commands):
        history = tt6_body_probe.collect_probe(wiring, seed, commands)
        with NamedTemporaryFile(dir=self.directory, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, key=path.stem, frames=np.stack([rgb for _, rgb in history]))
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return history

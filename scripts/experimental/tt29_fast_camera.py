"""Profile opt-in acceleration of the native RGB and near-side regulation loop."""

import argparse
from collections import Counter
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt29_action_memory import Experience, RecallPlanner
from .tt29_endurance import audit
from .tt29_partner import FOREHAND_LANES
from .tt29_partner_fast import FastPartnerArena, NativeBallFlight
from .tt29_rally_physics import OUTPUT, RallyArena, measured_launches, mujoco
from .tt29_vision import FlightPractice, OpticalFlight, Rig, Sighting, ball_pixels
from .tt29_vision_twin import OpticalPlanner


@lru_cache(maxsize=1)
def detector():
    work = TemporaryDirectory(prefix="tt29-camera-")
    output = Path(work.name) / "ball.so"
    subprocess.run(["c++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off",
                    str(Path(__file__).with_suffix(".cpp")), "-o", str(output)], check=True)
    function = ctypes.CDLL(str(output)).ball_component
    function.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t,
                         ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_void_p)
    function.restype = ctypes.c_bool
    return function, work


def native_ball_pixels(rgb, predicted=None):
    """Find the same orange component in ordinary, including strided, RGB."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        return ball_pixels(rgb, predicted)
    xy = None if predicted is None else np.ascontiguousarray(predicted, dtype=np.float64)
    if xy is not None and xy.shape != (2,):
        return ball_pixels(rgb, predicted)
    result = np.empty(5, dtype=np.float64)
    found = detector()[0](rgb.ctypes.data, *rgb.shape[:2], *rgb.strides,
                          None if xy is None else xy.ctypes.data, result.ctypes.data)
    if found:
        return Sighting(result[:2], float(result[2]), bool(result[3]), int(result[4]))
    return None


class StereoRenderer:
    """Submit two fixed cameras with one native scene build and RGB readback."""

    def __init__(self, arena):
        self.arena = arena
        model = arena.model
        window = model.vis.global_
        quality = model.vis.quality
        old = window.offwidth, window.offheight, quality.offsamples
        try:
            window.offwidth, window.offheight = 2 * arena.width, arena.height
            if arena.detail == "preview":
                quality.offsamples = 0
            self.renderer = mujoco.Renderer(model, height=arena.height, width=2 * arena.width)
        finally:
            window.offwidth, window.offheight, quality.offsamples = old
        self.cameras = []
        for stream in ("left", "right"):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = model.camera(CAMERAS[stream][0]).id
            self.cameras.append(camera)

    def _prepare(self, category):
        arena = self.arena
        arena._color_roles()
        mujoco.mj_kinematics(arena.model, arena.data)
        mujoco.mj_camlight(arena.model, arena.data)
        renderer = self.renderer
        scene = renderer.scene
        mujoco.mjv_updateScene(arena.model, arena.data, renderer._scene_option, None,
                               self.cameras[0], category, scene)
        left = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        mujoco.mjv_updateCamera(arena.model, arena.data, self.cameras[1], scene)
        right = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        for dest, source in zip(scene.camera, (left, right), strict=True):
            for attr in ("pos", "forward", "up"):
                getattr(dest, attr)[:] = getattr(source, attr)
            for attr in ("frustum_bottom", "frustum_center", "frustum_far", "frustum_near",
                         "frustum_top", "frustum_width", "orthographic"):
                setattr(dest, attr, getattr(source, attr))
        scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    def _split(self, frames):
        return {"left": frames[:, :self.arena.width], "right": frames[:, self.arena.width:]}

    def capture(self):
        self._prepare(mujoco.mjtCatBit.mjCAT_ALL)
        return self._split(self.renderer.render())

    def close(self):
        self.renderer.close()


class CachedStereoRenderer(StereoRenderer):
    """Composite ordinary RGB for a fixed station using cached static depth."""

    def __init__(self, arena):
        super().__init__(arena)
        shape = (arena.height, arena.width * 2)
        self.background = np.empty((*shape, 3), dtype=np.uint8)
        self.background_depth = np.empty(shape, dtype=np.float32)
        self.foreground = np.empty_like(self.background)
        self.foreground_depth = np.empty_like(self.background_depth)
        self._layer(mujoco.mjtCatBit.mjCAT_STATIC, self.background, self.background_depth)

    def _layer(self, category, rgb, depth):
        renderer = self.renderer
        self._prepare(category)
        if renderer._gl_context:
            renderer._gl_context.make_current()
        mujoco.mjr_render(renderer._rect, renderer.scene, renderer._mjr_context)
        mujoco.mjr_readPixels(rgb, depth, renderer._rect, renderer._mjr_context)

    def capture(self):
        self._layer(mujoco.mjtCatBit.mjCAT_DYNAMIC, self.foreground, self.foreground_depth)
        frames = self.background.copy()
        near = self.foreground_depth > self.background_depth
        np.copyto(frames, self.foreground, where=near[..., None])
        if self.renderer._gl_context:
            frames = np.flipud(frames)
        return self._split(frames)


def _sighting_equal(a, b):
    if a is None or b is None:
        return a is b
    return (np.array_equal(a.center, b.center) and a.radius == b.radius
            and a.clipped == b.clipped and a.area == b.area)


def compare_held(arena, shots, starts, stereo):
    """Compare detectors on unseen traces, including edge cases and native camera pixels."""
    totals = Counter()
    render_error = [0., 0.]
    rng = np.random.default_rng(192925)
    commands = np.zeros(arena.action_size)
    for sid in starts:
        arena.launch(shots[sid])
        for tick in range(48):
            if tick % 2 == 0:
                reference = arena.render_cameras()
                paired = stereo.capture()
                for stream in arena.streams:
                    rgb = reference[stream]
                    other = paired[stream]
                    offset = rng.uniform((0, 0), (arena.width, arena.height))
                    for prediction in (None, offset):
                        prior = ball_pixels(rgb, prediction)
                        same_detector = native_ball_pixels(rgb, prediction)
                        replacement = native_ball_pixels(other, prediction)
                        totals["cases"] += 1
                        totals["detector_exact"] += _sighting_equal(prior, same_detector)
                        totals["full_path_exact"] += _sighting_equal(prior, replacement)
                        totals["full_path_frac_pixel"] += (
                            _sighting_equal(prior, replacement) or (
                                prior is not None and replacement is not None
                                and np.linalg.norm(prior.center - replacement.center) <= .30
                                and abs(prior.radius - replacement.radius) <= .35
                                and prior.clipped == replacement.clipped)
                        )
                    delta = np.abs(rgb.astype(np.int16) - other)
                    render_error[0] = max(render_error[0], float(delta.max()))
                    render_error[1] += float(delta.mean())
                    totals["images"] += 1
            _, _, done, truncated, _ = arena.exchange(commands, np.zeros(5))
            if done or truncated:
                break
    return dict({name: int(count) for name, count in totals.items()}, maximum_channel_error=render_error[0],
                average_channel_error=render_error[1] / totals["images"])


def play(arena, twin, checkpoint, shot, *, mode, limit, stereo):
    from . import tt29_vision as vision

    rig = Rig.from_arena(arena)
    arena.launch(shot)
    twin.reset(seed=shot["id"], server=-1, render=False)
    eye = OpticalFlight(rig, FlightPractice(), learn=True)
    memory = Experience()
    for example in checkpoint["checkpoint"]:
        memory.add(np.asarray(example["flight"]), np.asarray(example["targets"]), example["time_to_contact"])
    pupil = OpticalPlanner(twin, seed=290110)
    pupil.nominal = RecallPlanner(twin, memory, seed=290110)
    if mode != "baseline":
        pupil.nominal.flight = NativeBallFlight(twin)
    q = arena.joints.copy()
    requested = q.copy()
    lower, upper = twin._joint_low[pupil.actuators], twin._joint_high[pupil.actuators]
    detector_fn = ball_pixels if mode == "baseline" else native_ball_pixels
    render = stereo.capture if mode in {"paired", "cached"} else arena.render_cameras
    timing, actions = Counter(), []

    def timed(rgb, predicted):
        began = process_time()
        result = detector_fn(rgb, predicted)
        timing["ball_detection"] += process_time() - began
        return result

    original = vision.ball_pixels
    vision.ball_pixels = timed
    began_wall, began_cpu = perf_counter(), process_time()
    try:
        for step in range(arena.max_steps):
            now = step * arena.control_dt
            current = arena.joints.copy()
            began = process_time()
            frames = render()
            timing["render"] += process_time() - began
            began = process_time()
            eye.observe(now, frames, current)
            timing["vision"] += process_time() - began
            began = process_time()
            action = pupil.command(step, eye.latest, current, (current - q) / arena.control_dt, requested)
            timing["near"] += process_time() - began
            requested = np.clip(requested + action * pupil.nominal.rates * arena.control_dt, lower, upper)
            q = current
            began = process_time()
            _, _, ended, capped, info = arena.exchange(action, np.zeros(5))
            timing["physics_and_partner"] += process_time() - began
            actions.append(action.tolist())
            if any(e["kind"] == "legal_return" and e["side"] == 1 for e in info["events"]):
                pupil.nominal.remember()
            if ended or capped or arena.legal_returns + arena.rival_returns >= limit + 1:
                break
    finally:
        vision.ball_pixels = original
    elapsed_cpu = process_time() - began_cpu
    elapsed_wall = perf_counter() - began_wall
    simulated = float(arena.data.time)
    timing["vision_without_detection"] = timing.pop("vision") - timing["ball_detection"]
    return dict(id=shot["id"], result=arena.point_result(), score=audit(arena.events),
                event_sha256=digest_json(arena.events), actions_sha256=digest_json(actions),
                elapsed_cpu=elapsed_cpu, elapsed_wall=elapsed_wall, simulated_seconds=simulated,
                control_frames=len(actions), cpu_ms_per_simulated_second=1000*elapsed_cpu/simulated,
                wall_ms_per_simulated_second=1000*elapsed_wall/simulated,
                phase_cpu_ms_per_simulated_second={name: 1000*seconds/simulated for name, seconds in timing.items()})


def run(path, output, checkpoint_path, starts, limit, *, cached=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint, shots = _read(checkpoint_path), measured_launches(path)
    settings = dict(fixture="table-edge", strike_ready="lateral", width=384, height=288,
                    clearance_guard=True, fast_search=True, max_steps=625)
    arena = FastPartnerArena(**settings, targets=FOREHAND_LANES, guard_table=True)
    twin = RallyArena(**settings)
    arena.streams = ("left", "right")
    detector()  # Compile outside the measurements.
    stereo = CachedStereoRenderer(arena) if cached else StereoRenderer(arena)
    receipt = dict(launch_sha256=_hash(path), checkpoint_sha256=_hash(checkpoint_path),
                   partner="fast native", control_hz=60, size=(384, 288), limit=limit, rows=[])
    try:
        observations = [6971, 10831, 13160] if cached else [5906, 6971, 10831, 9339, 13160, 3233]
        receipt["sensor_check"] = compare_held(arena, shots, observations, stereo)
        _write(output, receipt)
        print("sensors", receipt["sensor_check"], flush=True)
        if receipt["sensor_check"]["detector_exact"] != receipt["sensor_check"]["cases"]:
            raise ValueError("Native detector changed a held RGB observation")
        for index, i in enumerate(starts):
            modes = ["baseline", "cached"] if cached else ["baseline", "native", "paired"]
            if index % 2:
                modes.reverse()
            row = dict(id=i)
            for mode in modes:
                row[mode] = play(arena, twin, checkpoint, shots[i], mode=mode, limit=limit, stereo=stereo)
                result = row[mode]
                print(i, mode, result["score"]["returns"], round(result["cpu_ms_per_simulated_second"], 1),
                      result["phase_cpu_ms_per_simulated_second"], flush=True)
            receipt["rows"].append(row)
            _write(output, receipt)
    finally:
        stereo.close()
        arena.close()
        twin.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("--checkpoint", type=Path, default=OUTPUT / "camera-action-memory.json.gz")
    p.add_argument("--output", type=Path, default=OUTPUT / "tt29_fast_camera.json.gz")
    p.add_argument("--ids", type=int, nargs="+", default=(7271, 7345))
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--cached", action="store_true")
    args = p.parse_args()
    run(args.feeds, args.output, args.checkpoint, args.ids, args.limit, cached=args.cached)

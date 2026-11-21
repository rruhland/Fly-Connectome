"""Evaluate RGB camera registration through played MuJoCo hall trajectories."""

import argparse
from collections import Counter
from colorsys import hsv_to_rgb
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt6_body_arena import body_xml
from .tt4_hall import hall_model
from .tt8_service_arena import TT8ServiceArena
from .tt18_view import CameraFit, ImageColors, LANDMARKS, OBJECT_HUES, ViewRegistry, object_pixels, triangulate_views


ORIGIN = np.array((0., 0., TABLE_HEIGHT))
CAMERA_SEEDS = (43712043, 50931851, 84103627, 72906167, 95139402, 59731470)
PATH_SEEDS = (28592071, 72460301, 89912505, 40712978, 92660716, 70267841)
PHASES = ((0, 'source'), (5, 'remounted'), (10, 'added'), (15, 'removed'),
          (20, 'monocular'), (24, 'restored_stereo'))


def _coordinates(values):
    return ' '.join(f'{value:.6f}' for value in values)


def add_view_landmarks(model):
    """Install surveyed calibration stands on the hall floor, independent of the table."""
    world, asset = model.find('worldbody'), model.find('asset')
    for index, marker in enumerate(LANDMARKS):
        position = np.asarray(marker.position) + ORIGIN
        x, y, z = position
        name = f'tt18_landmark_{index}'
        color = hsv_to_rgb(marker.hue, .98, .58)
        ET.SubElement(asset, 'material', name=name, rgba=_coordinates((*color, 1.)),
                      emission='.9', specular='0', shininess='0')
        ET.SubElement(world, 'geom', name=name, type='sphere', pos=_coordinates(position),
                      size='.033', material=name, contype='0', conaffinity='0')
        ET.SubElement(world, 'geom', name=f'{name}_stand', type='cylinder',
                      pos=_coordinates((x, y, z / 2)), size=_coordinates((.006, z / 2)),
                      rgba='.22 .24 .25 1', contype='0', conaffinity='0')


def configure_view_colors(arena):
    """Use visible orange ball and pink blade role colors throughout this assay."""
    colors = {'ball': hsv_to_rgb(OBJECT_HUES['ball'], .99, .96),
              'learner_racket': hsv_to_rgb(OBJECT_HUES['blade'], .97, .86),
              'rival_racket': (.30, .33, .35)}
    for name, value in colors.items():
        arena.model.geom(name).rgba[:3] = value


def add_spare_camera(model):
    ET.SubElement(model.find('worldbody'), 'camera', name='tt18_spare', pos='0 -3 2', fovy='65')


def _resize_table(model):
    factors = np.array((1.07, .94, 1.))
    world = model.find('worldbody')
    for geom in world.findall('geom'):
        levels = [float(value) for value in geom.get('pos', '0 0 0').split()]
        if geom.get('name') != 'table' and tuple(levels[2:]) not in ((.7605,), (.355,)):
            continue
        size = np.fromstring(geom.get('size'), sep=' ')
        geom.set('pos', _coordinates(factors * levels))
        geom.set('size', _coordinates(factors * size))


class ViewHall(TT8ServiceArena):
    """Native hall fixture with surveyed stands and arbitrary physical camera rigs."""

    def __init__(self, *args, resize=False, **kwargs):
        self.resize = resize
        super().__init__(*args, **kwargs)

    def _load_model(self):
        model = body_xml(self.body) if self.body != 'stock' else ET.parse(
            Path(__file__).with_name('tt4_arena.xml')).getroot()
        add_view_landmarks(model)
        add_spare_camera(model)
        if self.resize:
            _resize_table(model)
        return hall_model(model)

    def reset(self, *, seed=None, render=True, server=-1):
        super().reset(seed=seed, render=False, server=server)
        configure_view_colors(self)
        return self.render_cameras() if render else None


@dataclass(frozen=True)
class ViewMount:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float
    roll: float

    def optics(self, width, height):
        forward = np.subtract(self.target, self.eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0., 0., 1.))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = Rotation.from_rotvec((0., 0., self.roll)).as_matrix() @ np.array((right, down, forward))
        focal = height / (2. * np.tan(np.deg2rad(self.fovy) / 2.))
        return CameraFit(np.asarray(self.eye), rotation, focal, width, height)


def hall_camera_renderer(arena):
    if arena._renderer is None:
        native = arena.model.vis.global_
        width, height = native.offwidth, native.offheight
        try:
            native.offwidth, native.offheight = arena.width, arena.height
            arena._renderer = mujoco.Renderer(arena.model, height=arena.height, width=arena.width)
        finally:
            native.offwidth, native.offheight = width, height
    return arena._renderer


def hall_images(arena, mounts):
    """Render RGB only. Internal projections remain on the evaluator side."""
    for mount in mounts.values():
        model_camera = arena.model.camera(mount.name)
        camera = mount.optics(arena.width, arena.height)
        model_camera.pos[:] = camera.eye + ORIGIN
        right, down, forward = camera.rotation
        matrix = np.column_stack((right, -down, -forward))
        mujoco.mju_mat2Quat(model_camera.quat, np.ascontiguousarray(matrix).ravel())
        model_camera.fovy[0] = mount.fovy
    mujoco.mj_kinematics(arena.model, arena.data)
    mujoco.mj_camlight(arena.model, arena.data)
    renderer = hall_camera_renderer(arena)
    images = {}
    for key, mount in mounts.items():
        renderer.update_scene(arena.data, camera=mount.name)
        images[key] = renderer.render()
    return images


def rig_schedule(seed):
    rng = np.random.default_rng(seed)

    def mount(name, eye, target, fovy, roll):
        eye = tuple(np.asarray(eye) + rng.uniform(-.08, .08, 3))
        target = tuple(np.asarray(target) + rng.uniform(-.07, .07, 3))
        return ViewMount(name, eye, target, fovy + rng.uniform(-2., 2.), roll + rng.uniform(-.025, .025))

    source = {'left': mount('eye_left', (-.36, -2.65, 1.16), (-.08, -.01, .09), 66., 0.),
              'right': mount('eye_right', (.36, -2.63, 1.20), (.09, -.05, .08), 65., 0.)}
    moved = {'left': mount('eye_left', (-1.60, -2.44, 1.57), (-.02, -.14, .12), 64., .18),
             'right': mount('eye_right', (.95, -2.58, 1.10), (.02, .02, .13), 70., -.23)}
    spare = {'spare': mount('tt18_spare', (2.14, -.48, 2.40), (.01, -.30, .07), 68., .28)}
    restored_right = mount('eye_right', (.60, -2.75, 1.56), (.02, -.10, .15), 62., .10)
    return dict(source=source, remounted=moved, added={**moved, **spare}, removed={'left': moved['left'], **spare},
                monocular=spare, restored_stereo={'right': restored_right, **spare})


def _metrics(values, scale=1.):
    return None if not values else {'median': float(scale * np.median(values)),
                                    'p95': float(scale * np.quantile(values, .95)), 'n': len(values)}


def score_observations(fits, true_optics, pixels, positions):
    """Evaluator metrics using the same detected pixels for every camera method."""
    objects = {}
    for role, position in positions.items():
        estimate = triangulate_views(fits, pixels[role])
        expected = {key: view.project(position)[:2] for key, view in true_optics.items()}
        objects[role] = dict(
            position=position.tolist(),
            estimate=None if estimate is None else estimate.tolist(),
            world_error_m=None if estimate is None else float(np.linalg.norm(position - estimate)),
            calibration_px={key: float(np.linalg.norm(fit.project(position)[:2] - expected[key]))
                            for key, fit in fits.items() if key in pixels[role]},
            detection_px={key: float(np.linalg.norm(pixel - expected[key])) for key, pixel in pixels[role].items()},
        )
    return dict(views=list(fits), objects=objects)


def run_one(camera_seed, trajectory_seed, *, resize, width, height):
    arena = ViewHall(seed=trajectory_seed, body='rotary', venue='hall', width=width, height=height,
                    resize=resize, max_steps=32, frameskip=8)
    schedule = rig_schedule(camera_seed)
    registry = ViewRegistry()
    stale = None
    records = []
    costs = []
    rng = np.random.default_rng(trajectory_seed)
    amplitude = rng.uniform(.22, .53, 6)
    offset = rng.uniform(-np.pi, np.pi, 6)
    try:
        arena.reset(seed=trajectory_seed, render=False)
        for tick in range(arena.max_steps):
            phase = next(name for start, name in reversed(PHASES) if tick >= start)
            phase_start = next(start for start, name in PHASES if name == phase)
            mounts = schedule[phase]
            frames = {key: ImageColors(image) for key, image in hall_images(arena, mounts).items()}
            truth = {key: mount.optics(width, height) for key, mount in mounts.items()}
            start = perf_counter()
            observed = registry.observe(frames)
            costs.append(1000. * (perf_counter() - start))
            if stale is None:
                stale = observed.copy()
            current = {'registered': observed, 'stale': {key: fit for key, fit in stale.items() if key in frames},
                       'true_optics': truth}
            pixels = object_pixels(frames)
            world = {'ball': arena.ball.copy() - ORIGIN, 'blade': arena.paddle.copy() - ORIGIN}
            state = dict(tick=tick, phase=phase, phase_start=phase_start, events=registry.last_events.copy(),
                         images=list(frames), detections={key: list(value) for key, value in pixels.items()},
                         registration_ms=costs[-1],
                         active_fits={key: dict(rms_px=fit.rms, markers=fit.markers) for key, fit in observed.items()},
                         methods={method: score_observations(fits, truth, pixels, world)
                                  for method, fits in current.items()})
            if tick in (20, 21):
                fit = observed.get('spare')
                target = pixels['ball'].get('spare')
                if fit is not None and target is not None:
                    ray = fit.ray(target)
                    ray /= np.linalg.norm(ray)
                    separation = .35
                    near = fit.eye + 2.6 * ray
                    far = near + separation * ray
                    state['monocular_depth_ambiguity'] = dict(separation_m=separation,
                        reprojected_separation_px=float(np.linalg.norm((fit.project(near) - fit.project(far))[:2])))
            records.append(state)
            command = amplitude * np.sin(.18 * tick + offset)
            if arena.terminated or arena.truncated:
                break
            arena.step(command, render=False)
    finally:
        arena.close()
    return dict(camera_seed=camera_seed, trajectory_seed=trajectory_seed, table_resized=resize,
                records=records, registration_counts=registry.counts,
                cost_ms=_metrics(costs))


def summarize(courses):
    records = [record for course in courses for record in course['records']]
    summary = {}
    for _, phase in PHASES:
        phase_records = [record for record in records if record['phase'] == phase]
        result = {'frames': len(phase_records),
                  'stream_events': dict(Counter(event for record in phase_records for event in record['events'].values())),
                  'registry_wall_ms': _metrics([record['registration_ms'] for record in phase_records]),
                  'first_frame_wall_ms': _metrics([record['registration_ms'] for record in phase_records
                                                   if record['tick'] == record['phase_start']]),
                  'first_frame_registrations': sum(set(record['active_fits']) == set(record['images'])
                                                  for record in phase_records
                                                  if record['tick'] == record['phase_start'])}
        for method in ('registered', 'stale', 'true_optics'):
            result[method] = {}
            for role in OBJECT_HUES:
                rows = [record['methods'][method]['objects'][role] for record in phase_records]
                world = [row['world_error_m'] for row in rows if row['world_error_m'] is not None]
                reprojection = [value for row in rows for value in row['calibration_px'].values()]
                result[method][role] = dict(depth_frames=len(world), world_mm=_metrics(world, 1000.),
                                            calibration_px=_metrics(reprojection))
        summary[phase] = result
    return summary


def write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        data = json.dumps(receipt, separators=(',', ':'), allow_nan=False).encode()
        path.write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt18/view-registration.json.gz'))
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=192)
    args = parser.parse_args()
    courses = []
    for index, (camera_seed, trajectory_seed) in enumerate(zip(CAMERA_SEEDS, PATH_SEEDS, strict=True)):
        courses.append(run_one(camera_seed, trajectory_seed, resize=index % 2 == 1,
                               width=args.width, height=args.height))
        print(f'course {index + 1}: {courses[-1]["registration_counts"]}', flush=True)
    result = dict(schema='tt18-rgb-view-registration-v1', resolution=[args.width, args.height],
                  nominal_table_width_m=1.525, nominal_table_length_m=2.74,
                  alternative_table_width_m=1.525 * 1.07, alternative_table_length_m=2.74 * .94,
                  renderer=f'MuJoCo {mujoco.__version__} native hall', calibration_motor_commands=0,
                  calibration_landmark_map=[dict(position=mark.position, hue=mark.hue) for mark in LANDMARKS],
                  summary=summarize(courses), courses=courses)
    write_receipt(args.output, result)


if __name__ == '__main__':
    main()

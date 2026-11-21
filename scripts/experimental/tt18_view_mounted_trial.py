"""Measure RGB re-registration of the payload camera as the physical arm moves."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .tt18_plant import MOUNTED_CAMERA, PLANTS, TT18PlantArena
from .tt18_view import CameraFit, ImageColors, OBJECT_HUES, ViewRegistry, object_pixels
from .tt18_view_trial import (ORIGIN, _metrics, add_view_landmarks, configure_view_colors,
                            hall_images, rig_schedule, score_observations, write_receipt)


COURSES = {'compact': (65483197, 26438173, 82094301), 'extended': (35409283, 10978376, 60127499)}


class MountedLandmarkHall(TT18PlantArena):
    def _model_document(self):
        model = super()._model_document()
        add_view_landmarks(model)
        return model

    def reset(self, *, seed=None, render=True, server=-1):
        super().reset(seed=seed, render=False, server=server)
        configure_view_colors(self)
        return self.render_cameras() if render else None


def evaluator_mounted_camera(arena):
    camera = arena.model.camera(MOUNTED_CAMERA)
    data = arena.data.camera(MOUNTED_CAMERA)
    center = data.xpos.copy() - ORIGIN
    rotation = np.diag((1., -1., -1.)) @ data.xmat.reshape(3, 3).T
    focal = arena.height / (2. * np.tan(np.deg2rad(camera.fovy[0]) / 2.))
    return CameraFit(center, rotation, focal, arena.width, arena.height)


def run_one(plant, seed, width, height):
    ticks = 28
    arena = MountedLandmarkHall(seed=seed, plant=PLANTS[plant], body='rotary', venue='hall',
                               width=width, height=height, frameskip=8, max_steps=ticks)
    court = {'court': rig_schedule(seed)['remounted']['left']}
    registry = ViewRegistry()
    initial_fits = None
    initial_mount = None
    observations = []
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(.15, .30, arena.action_size)
    initial_phase = rng.uniform(-np.pi, np.pi, arena.action_size)
    try:
        for tick in range(ticks):
            images = hall_images(arena, court)
            images['mount'] = arena.render_mounted_camera()
            frames = {key: ImageColors(rgb) for key, rgb in images.items()}
            start = perf_counter()
            live = registry.observe(frames)
            cost_ms = 1000. * (perf_counter() - start)
            real = {'court': court['court'].optics(width, height), 'mount': evaluator_mounted_camera(arena)}
            if initial_fits is None:
                initial_fits, initial_mount = live.copy(), real['mount']
            stale = {key: camera for key, camera in initial_fits.items() if key in frames}
            pixels = object_pixels(frames)
            positions = {'ball': arena.ball.copy() - ORIGIN, 'blade': arena.paddle.copy() - ORIGIN}
            relative = Rotation.from_matrix(real['mount'].rotation @ initial_mount.rotation.T)
            fit = live.get('mount')
            observations.append(dict(
                tick=tick,
                registration_ms=cost_ms,
                mount_event=registry.last_events['mount'],
                mount_landmarks=None if fit is None else fit.markers,
                mount_eye_error_m=None if fit is None else float(np.linalg.norm(real['mount'].eye - fit.eye)),
                mount_translation_m=float(np.linalg.norm(real['mount'].eye - initial_mount.eye)),
                mount_rotation_deg=float(np.rad2deg(relative.magnitude())),
                methods={name: score_observations(views, real, pixels, positions)
                         for name, views in (('registered', live), ('stale', stale), ('true_optics', real))},
            ))
            if arena.terminated or arena.truncated:
                break
            command = amplitude * np.sin(.47 * tick + initial_phase)
            arena.step(command, render=False)
    finally:
        arena.close()
    return dict(plant=plant, seed=seed, settings=asdict(PLANTS[plant]),
                registration_counts=registry.counts, observations=observations)


def summarize(courses):
    summary = {}
    for name in COURSES:
        courses_of_plant = [course for course in courses if course['plant'] == name]
        records = [row for course in courses_of_plant for row in course['observations']]
        result = dict(frames=len(records), mounted_registered=sum(row['mount_landmarks'] is not None for row in records),
                      mounted_translation_max_m=max(row['mount_translation_m'] for row in records),
                      mounted_rotation_max_deg=max(row['mount_rotation_deg'] for row in records),
                      ledger_updates=sum(course['registration_counts']['relocated'] for course in courses_of_plant),
                      mounted_eye_mm=_metrics([row['mount_eye_error_m'] for row in records
                                               if row['mount_eye_error_m'] is not None], 1000.),
                      registry_wall_ms=_metrics([row['registration_ms'] for row in records]))
        for method in ('registered', 'stale', 'true_optics'):
            result[method] = {}
            for role in OBJECT_HUES:
                rows = [row['methods'][method]['objects'][role] for row in records]
                world = [row['world_error_m'] for row in rows if row['world_error_m'] is not None]
                reprojection = [row['calibration_px']['mount'] for row in rows if 'mount' in row['calibration_px']]
                visible = sum('mount' in row['detection_px'] for row in rows)
                result[method][role] = dict(depth_frames=len(world), world_mm=_metrics(world, 1000.),
                                            mount_detection_frames=visible, mount_calibration_px=_metrics(reprojection))
        summary[name] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tt18/view-mounted-registration.json.gz'))
    args = parser.parse_args()
    width, height = 256, 192
    courses = []
    for plant, seeds in COURSES.items():
        for seed in seeds:
            courses.append(run_one(plant, seed, width, height))
            print(plant, seed, courses[-1]['registration_counts'], flush=True)
    write_receipt(args.output, dict(schema='tt18-moving-rgb-camera-v1', resolution=[width, height],
                                   calibration_motor_commands=0, summary=summarize(courses), courses=courses))


if __name__ == '__main__':
    main()

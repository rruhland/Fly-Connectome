"""Score natural monocular registration and range on native unmarked RGB."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_camera import PerspectiveCamera
from .tt4_transfer_trial import ROOT, _write
from .tt18_view import CameraFit
from .tt19_so101 import WRIST_CAMERA
from .tt19_view_trial import CAMERAS
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport


SETTINGS = dict(venue='hall', marked=False, strict_endline=True, width=480, height=360,
                view_mode='wrist+fixed', detail='preview', max_steps=56)
SEEDS = (80703731, 213527479, 107493337, 393038593)
ORIGIN = np.array((0., 0., TABLE_HEIGHT))
COURT_POINTS = np.array(((-.424, -.80, 0.), (.424, -.80, 0.), (-.424, .80, 0.), (.424, .80, 0.),
                        (0., 0., .15), (-.31, -.45, .32), (.29, -.65, .41)))
SWITCH = 16


def _camera_truth(arena, stream):
    name = WRIST_CAMERA if stream == 'wrist' else CAMERAS[stream][0]
    model = arena.data.camera(name)
    right, up, backward = model.xmat.reshape(3, 3).T
    return CameraFit(model.xpos.copy() - ORIGIN, np.array((right, -up, -backward)),
                     arena.lens_intrinsics()[stream]['focal'], arena.width, arena.height)


def _move_outside(arena, index):
    name, eye, target, fovy = CAMERAS['left']
    shift = (.33, .19, .16) if index % 2 else (-.23, .28, -.12)
    camera = PerspectiveCamera(np.asarray(eye) + shift, np.asarray(target) + (.038, -.05, .01),
                               arena.width, arena.height, fovy)
    right, up, forward = camera.basis
    model = arena.model.camera(name)
    model.pos[:] = np.asarray(camera.eye) + ORIGIN
    axes = np.column_stack((right, up, -forward))
    mujoco.mju_mat2Quat(model.quat, np.ascontiguousarray(axes).ravel())


def _measurement(tracker, stream, truth_camera, truth_ball, truth_blade):
    fit = tracker.fits.get(stream)
    if fit is None:
        camera = None
    else:
        displacement = np.linalg.norm(fit.eye - truth_camera.eye)
        projection = fit.project(COURT_POINTS)[:, :2] - truth_camera.project(COURT_POINTS)[:, :2]
        camera = dict(eye_mm=float(1000. * displacement), table_px=float(np.median(np.linalg.norm(projection, axis=1))))
    return dict(camera=camera, detected=stream in tracker.detections['ball'],
                ball_mm=None if tracker.ball is None else float(1000. * np.linalg.norm(tracker.ball.position - truth_ball)),
                raw_mm=None if tracker.raw_ball is None else float(1000. * np.linalg.norm(tracker.raw_ball - truth_ball)),
                blade_mm=None if tracker.actor is None else float(1000. * np.linalg.norm(tracker.actor.position - truth_blade)),
                ball_source=tracker.sources.get('ball_range'), event=tracker.transport.registry.last_events[stream])


def play(seed, index):
    arena = TT20Arena(**SETTINGS)
    lenses = arena.lens_intrinsics()
    wrist = NaturalViewTransport('wrist', lens_fov=lenses).tracker()
    outside = NaturalViewTransport('outside', lens_fov=lenses).tracker()
    alternating = NaturalViewTransport('wrist+fixed', lens_fov=lenses).tracker()
    qualified = NaturalViewTransport('wrist', lens_fov=lenses, continuous_wrist=True).tracker()
    first, second = ('wrist', 'left') if index % 2 == 0 else ('left', 'wrist')
    random = np.random.default_rng(seed)
    phases = random.uniform(0., 2. * np.pi, size=5)
    strengths = random.uniform(.08, .19, size=5)
    rows = []
    try:
        arena.reset(seed=seed, server=1 if index % 2 else -1, render=False)
        for tick in range(arena.max_steps):
            if tick == SWITCH:
                _move_outside(arena, index)
            images = arena.render_cameras()
            active = first if tick < SWITCH or tick >= 2 * SWITCH else second
            truth_ball, truth_blade = arena.ball.copy() - ORIGIN, arena.paddle.copy() - ORIGIN
            records = {}
            for name, stream, tracker in (('wrist', 'wrist', wrist), ('outside', 'left', outside),
                                          ('wrist_qualified', 'wrist', qualified),
                                          ('alternating', active, alternating)):
                tracker.update({stream: images[stream]})
                records[name] = dict(stream=stream, **_measurement(
                    tracker, stream, _camera_truth(arena, stream), truth_ball, truth_blade))
            action = np.r_[strengths * np.sin(.24 * tick + phases), 0.]
            rows.append(dict(tick=tick, records=records, action=action.tolist()))
            _, _, ended, capped, _ = arena.step(action, render=False)
            if ended or capped:
                break
    finally:
        arena.close()
    return dict(seed=seed, server=1 if index % 2 else -1, sensor_order=[first, second, first], frames=rows)


def _statistics(values):
    if not values:
        return None
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95.)), n=len(values))


def summarize(rows):
    return dict(frames=len(rows), detections=sum(row['detected'] for row in rows),
                camera_frames=sum(row['camera'] is not None for row in rows),
                camera_within_50mm_6px=sum(row['camera'] is not None and row['camera']['eye_mm'] < 50.
                                          and row['camera']['table_px'] < 6. for row in rows),
                eye_mm=_statistics([row['camera']['eye_mm'] for row in rows if row['camera'] is not None]),
                court_reprojection_px=_statistics([row['camera']['table_px'] for row in rows if row['camera'] is not None]),
                ball_mm=_statistics([row['ball_mm'] for row in rows if row['ball_mm'] is not None]),
                raw_mm=_statistics([row['raw_mm'] for row in rows if row['raw_mm'] is not None]),
                blade_mm=_statistics([row['blade_mm'] for row in rows if row['blade_mm'] is not None]),
                sources=Counter(row['ball_source'] or 'abstain' for row in rows))


def run(output):
    trials = []
    for index, seed in enumerate(SEEDS):
        trials.append(play(seed, index))
        print(seed, {key: summarize([row['records'][key] for row in trials[-1]['frames']])
                     for key in ('wrist', 'outside')}, flush=True)
    flat = [row for trial in trials for row in trial['frames']]
    stages = {'wrist': [r['records']['wrist'] for r in flat],
              'wrist_qualified': [r['records']['wrist_qualified'] for r in flat],
              'outside_original': [r['records']['outside'] for r in flat if r['tick'] < SWITCH],
              'outside_moved': [r['records']['outside'] for r in flat if r['tick'] >= SWITCH],
              'left_then_wrist': [r['records']['alternating'] for tr in trials if tr['sensor_order'][0] == 'left'
                                  for r in tr['frames'] if SWITCH <= r['tick'] < 2 * SWITCH],
              'wrist_then_left': [r['records']['alternating'] for tr in trials if tr['sensor_order'][0] == 'wrist'
                                  for r in tr['frames'] if SWITCH <= r['tick'] < 2 * SWITCH]}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, dict(schema='tt20-natural-rgb-v1', arena=SETTINGS, trials=trials, camera_moved_at=SWITCH,
                        stages={key: summarize(rows) for key, rows in stages.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tt20/natural-view.json.gz')
    args = parser.parse_args()
    run(args.output)

"""Replay archived TT20 motors through a native RGB post/arm-tag factorial."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt18_view import ImageColors
from .tt19_view import RIM_HUES, TIP_HUE, ViewTransport, _blob, _image_rim, _in_court
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport, _blade_shape, _size_ball
from .tt20_natural_trial import ORIGIN, _camera_truth


DATA = ROOT / 'artifacts/tt20'
SOURCE_TAPE = DATA / 'camera-carry-left-game-40582930-0.json.gz'
VARIANTS = ((False, False), (False, True), (True, False), (True, True))
EYES = {'wrist': 'wrist', 'outside': 'left'}
GENERIC_NORMAL = np.array((0., 1., 0.))


class TagReplayArena(TT20Arena):
    def __init__(self, *, posts, tags, **settings):
        self.posts = posts
        self.tags = tags
        super().__init__(**dict(settings, marked=posts or tags), view_mode='wrist+fixed')
        self.streams = tuple(EYES.values())

    def _model_document(self):
        document = super()._model_document()
        for parent in document.iter():
            for item in list(parent):
                name = item.get('name', '')
                if ((not self.posts and name.startswith('tt18_landmark_'))
                        or (not self.tags and name.startswith(('tt19_rim_', 'tt19_tip_mark')))):
                    parent.remove(item)
        return document


def _vector(value):
    return None if value is None else np.asarray(value).tolist()


def _facing(value):
    return value if value[1] >= 0. else -value


def _tag_pose(camera, marks):
    if camera is None or any(mark is None for mark in marks):
        return None
    pose = _image_rim(camera, np.stack([mark.center for mark in marks]), refine=True)
    if pose is None or not _in_court('blade', pose[0]):
        return None
    return pose[0], _facing(pose[1])


def _measure(rgb, stream, natural, survey, tick):
    natural.update({stream: rgb}, tick=tick)
    camera = natural.fits.get(stream)
    color = ImageColors(rgb)
    survey_camera = None
    if survey is not None:
        survey_camera = survey.observe({stream: color}, tick, survey.registry.active).get(stream)
    blob = natural.detections['blade'].get(stream)
    anchor = None if blob is None else blob.center
    marks = [_blob(color, hue, tolerance=.016, minimum=85., span=max(10, rgb.shape[0] // 4), around=anchor)
             for hue in RIM_HUES]
    tip = _blob(color, TIP_HUE, tolerance=.016, minimum=82., span=max(10, rgb.shape[0] // 3), around=anchor)
    pose = _tag_pose(camera, marks)
    survey_pose = _tag_pose(survey_camera, marks)
    shape = None if camera is None or blob is None else _blade_shape(camera, blob)
    if shape is not None and not _in_court('blade', shape[0]):
        shape = None
    survey_ball_blob = natural.detections['ball'].get(stream)
    survey_ball = None
    if survey_camera is not None and survey_ball_blob is not None:
        estimate = _size_ball(survey_camera, survey_ball_blob)
        if estimate is not None and _in_court('ball', estimate[0]):
            survey_ball = estimate[0]
    return dict(eye=_vector(None if camera is None else camera.eye),
                survey_eye=_vector(None if survey_camera is None else survey_camera.eye),
                ball=_vector(None if natural.ball is None else natural.ball.position),
                survey_ball=_vector(survey_ball),
                ball_pixels=survey_ball_blob is not None,
                silhouette=_vector(None if shape is None else shape[0]),
                normals=_vector(None if shape is None else shape[1]),
                rim=_vector(None if pose is None else pose[0]),
                rim_normal=_vector(None if pose is None else pose[1]),
                survey_rim=_vector(None if survey_pose is None else survey_pose[0]),
                survey_rim_normal=_vector(None if survey_pose is None else survey_pose[1]),
                rim_detections=sum(mark is not None for mark in marks), tip_detection=tip is not None)


def _points(source):
    # First two occurrences of each role, selected by receipt order only.
    groups = {role: [row for row in source['points'] if row['role'] == role][:2] for role in (1, -1)}
    return [(name, row) for occurrence, name in enumerate(('teacher', 'held'))
            for group in groups.values() for row in group[occurrence:occurrence + 1]]


def _replay(variant, settings, inputs):
    posts, tags = variant
    arena = TagReplayArena(posts=posts, tags=tags, **settings)
    lenses = arena.lens_intrinsics()
    if arena.role_palette != 'natural':
        raise ValueError('the factorial requires the same natural paint in all four scenes')
    rows = []
    try:
        for split, original in inputs:
            seed, server = original['seed'], original['role']
            arena.reset(seed=seed, render=False, server=server)
            natural = {eye: NaturalViewTransport(eye, lens_fov=lenses, flight=False, continuous_wrist=True).tracker()
                       for eye in EYES}
            survey = {eye: ViewTransport('wrist' if eye == 'wrist' else 'fixed', wrist_fovy=lenses['wrist']['fovy'])
                      for eye in EYES} if posts else {}
            physics = sha256()
            pixels = {eye: sha256() for eye in EYES}
            frames = []
            for tick, action in enumerate(original['actions']):
                images = arena.render_cameras()
                estimates = {}
                for eye, stream in EYES.items():
                    pixels[eye].update(images[stream])
                    estimates[eye] = _measure(images[stream], stream, natural[eye], survey.get(eye), tick)
                # All evaluator state stays below this line and outside calibration fitting.
                normal = arena.data.geom('learner_racket').xmat.reshape(3, 3)[:, 2]
                frames.append(dict(tick=tick, views=estimates,
                                   truth=dict(blade=_vector(arena.paddle - ORIGIN), normal=_vector(_facing(normal)),
                                              ball=_vector(arena.ball - ORIGIN),
                                              eyes={eye: _vector(_camera_truth(arena, stream).eye)
                                                    for eye, stream in EYES.items()})))
                arena.step(action, render=False)
                physics.update(np.r_[arena.data.qpos, arena.data.qvel].astype('<f8').tobytes())
            if arena.point_result() != original['outcome'] or json.dumps(arena.events) != json.dumps(original['events']):
                raise ValueError(f'archived point {seed} changed under {variant}')
            rows.append(dict(seed=seed, server=server, split=split, frames=frames,
                             physics_sha256=physics.hexdigest(),
                             rgb_sha256={eye: h.hexdigest() for eye, h in pixels.items()},
                             actions_sha256=sha256(np.asarray(original['actions'], dtype='<f8').tobytes()).hexdigest(),
                             result=arena.point_result()))
            print(variant, split, seed, len(frames), flush=True)
    finally:
        arena.close()
    return rows


def _distance(first, second):
    return 1000. * np.linalg.norm(np.subtract(first, second))


def _angle(first, second):
    return np.degrees(np.arccos(np.clip(abs(np.dot(first, second)), 0., 1.)))


def _pick_normal(normals, reference):
    return max(normals, key=lambda candidate: abs(np.dot(candidate, reference)))


def _summary(values):
    return dict(count=len(values), median=round(float(np.median(values)), 3) if values else None,
                p90=round(float(np.quantile(values, .9)), 3) if values else None)


def _metrics(points, eye, *, tags):
    counts = dict(frames=0, ball_pixels=0, tip_pixels=0, four_rim_candidates=0)
    fields = ('eye_mm', 'survey_eye_mm', 'ball_mm', 'survey_ball_mm', 'silhouette_mm', 'rim_mm', 'rim_degrees',
              'survey_rim_mm', 'survey_rim_degrees', 'fallback_mm', 'fresh_ellipse_degrees',
              'paired_silhouette_mm', 'paired_rim_mm', 'paired_ellipse_degrees', 'paired_rim_degrees')
    errors = {field: [] for field in fields}
    for point in points:
        for frame in point['frames']:
            row, truth = frame['views'][eye], frame['truth']
            counts['frames'] += 1
            counts['ball_pixels'] += row['ball_pixels']
            counts['tip_pixels'] += row['tip_detection']
            counts['four_rim_candidates'] += row['rim_detections'] == 4
            for name, target, metric, key in (
                ('eye', truth['eyes'][eye], _distance, 'eye_mm'),
                ('survey_eye', truth['eyes'][eye], _distance, 'survey_eye_mm'),
                ('ball', truth['ball'], _distance, 'ball_mm'),
                ('survey_ball', truth['ball'], _distance, 'survey_ball_mm'),
                ('silhouette', truth['blade'], _distance, 'silhouette_mm'),
                ('rim', truth['blade'], _distance, 'rim_mm'),
                ('rim_normal', truth['normal'], _angle, 'rim_degrees'),
                ('survey_rim', truth['blade'], _distance, 'survey_rim_mm'),
                ('survey_rim_normal', truth['normal'], _angle, 'survey_rim_degrees'),
            ):
                if row[name] is not None:
                    errors[key].append(metric(row[name], target))
            best = row['rim'] if tags and row['rim'] is not None else row['silhouette']
            if best is not None:
                errors['fallback_mm'].append(_distance(best, truth['blade']))
            if row['normals'] is not None:
                guess = _pick_normal(row['normals'], GENERIC_NORMAL)
                errors['fresh_ellipse_degrees'].append(_angle(guess, truth['normal']))
                if row['rim'] is not None:
                    errors['paired_silhouette_mm'].append(_distance(row['silhouette'], truth['blade']))
                    errors['paired_rim_mm'].append(_distance(row['rim'], truth['blade']))
                    errors['paired_ellipse_degrees'].append(_angle(guess, truth['normal']))
                    errors['paired_rim_degrees'].append(_angle(row['rim_normal'], truth['normal']))
    return dict(**counts, **{key: _summary(values) for key, values in errors.items()})


def _learn(teacher_rows, eye):
    """Read the metric rim teacher and the painted ellipse in the same tagged RGB."""
    tagged, scale, seeds = [], [], []
    for point in teacher_rows:
        if point['split'] != 'teacher':
            continue
        for frame in point['frames']:
            row = frame['views'][eye]
            if row['rim'] is None or row['silhouette'] is None:
                continue
            tag_distance = np.linalg.norm(np.subtract(row['rim'], row['eye']))
            simple_distance = np.linalg.norm(np.subtract(row['silhouette'], row['eye']))
            if not (.85 <= tag_distance / simple_distance <= 1.15):
                continue
            scale.append(tag_distance / simple_distance)
            tagged.append(row['rim_normal'])
            seeds.append(point['seed'])
    if len(tagged) < 5 or len(set(seeds)) < 2:
        return dict(samples=len(tagged), source_seeds=sorted(set(seeds)), scale=None, normal=None)
    normal = np.mean(tagged, axis=0)
    return dict(samples=len(tagged), source_seeds=sorted(set(seeds)), scale=float(np.median(scale)),
                normal=_vector(normal / np.linalg.norm(normal)))


def _transfer(calibration, clean, eye):
    errors = {key: [] for key in ('fresh_mm', 'carried_mm', 'fresh_degrees', 'carried_degrees', 'oracle_branch_degrees')}
    evaluation = [row for row in clean if row['split'] == 'held']
    for point in evaluation:
        for frame in point['frames']:
            row, truth = frame['views'][eye], frame['truth']
            if row['silhouette'] is None:
                continue
            errors['fresh_mm'].append(_distance(row['silhouette'], truth['blade']))
            errors['fresh_degrees'].append(_angle(_pick_normal(row['normals'], GENERIC_NORMAL), truth['normal']))
            errors['oracle_branch_degrees'].append(min(_angle(n, truth['normal']) for n in row['normals']))
            if calibration['scale'] is not None:
                position = np.asarray(row['eye']) + calibration['scale'] * np.subtract(row['silhouette'], row['eye'])
                errors['carried_mm'].append(_distance(position, truth['blade']))
                selected = _pick_normal(row['normals'], calibration['normal'])
                errors['carried_degrees'].append(_angle(selected, truth['normal']))
    return dict(evaluation_seeds=[row['seed'] for row in evaluation], calibration=calibration,
                **{key: _summary(values) for key, values in errors.items()})


def run(output, workers):
    source = _read(SOURCE_TAPE)
    points = _points(source)
    settings = source['arena']
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {variant: pool.submit(_replay, variant, settings, points) for variant in VARIANTS}
        groups = {variant: job.result() for variant, job in futures.items()}
    scenarios, metrics = {}, {}
    for variant, rows in groups.items():
        baseline = groups[False, False]
        if [row['physics_sha256'] for row in rows] != [row['physics_sha256'] for row in baseline]:
            raise ValueError(f'nonvisual physics change under {variant}')
        posts, tags = variant
        label = f'posts_{int(posts)}_tags_{int(tags)}'
        scenarios[label] = rows
        metrics[label] = {eye: _metrics(rows, eye, tags=tags) for eye in EYES}
    transfer = {eye: _transfer(_learn(groups[False, True], eye), groups[False, False], eye) for eye in EYES}
    implementation = {name: _hash(Path(__file__).with_name(name)) for name in (
        'tt20_arm_tags_trial.py', 'tt20_fixture.py', 'tt20_natural.py', 'tt20_natural_geometry.py',
        'tt19_view.py', 'tt19_view_trial.py')}
    _write(output, dict(schema='tt20-arm-tags-factorial-1', archived_actions=str(SOURCE_TAPE.relative_to(ROOT)),
                        source_sha256=_hash(SOURCE_TAPE), implementation=implementation,
                        settings=settings, scenarios=scenarios, strict_replays=len(points)*len(VARIANTS),
                        metrics=metrics, transfer=transfer))
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA / 'arm-tag-factorial.json.gz')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)

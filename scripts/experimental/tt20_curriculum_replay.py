"""Compare fresh and carried camera fits on the same native action tapes."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_trial import CAMERAS, WRIST_CAMERA
from .tt20_curriculum import FRAME, GAME_SEED, OUTPUT, _metric, _sighting, apparatus, perception, transport, world_measurements


MODES = ('marked', 'fresh', 'carried')


def _camera_errors(arena, tracker):
    errors = {}
    for stream, fit in getattr(tracker, 'fits', {}).items():
        name = WRIST_CAMERA if stream == 'wrist' else CAMERAS[stream][0]
        true_eye = arena.data.camera(name).xpos - FRAME
        errors[stream] = float(np.linalg.norm(np.asarray(fit.eye) - true_eye))
    return errors


def timed_tapes(source, max_points):
    points = source['points'][:max_points]
    if len(points) < max_points or not any(row['server'] == -1 for row in points) or not any(
            row['server'] == 1 for row in points):
        raise ValueError('the prechosen action tape prefix must cover both service roles')
    return points


def record(points, mode, fixed, *, adaptive=False):
    marked = mode == 'marked'
    arena = apparatus(marked, opponent=adaptive)
    registered = transport(arena, marked, fixed if mode == 'carried' else None)
    rows = []
    try:
        for old in points:
            arena.reset(seed=old['seed'], server=old['server'], render=False)
            observer = registered.tracker()
            tick_set = {r['tick'] for r in old['captures']}
            capture = []
            path = []
            for tick, action in enumerate(old['actions']):
                point = arena.ball.copy() - FRAME
                if not np.array_equal(point, old['world_ball'][tick]):
                    raise ValueError('the marked and unmarked geometry changed a paired trajectory')
                path.append(point.tolist())
                if tick in tick_set:
                    observer.update(arena.render_cameras())
                    row = world_measurements(arena, observer, tick, _sighting(observer.ball))
                    row['camera_error_m'] = _camera_errors(arena, observer)
                    row['survey_streams'] = list(getattr(registered, 'survey_streams', ()))
                    capture.append(row)
                arena.step(action, render=False)
            if (arena.point_result() != old['result']
                    or json.dumps(arena.events, sort_keys=True) != json.dumps(old['events'], sort_keys=True)):
                raise ValueError('the camera comparison changed a physical outcome')
            rgb = observer.digest.hexdigest()
            if mode == 'fresh' and rgb != old['rgb']:
                raise ValueError('fresh replay rendered different native RGB at the archived cadence')
            rows.append(dict(seed=old['seed'], server=old['server'], action_sha256=old['action_sha256'],
                             captures=capture, world_ball=path, events=old['events'], rgb=rgb))
            print(mode, old['seed'], len(capture), sum('ball_error_m' in sample for sample in capture), flush=True)
    finally:
        arena.close()
    return rows


def summarize(courses):
    result = {}
    for mode, rows in courses.items():
        values = perception(rows)
        samples = [sample for row in rows for sample in row['captures']]
        values['camera_eye'] = {stream: _metric([sample['camera_error_m'][stream] for sample in samples
                                                  if stream in sample['camera_error_m']])
                                for stream in ('wrist', 'left', 'right')}
        values['frames'] = len(samples)
        values['survey_frames'] = {stream: sum(stream in sample['survey_streams'] for sample in samples)
                                   for stream in ('wrist', 'left', 'right')}
        values['first_ball_tick'] = [next((item['tick'] for item in row['captures'] if item['ball'] is not None), None)
                                     for row in rows]
        result[mode] = values
    return result


def run(source, practice, output, limit=4):
    game = _read(source)
    control = _read(practice)
    points = timed_tapes(game, limit)
    fixed = {key: value for key, value in control['registration']['marked'].items() if key in ('left','right')}
    if set(fixed) != {'left','right'}:
        raise ValueError('both fixed views must have been fitted during marked practice')
    courses = {mode: record(points, mode, fixed, adaptive=game['opponent']=='adaptive') for mode in MODES}
    receipt = dict(schema='tt20-curriculum-paired-perception-1', source=source.name, source_sha256=_hash(source),
                   calibration=practice.name, calibration_sha256=_hash(practice), points_compared=len(points),
                   native_rgb_replayed=len(points),
                   fixed_start=fixed, modes=courses, summary=summarize(courses))
    _write(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT/f'curriculum-frozen-{GAME_SEED}.json.gz')
    parser.add_argument('--practice', type=Path, default=OUTPUT/'curriculum-fade-practice.json.gz')
    parser.add_argument('--output', type=Path, default=OUTPUT/'curriculum-paired-perception.json.gz')
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source, arguments.practice, arguments.output)['summary'], indent=2))

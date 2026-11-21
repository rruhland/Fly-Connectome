"""Score unlabelled grayscale motor-probe candidates on disjoint native RGB replays."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt3_visual import RoleModel
from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt20_natural_trial import ORIGIN, _camera_truth
from .tt22_anonymous import CandidateFrames, pixel
from .tt22_anonymous_babble import BabblingEye, gray, patch_bank
from .tt22_anonymous_trial import ROLES, SOURCE, TOLERANCE
from .tt22_opponent_trial import arena_for
from .tt22_relation_trial import library


TARGET = SOURCE.parents[1] / 'tt22/relation-urgency.json.gz'
OUTPUT = TARGET.with_name('anonymous-babbling.json.gz')
PROBE_SEED = 436991627
TIMES = (4, 8)
AMPLITUDE = .30
STREAMS = ('left', 'right')


def probe(arena):
    responses = {key: [] for key in STREAMS}
    hashes = {}
    arena.reset(seed=PROBE_SEED, server=-1, render=False)
    ready = {key: gray(frame) for key, frame in arena.render_cameras().items()}
    for axis in range(arena.action_size):
        trajectories = []
        for direction in (-1, 1):
            arena.reset(seed=PROBE_SEED, server=-1, render=False)
            action = np.zeros(arena.action_size)
            action[axis] = direction * AMPLITUDE
            images = {key: [] for key in STREAMS}
            digest = sha256()
            for tick in range(1, TIMES[-1] + 1):
                _, _, done, capped, _ = arena.step(action, render=False)
                if done or capped:
                    raise ValueError('motor probe ended the physical point too early')
                if tick in TIMES:
                    for key, frame in arena.render_cameras().items():
                        digest.update(key.encode())
                        digest.update(frame)
                        images[key].append(gray(frame))
            hashes[f'{axis}:{direction}'] = digest.hexdigest()
            trajectories.append(images)
        for key in STREAMS:
            first, second = (np.array(trial[key]) for trial in trajectories)
            responses[key].append(np.max(np.abs(second - first), axis=0))
    return {key: patch_bank(ready[key], np.array(value)) for key, value in responses.items()}, hashes


def render_labels(arena, stream):
    renderer = arena._view_renderer()
    renderer.enable_segmentation_rendering()
    try:
        labels = arena._render_view(renderer, arena.data, CAMERAS[stream][0])
    finally:
        renderer.disable_segmentation_rendering()
    return labels


def selected_part(labels, arena, center):
    x, y = np.rint(center).astype(int)
    left, right = max(0, x-5), min(labels.shape[1], x+6)
    top, bottom = max(0, y-5), min(labels.shape[0], y+6)
    values = labels[top:bottom, left:right]
    rows, columns = np.nonzero(values[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    if not len(rows):
        return None
    # Annotate the geometry nearest the selected point and the nearest learner
    # part in the same local patch. Neither label enters proposal generation.
    index = np.argsort((rows + top - y)**2 + (columns + left - x)**2)
    names = []
    for identity in np.unique(values[rows[index], columns[index], 0], return_index=True)[1]:
        row, column = rows[index[identity]], columns[index[identity]]
        geom = arena.model.geom(int(values[row, column, 0]))
        names.append((identity, geom.name or arena.model.body(int(geom.bodyid[0])).name or 'unnamed'))
    names = [name for _, name in sorted(names)]
    return dict(center=names[0], nearby_learner=next((name for name in names if name.startswith('learner')), None))


def measure(arena, stream, frame, source, active, speed, window):
    """Evaluate visibility and geometry after the anonymous selections are final."""
    labels = render_labels(arena, stream)
    world = np.array((arena.paddle, arena.ball)) - ORIGIN
    references = _camera_truth(arena, stream).project(world)[:, :2]
    output = dict(before_contact=window)
    for index, (role, geom) in enumerate(zip(ROLES, ('learner_racket', 'ball'), strict=True)):
        geos = (labels[..., 0] == arena.model.geom(geom).id) & (labels[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        def error(track):
            return None if track is None else float(np.linalg.norm(pixel(track, frame.shape) - references[index]))

        ranked = (dict(pong=source.controlled, active=active) if role == 'actor'
                  else dict(pong=source.relevant, speed=speed))
        available = [error(track) for track in source.tracks]
        aged = [error(track) for track in source.tracks if track.age >= (9 if index == 0 else 5)]
        output[role] = dict(visible=bool(np.count_nonzero(geos) >= 3),
                            closest=min(available, default=None), persistent_closest=min(aged, default=None),
                            candidates=len(available), selected={name: error(track) for name, track in ranked.items()})
        if role == 'actor':
            output[role]['parts'] = {name: None if track is None else selected_part(
                labels, arena, pixel(track, frame.shape)) for name, track in ranked.items()}
    return output


def replay(arena, row, models, patches):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    eyes = {key: BabblingEye(patches[key], models) for key in STREAMS}
    digest = sha256()
    history = {stream: [] for stream in STREAMS}
    previous = None
    contacts = {event['step'] for event in row['events']
                if event['kind'] == 'contact' and event['side'] == 1}
    for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
        frames = arena.render_cameras()
        window = any(0 < step - tick <= 10 for step in contacts)
        for stream, frame in frames.items():
            source, active, speed = eyes[stream].observe(frame, previous)
            digest.update(stream.encode())
            digest.update(frame)
            history[stream].append(measure(arena, stream, frame, source, active, speed, window))
        arena._rival_attempt = True
        arena._aim = lambda *args: np.asarray(far)
        _, reward, done, capped, _ = arena.step(near, render=False)
        if bool(done or capped) != (tick + 1 == len(row['actions'])):
            raise ValueError('the original action tapes changed point length')
        previous = near
    if reward != row['reward'] or arena.point_result() != row['outcome'] or digest_json(arena.events) != digest_json(
            row['events']) or digest.hexdigest() != row['rgb_sha256']:
        raise ValueError('the original tapes did not reproduce RGB, point and referee events')
    return dict(seed=row['seed'], steps=len(row['actions']), near_contacts=sorted(contacts),
                outcome=row['outcome'], rgb_sha256=digest.hexdigest(), streams=history)


def summarize(points):
    frames_by_stream = {key: [frame for point in points for frame in point['streams'][key]] for key in STREAMS}
    output = {}
    for key, frames in frames_by_stream.items():
        result = {}
        for index, role in enumerate(ROLES):
            tolerance = TOLERANCE[index]
            rows = [frame[role] for frame in frames if frame['before_contact'] and frame[role]['visible']]
            modes = ('pong', 'active') if index == 0 else ('pong', 'speed')
            proposal = dict(visible=len(rows))
            for field in ('closest', 'persistent_closest'):
                proposal[field] = sum(row[field] is not None and row[field] <= tolerance for row in rows)
            for mode in modes:
                errors = [row['selected'][mode] for row in rows if row['selected'][mode] is not None]
                proposal[mode] = dict(proposed=len(errors), correct=sum(error <= tolerance for error in errors),
                                      median_error_px=float(np.median(errors)) if errors else None)
                if role == 'actor':
                    parts = [row['parts'][mode] for row in rows if row['parts'][mode] is not None]
                    for field in ('center', 'nearby_learner'):
                        proposal[mode][field] = dict(Counter(part[field] or 'none' for part in parts))
            result[role] = proposal
        output[key] = result
    output['joint'] = {}
    for index, role in enumerate(ROLES):
        tolerance = TOLERANCE[index]
        both = [tuple(frame[role] for frame in pair)
                for pair in zip(*frames_by_stream.values(), strict=True)
                if pair[0]['before_contact'] and all(frame[role]['visible'] for frame in pair)]
        paired = dict(visible=len(both))
        for field in ('closest', 'persistent_closest'):
            paired[field] = sum(all(row[field] is not None and row[field] <= tolerance for row in eyes)
                                for eyes in both)
        for mode in ('pong', 'active' if index == 0 else 'speed'):
            values = [tuple(row['selected'][mode] for row in eyes) for eyes in both]
            proposed = [value for value in values if None not in value]
            paired[mode] = dict(proposed=len(proposed), correct=sum(max(value) <= tolerance for value in proposed))
        output['joint'][role] = paired
    return output


def run(output):
    source = _read(SOURCE)
    model = RoleModel(**source['roles']['learned_reward_time'])
    target = _read(TARGET)
    rows = [row['options']['0'] for row in target['rows']]
    if PROBE_SEED in [row['seed'] for row in rows]:
        raise ValueError('the probe and held seeds overlap')
    arena = arena_for('anchor', live=True)
    points = []
    try:
        patches, hashes = probe(arena)
        print('probe', {key: len(value) for key, value in patches.items()}, flush=True)
        data = dict(schema='tt22-anonymous-babbling-1', probe_seed=PROBE_SEED, probe_amplitude=AMPLITUDE,
                    probe_sample_controls=TIMES, probe_rgb_sha256=hashes,
                    probes={key: [[patch.point.tolist(), float(patch.support)] for patch in value]
                            for key, value in patches.items()},
                    source_sha256=_hash(SOURCE), target_sha256=_hash(TARGET), points=points, complete=False)
        for row in rows:
            point = replay(arena, row, model, patches)
            points.append(point)
            print(point['seed'], point['steps'], point['near_contacts'], flush=True)
            data['summary'] = summarize(points)
            _write(output, data)
        data['complete'] = True
        _write(output, data)
    finally:
        arena.close()


def compare_candidate_ceiling(output):
    result = _read(output)
    if not result['complete']:
        raise ValueError('finish the active assay before comparing its candidate ceiling')
    rows = [row['options']['0'] for row in _read(TARGET)['rows']]
    arena = arena_for('anchor', live=True)
    counts = {key: dict(visible=0, closest=0, persistent_closest=0) for key in (*STREAMS, 'joint')}
    try:
        with TemporaryDirectory(prefix='tt22-babble-baseline-') as directory:
            shared = library(directory)
            for row, measured in zip(rows, result['points'], strict=True):
                arena.reset(seed=row['seed'], server=row['role'], render=False)
                eyes = {key: CandidateFrames(shared) for key in STREAMS}
                digest = sha256()
                for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                    selected = {}
                    for key, frame in arena.render_cameras().items():
                        digest.update(key.encode())
                        digest.update(frame)
                        tracks = eyes[key].observe(frame)
                        past = measured['streams'][key][tick]
                        if not (past['before_contact'] and past['actor']['visible']):
                            continue
                        reference = _camera_truth(arena, key).project(arena.paddle - ORIGIN)[:2]
                        near_center = [t for t in tracks
                                       if np.linalg.norm(pixel(t, frame.shape) - reference) <= TOLERANCE[0]]
                        selected[key] = (bool(near_center), any(t.age >= 9 for t in near_center))
                    for key, (candidate, persistent) in selected.items():
                        counts[key]['visible'] += 1
                        counts[key]['closest'] += candidate
                        counts[key]['persistent_closest'] += persistent
                    if len(selected) == 2:
                        counts['joint']['visible'] += 1
                        counts['joint']['closest'] += all(value[0] for value in selected.values())
                        counts['joint']['persistent_closest'] += all(value[1] for value in selected.values())
                    arena._rival_attempt = True
                    arena._aim = lambda *args: np.asarray(far)
                    arena.step(near, render=False)
                if digest.hexdigest() != measured['rgb_sha256']:
                    raise ValueError('the original candidate control used different RGB')
                print('matched original detector', measured['seed'], counts, flush=True)
    finally:
        arena.close()
    result['original_candidate_ceiling'] = counts
    _write(output, result)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--compare-original', action='store_true')
    args = parser.parse_args()
    if args.compare_original:
        compare_candidate_ceiling(args.output)
    else:
        run(args.output)

"""Audit source-discovered object roles against held, unmarked physical SO-101 replay."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt3_visual import RoleModel
from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt19_view_trial import CAMERAS
from .tt20_natural_trial import ORIGIN, _camera_truth
from .tt22_anonymous import AnonymousEye, Controls, pixel
from .tt22_opponent_trial import arena_for
from .tt22_relation_trial import library


SOURCE = ROOT / 'artifacts/tt3/discovered-reward-roles.json.gz'
TARGET = ROOT / 'artifacts/tt22/tt22_opponent-native.json.gz'
OUTPUT = TARGET.with_name('anonymous-roles.json.gz')
MODES = ('pong', 'reward_shuffled', 'command_shuffled', 'speed', 'random')
ROLES = ('actor', 'mover')
TOLERANCE = (8., 4.)
CONFIDENCE = (.0, .05, .10, .20, .35, .50)


def visibility(arena, stream, geom):
    """Evaluator-only pixel IDs. This image never reaches the observer."""
    renderer = arena._view_renderer()
    renderer.enable_segmentation_rendering()
    try:
        image = arena._render_view(renderer, arena.data, CAMERAS[stream][0])
    finally:
        renderer.disable_segmentation_rendering()
    return [int(np.count_nonzero((image[..., 0] == target)
                                 & (image[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))) for target in geom]


def measurement(frame, arena, stream, primary, comparisons, *, window, geom):
    pose = _camera_truth(arena, stream)
    world = np.array((arena.paddle, arena.ball)) - ORIGIN
    center = pose.project(world)[:, :2]
    pixels = visibility(arena, stream, geom)
    eligible = ([t for t in primary.tracks if t.age >= 9],
                [t for t in primary.tracks if t.age >= 5])
    metrics = {}
    for index, role in enumerate(ROLES):
        def distance(track):
            return float(np.linalg.norm(pixel(track, frame.shape) - center[index]))

        best = min((distance(track) for track in eligible[index]), default=None)
        chosen = {name: None if pair[index] is None else distance(pair[index])
                  for name, pair in comparisons.items() if name != 'speed' or index == 1}
        metrics[role] = dict(visible=pixels[index] >= 3, size_px=pixels[index],
                             labeled_center=center[index].tolist(),
                             candidate_px=best, proposals=len(eligible[index]),
                             choices_px=chosen, margin=(primary.controlled_evidence,
                                                        primary.relevant_evidence)[index])
    metrics['before_contact'] = window
    return metrics


def replay(arena, row, shared, models, streams):
    arena.reset(seed=row['seed'], server=row['role'], render=False)
    geom = [arena.model.geom(name).id for name in ('learner_racket', 'ball')]
    digest = sha256()
    result = {stream: [] for stream in streams}
    eyes = {stream: AnonymousEye(shared, models[0]) for stream in streams}
    controls = {stream: Controls(eye, models[1], row['seed'] + index)
                for index, (stream, eye) in enumerate(eyes.items())}
    contacts = {event['step'] for event in row['events']
                if event['kind'] == 'contact' and event['side'] == 1}
    previous = None
    for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
        frames = arena.render_cameras()
        for stream, frame in frames.items():
            digest.update(stream.encode())
            digest.update(frame)
            if stream in eyes:
                primary = eyes[stream].update(frame, previous)
                variants = controls[stream].update(frame, previous, primary)
                window = any(0 < event - tick <= 10 for event in contacts)
                result[stream].append(measurement(frame, arena, stream, primary, variants,
                                                  window=window, geom=geom))
        arena._rival_attempt = True
        arena._aim = lambda *args: np.asarray(far)
        _, reward, done, capped, _ = arena.step(near, render=False)
        previous = near
        if bool(done or capped) != (tick + 1 == len(row['actions'])):
            raise ValueError('two-arm replay changed point duration')
    if (row['reward'] != reward or row['outcome'] != arena.point_result()
            or digest_json(row['events']) != digest_json(arena.events)
            or row['rgb'] != digest.hexdigest()):
        raise ValueError('raw RGB or strict two-arm replay changed')
    return dict(seed=row['seed'], role=row['role'], contacts=sorted(contacts), outcome=row['outcome'],
                steps=len(row['actions']), rgb_sha256=digest.hexdigest(), streams=result)


def summarize(points, streams):
    output = {}
    for stream in streams:
        output[stream] = {}
        for split in ('all', 'before_contact'):
            frames = [frame for point in points for frame in point['streams'][stream]
                      if split == 'all' or frame[split]]
            result = {}
            for index, role in enumerate(ROLES):
                records = [frame[role] for frame in frames if frame[role]['visible']]
                invisible = [frame[role] for frame in frames if not frame[role]['visible']]
                tolerance = TOLERANCE[index]
                rankers = {}
                for name in MODES:
                    if role == 'actor' and name == 'speed':
                        continue
                    proposals = [r['choices_px'][name] for r in records if r['choices_px'][name] is not None]
                    rankers[name] = dict(proposed=len(proposals), correct=sum(p <= tolerance for p in proposals),
                                         proposed_invisible=sum(r['choices_px'][name] is not None for r in invisible),
                                         median_error_px=float(np.median(proposals)) if proposals else None)
                margins = {}
                for threshold in CONFIDENCE:
                    proposals = [r['choices_px']['pong'] for r in records
                                 if r['margin'] >= threshold and r['choices_px']['pong'] is not None]
                    margins[str(threshold)] = dict(proposed=len(proposals), correct=sum(p <= tolerance for p in proposals))
                candidates = [r['candidate_px'] for r in records if r['candidate_px'] is not None]
                result[role] = dict(visible=len(records), candidates=sum(p <= tolerance for p in candidates),
                                    rankers=rankers, confidence=margins)
            output[stream][split] = result
    return output


def summarize_joint(points, streams):
    if len(streams) != 2:
        return {}
    output = {}
    for split in ('all', 'before_contact'):
        paired = (frames for point in points for frames in zip(
            *(point['streams'][stream] for stream in streams), strict=True)
                  if split == 'all' or frames[0][split])
        scores = {role: dict(visible=0, candidates=0, proposed=0, correct=0) for role in ROLES}
        for frames in paired:
            for role, tolerance in zip(ROLES, TOLERANCE, strict=True):
                rows = [frame[role] for frame in frames]
                if not all(row['visible'] for row in rows):
                    continue
                counts = scores[role]
                counts['visible'] += 1
                counts['candidates'] += all(row['candidate_px'] is not None and row['candidate_px'] <= tolerance
                                            for row in rows)
                choices = [row['choices_px']['pong'] for row in rows]
                if all(value is not None for value in choices):
                    counts['proposed'] += 1
                    counts['correct'] += all(value <= tolerance for value in choices)
        output[split] = scores
    return output


def run(output, streams, variant):
    source = _read(SOURCE)
    weights = {name: source['roles'][name] for name in ('learned_reward_time', 'shuffled_reward_time')}
    models = tuple(RoleModel(**values) for values in weights.values())
    target = _read(TARGET)
    arena = arena_for(variant, live=True)
    points = []
    try:
        with TemporaryDirectory(prefix='tt22-anonymous-') as folder:
            shared = library(folder)
            for original in target['variants'][variant]['rows']:
                point = replay(arena, original, shared, models, streams)
                points.append(point)
                print(point['seed'], point['steps'], point['contacts'], flush=True)
    finally:
        arena.close()
    data = dict(schema='tt22-anonymous-roles-1', source_sha256=_hash(SOURCE), target_sha256=_hash(TARGET),
                source_training_seeds=source['train_seeds'], source_models=weights, variant=variant,
                tolerance_px=dict(zip(ROLES, TOLERANCE, strict=True)), before_contact_frames=10,
                points=points, summary=summarize(points, streams), joint=summarize_joint(points, streams))
    _write(output, data)
    return data


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--streams', type=str, choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--variant', choices=('anchor', 'steered', 'coach'), default='anchor')
    args = parser.parse_args()
    streams = ('left', 'right') if args.streams == 'both' else (args.streams,)
    run(args.output, streams, args.variant)


if __name__ == '__main__':
    main()

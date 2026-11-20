"""Replay acquired carriage preferences on an RGB arm after a new camera placement."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from body_effect_learner import BodyLearner
from body_reaching_arena import (
    DEFAULT_MAX_STEPS, GOAL_TOLERANCE, SUCCESS_TICKS, ReachCase, decode_markers, render_markers,
)
from body_visual_utility import FAMILIES, TARGET_BODY, UtilityLearner, offset_features, sample_layout, sample_rule
from camera_reaching_arena import RGB_CAMERA_SIZE, CameraView, decode_camera, render_camera
from run_position_arm_transfer import position_arm


REGISTRATION_MARKS = ((.10, -.43), (.78, -.43), (.10, .43))
CONDITIONS = ('registered', 'stale', 'fresh', 'shuffled')
SOURCE_ARCHIVE = Path(__file__).resolve().parents[2] / 'artifacts/body-transfer/cross-body-results.zip'


def source_markers():
    # Take an additional registration exposure at the source camera setting.
    # The consumer receives only the detected centers, in the acquired feature space.
    return np.stack([decode_markers(render_markers(mark, (.40, 0.)))[0]
                     for mark in REGISTRATION_MARKS])


def register(seen, source):
    """Estimate the affine map between corresponding painted marks in two views."""
    x = np.column_stack((seen, np.ones(len(seen))))
    if x.shape != (3, 3) or len(source) != 3 or abs(np.linalg.det(x)) < 1:
        raise ValueError('expected three noncollinear registration marks')
    return np.linalg.solve(x, source)


def choice_features(observation, transform):
    # Translation cancels out when subtracting the visible reference.
    offsets = (observation.markers - observation.reference) @ transform[:2]
    return offset_features(offsets, np.zeros(2))


def pixels(view, tick, tip, layout):
    return render_camera(view, tick, tip, layout.markers, layout.reference, fiducials=REGISTRATION_MARKS)


def play(layout, rule, view, old_transform, source, prior, condition, seed):
    winner = rule.winner(layout.markers, layout.reference, False)
    case = ReachCase(layout.start, layout.markers[winner])
    arena = position_arm(TARGET_BODY, case, max_steps=DEFAULT_MAX_STEPS)
    motor = BodyLearner(action_dim=3, seed=seed)
    utility = UtilityLearner(prior if condition != 'fresh' else None)
    tie = int(np.random.default_rng(seed).integers(2))
    image = pixels(view, 0, arena.ground_truth_position, layout)
    chosen_pixel = None
    first_paid = None
    visible_dwell = actual_dwell = 0
    last_contact = None
    missing_frames = 0
    actions = sha256()
    observations = sha256()
    previous_tip = previous_action = None
    waited_for_absent_marker = False
    while not arena.done:
        try:
            obs = decode_camera(image)
        except ValueError:
            obs = None
        if obs is None or obs.tip is None or obs.markers is None or len(obs.markers) != 2 or obs.reference is None:
            # Charge every tick of a dropout. Skip the motion transition across that gap.
            action = np.zeros(3)
            missing_frames += 1
            previous_tip = None
            waited_for_absent_marker = True
            visible_dwell = 0
        else:
            tip = (obs.tip - RGB_CAMERA_SIZE / 2) / (RGB_CAMERA_SIZE / 2)
            if previous_tip is not None:
                motor.observe(previous_tip, previous_action, tip)
            if chosen_pixel is None:
                if len(obs.fiducials) == 3:
                    current_transform = register(obs.fiducials, source)
                    transform = old_transform if condition == 'stale' else current_transform
                    chosen_pixel = obs.markers[utility.choose(choice_features(obs, transform), tie)]
            if chosen_pixel is None:
                action = np.zeros(3)
                missing_frames += 1
            else:
                selected = int(np.argmin(np.linalg.norm(obs.markers - chosen_pixel, axis=1)))
                chosen_pixel = obs.markers[selected]
                # A cautious visual dwell is common to all arms. A wrong hold yields zero.
                distance = np.linalg.norm(obs.tip - chosen_pixel)
                visible_dwell = visible_dwell + 1 if distance < 3 else 0
                if visible_dwell >= SUCCESS_TICKS + 2:
                    chosen_pixel = obs.markers[1 - selected]
                    visible_dwell = 0
                target = (chosen_pixel - RGB_CAMERA_SIZE / 2) / (RGB_CAMERA_SIZE / 2)
                action = motor.act(tip, target)
            previous_tip = tip
        actions.update(np.asarray(action, dtype='<f8').tobytes())
        observations.update(image.tobytes())
        arena.step(action)
        previous_action = action
        # The evaluator alone sees body geometry and exact paying region.
        pos = arena.ground_truth_position
        contact = next((i for i, point in enumerate(layout.markers)
                        if np.linalg.norm(np.asarray(pos) - point) <= GOAL_TOLERANCE), None)
        actual_dwell = actual_dwell + 1 if contact == last_contact and contact is not None else 1
        last_contact = contact
        if first_paid is None and contact is not None and actual_dwell == SUCCESS_TICKS:
            first_paid = contact == winner
        image = pixels(view, arena.steps, pos, layout)
    return dict(first_hold_paid=bool(first_paid), any_hold=first_paid is not None,
                success=bool(arena.success), commands=arena.steps, missing_frames=missing_frames,
                marker_dropout=waited_for_absent_marker, actions_sha256=actions.hexdigest(),
                observations_sha256=observations.hexdigest())


def run(seed, count, *, moved=True):
    with ZipFile(SOURCE_ARCHIVE) as archive:
        checkpoints = json.loads(archive.read('visual-utility/frozen.json'))['utility_checkpoints']
    rows = []
    source = source_markers()
    for i, key in enumerate(sorted(checkpoints)[:count]):
        family = FAMILIES[i % 2]
        rule = sample_rule(int(key), family)
        layout_seed, before_seed, after_seed, motor_seed = np.random.SeedSequence([seed,i]).spawn(4)
        layout = sample_layout(np.random.default_rng(layout_seed))
        previous_view = CameraView.sample(np.random.default_rng(before_seed), changed=False)
        camera = CameraView.sample(np.random.default_rng(after_seed), changed=True)
        previous_view = replace(previous_view, scale=previous_view.scale * .88)
        camera = replace(camera, scale=camera.scale * .88) if moved else previous_view
        # Calibration exposure without the real layout prevents target/mark overlap.
        # Dedicated calibration is possible before moving the camera.
        calibration = type(layout)((.41, .06), ((.41, -.18), (.52, .28)), (.30, .09))
        ref = decode_camera(pixels(previous_view, 0, calibration.start, calibration))
        old_transform = register(ref.fiducials, source)
        outcomes = {}
        for condition in CONDITIONS:
            prior = checkpoints[key]['shuffled' if condition == 'shuffled' else 'carried']
            outcomes[condition] = play(layout, rule, camera, old_transform, source,
                                       prior, condition, motor_seed)
        rows.append(dict(source_seed=int(key), family=family, **outcomes))
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {k:sum(row[condition][k] for row in rows) for k in
                              ('first_hold_paid', 'any_hold', 'success', 'commands', 'missing_frames')}
    rng = np.random.default_rng(830453)
    indices = rng.integers(len(rows),size=(4000, len(rows)))
    comparisons = {}
    for other in ('stale', 'fresh', 'shuffled'):
        differences = np.array([r[other]['commands']-r['registered']['commands'] for r in rows])
        comparisons[other] = dict(mean=float(differences.mean()),
                                  paired_ci95=np.quantile(differences[indices].mean(axis=1), [.025,.975]).tolist())
    files = [Path(__file__),Path(__file__).with_name('camera_reaching_arena.py'),SOURCE_ARCHIVE]
    return dict(seed=seed, moved_camera=moved, streams=len(rows), summary=summary, savings=comparisons, rows=rows,
                source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--seed',type=int,default=434850)
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--stationary-camera', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.count <= 64:
        parser.error('count must be between 1 and 64')
    result = run(args.seed, args.count, moved=not args.stationary_camera)
    with args.output.open('x') as f:
        json.dump(result,f,allow_nan=False)
    print(result['summary'], result['savings'])

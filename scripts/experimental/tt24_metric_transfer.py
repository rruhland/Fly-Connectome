"""Transfer a fixed wrist action readout with public metric stereo calibration."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import WristPupil, WristState, _INC, _unit_camera_ray, episode, summarize
from .tt24_bounce_ident_transfer import TARGET_REGIME, _verify_tape
from .tt24_metric_stereo import Stereo, calibrated_cameras, triangulate
from .tt24_predict_reference import TrainingEyes, image_read, velocities


CALIBRATION = tuple(range(256_001, 256_005))
DEVELOPMENT = tuple(range(256_101, 256_107))
HELD = tuple(range(256_201, 256_217))
LONG_HELD = tuple(range(256_401, 256_405))
TICKS_PER_DROP = 80


def _training_rig():
    fixed = TrainingEyes()
    arena = BounceArena(view='wrist', width=192, height=144, arm='alternate')
    arena._model.vis.global_.fovy = fixed._model.vis.global_.fovy
    arena.eyes = fixed.eyes
    fixed.close()
    return arena


def capture(path):
    """Record original pupil and small exploratory motor variation with public sensors."""
    arena = _training_rig()
    records = []
    try:
        eyes = calibrated_cameras(arena)
        for seed in CALIBRATION:
            policy = WristPupil.load(path)
            obs = arena.reset(seed=seed, split='held', regime=TARGET_REGIME)
            rng = np.random.default_rng(seed)
            perturbation = np.zeros(arena.action_size)
            retinal, stationary, encoders, commands = [], [], [], []
            for index in range(TICKS_PER_DROP):
                retinal.append(image_read(obs['rgb']))
                stationary.append([image_read(frame) for frame in TrainingEyes.fixed_images(arena)])
                encoders.append(np.r_[obs['joints'], obs['joint_velocity']])
                intended = policy.command(obs)
                if index % 5 == 0:
                    perturbation = rng.uniform(-.052, .052, arena.action_size)
                action = np.clip(intended + perturbation, -1., 1.)
                policy.issued += _INC * (action - intended)
                commands.append(action)
                obs, _, done = arena.step(action)
                if done:
                    break
            records.append(dict(seed=seed, wrist=np.asarray(retinal), fixed=np.asarray(stationary),
                                enc=np.asarray(encoders), commands=np.asarray(commands)))
            print('ordinary RGB calibration', seed, len(commands), flush=True)
        return records, eyes
    finally:
        arena.close()


def split_metric(records, eyes):
    """Triangulate only paired visible objects and retain a ray-gap quality weight."""
    output = []
    for data in records:
        photo = data['fixed']
        ball, gaps, _ = triangulate(eyes, photo[:, :, :2], tolerance=.016)
        blade, blade_gaps, _ = triangulate(eyes, photo[:, :, 3:5], tolerance=.023)
        output.append(data | dict(ball=ball, blade=blade, weight=1. / (1. + (gaps / .006)**2),
                                  blade_weight=1. / (1. + (blade_gaps / .008)**2)))
    return output


class ResidualRig:
    """Calibrate a small pose correction around the existing wrist decoder."""

    def __init__(self, source, examples, *, correct_blade=True, trust=1.):
        self.source = source
        self.default = source.default
        self.image_height = getattr(source, 'image_height', 144)
        q, pixels, ball, blade, weight, blade_weight = (np.concatenate([item[key] for item in examples])
                                                       for key in ('enc', 'wrist', 'ball', 'blade', 'weight', 'blade_weight'))
        q = q[:, :5]
        self.center = np.median(q, axis=0)
        forms = self.forms(q)
        source_body = np.asarray([source.body(row) for row in q])
        common = np.isfinite(pixels[:, :3]).all(axis=1) & np.isfinite(ball).all(axis=1)
        strength = np.sqrt(weight[common])
        visible = pixels[common, :3]
        clipped = abs(visible[:, 1]) + visible[:, 2] > self.image_height * .48
        strength[clipped] *= .46
        self.rotation = np.zeros((6, 3))
        self.translation = np.zeros((6, 3))
        self.blade_delta = np.zeros((6, 3))
        local = self._local(visible)
        if len(local) >= 12:
            known, mat = source_body[common, :3], source_body[common, 3:12].reshape(-1, 3, 3)
            context = forms[common]
            reference = ball[common]
            spread = float(np.linalg.svd(local - local.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(local)))
            self.trusted = min(1., len(local) / 50) * min(1., spread / .022) * trust
            penalty = np.r_[[.055] * 3, [.33] * 15, [.17] * 3, [.61] * 15] / max(self.trusted, .07)

            def cost(theta):
                rotation, shift = theta.reshape(2, 6, 3)
                rotated = Rotation.from_rotvec(context @ rotation).apply(local) + context @ shift
                prediction = known + np.einsum('ijk,ik->ij', mat, rotated)
                return np.r_[((prediction - reference) * strength[:, None]).ravel(), penalty * theta]

            bound = np.r_[[.33] * 3, [.12] * 15, [.10] * 3, [.043] * 15]
            result = least_squares(cost, np.zeros(36), bounds=(-bound, bound), loss='soft_l1',
                                   f_scale=.006, max_nfev=32)
            self.rotation, self.translation = result.x.reshape(2, 6, 3)
            self.metrics = dict(input_frames=len(q), usable_ball_pairs=int(common.sum()),
                                local_position_minor_span_m=spread, prior_trust_scale=self.trusted,
                                initial_ball_rmse_m=float(np.sqrt(np.mean(cost(np.zeros(36))[:-36]**2))),
                                fitted_ball_rmse_m=float(np.sqrt(np.mean(cost(result.x)[:-36]**2))))
        else:
            self.metrics = dict(input_frames=len(q), usable_ball_pairs=int(common.sum()), prior_trust_scale=0.)

        good_blade = np.isfinite(blade).all(axis=1)
        if correct_blade and good_blade.sum() >= 12:
            design = forms[good_blade]
            delta = np.clip(blade[good_blade] - source_body[good_blade, 12:15], -.09, .09)
            weights = blade_weight[good_blade]
            precision = design.T @ (weights[:, None] * design) + np.diag((4., 14., 14., 14., 14., 14.))
            self.blade_delta = np.linalg.solve(precision, design.T @ (weights[:, None] * delta))
        self.metrics.update(usable_blade_pairs=int(good_blade.sum()), blade_map_updated=correct_blade,
                            rotation_correction_at_center=self.rotation[0].tolist(),
                            camera_translation_at_center_m=self.translation[0].tolist(),
                            blade_translation_at_center_m=self.blade_delta[0].tolist())

    def forms(self, q):
        x = np.atleast_2d(q)
        return np.c_[np.ones(len(x)), np.clip((x - self.center) / .16, -2.2, 2.2)]

    def _local(self, ellipse):
        return self.source._camera_forms(_unit_camera_ray(ellipse, self.source.focal)) @ self.source.camera_weights

    def body(self, joints):
        self.context = self.forms(joints)[0]
        values = self.source.body(joints)
        values[12:15] += self.context @ self.blade_delta
        return values

    def ball(self, ellipse, body):
        if ellipse is None:
            return None
        local = self._local(ellipse)[0]
        shifted = Rotation.from_rotvec(self.context @ self.rotation).apply(local) + self.context @ self.translation
        return body[:3] + body[3:12].reshape(3, 3) @ shifted


def train_public_motion(source, optics, records):
    features, targets = [], []
    for data in records:
        observer = WristState(optics, source)
        sequence = []
        for q, retinal in zip(data['enc'][:, :5], data['wrist'][:, :3], strict=True):
            visible = np.isfinite(retinal).all()
            observer.read(retinal if visible else None, q)
            sequence.append(observer.motion_input)

        positions = data['ball'].copy()
        valid = np.isfinite(positions).all(axis=1)
        if valid.sum() < 8:
            continue
        for axis in range(3):
            positions[:, axis] = np.interp(np.arange(len(valid)), np.flatnonzero(valid), positions[valid, axis])
        velocity = velocities(positions)
        index = np.arange(3, len(valid) - 2)
        index = index[valid[index] & valid[index - 1] & valid[index + 1]]
        features.extend(np.asarray(sequence)[index])
        targets.extend(velocity[index])
    motion = deepcopy(source)
    if len(features) >= 24:
        values = motion.design(features)
        residual = np.asarray(targets) - values @ motion.weights
        ridge = np.eye(values.shape[1]) * 34.
        motion.weights += np.linalg.solve(values.T @ values + ridge, values.T @ residual)
    return motion


def _evaluate(arena, seeds, actor, decoder, motion):
    sensor = lambda _: WristState(decoder, motion)
    cases = [episode(arena, seed, actor=actor, sensor_factory=sensor, regime=TARGET_REGIME)[0] for seed in seeds]
    return dict(summary=summarize(cases), episodes=cases)


def variants_for(source, fit_data):
    centre, motion = source.sensor.decoder, source.sensor.motion
    full = ResidualRig(centre, fit_data)
    ball = ResidualRig(centre, fit_data, correct_blade=False)
    one = ResidualRig(centre, fit_data[:1])
    slow = ResidualRig(centre, fit_data, trust=.35)
    corrected_motion = train_public_motion(motion, full, fit_data)
    return dict(frozen=(centre, motion), public_one=(one, motion), public_ball=(ball, motion),
                public_full=(full, motion), public_strong_prior=(slow, motion), public_motion=(full, corrected_motion))


def _matched_privileged_examples(records):
    """Create an isolated label-quality control on exactly the recorded decisions."""
    arena = BounceArena(view='none', width=192, height=144, arm='alternate')
    rebuilt = []
    try:
        for data in records:
            obs = arena.reset(seed=data['seed'], split='held', regime=TARGET_REGIME)
            ball, blade = [], []
            for expected, action in zip(data['enc'], data['commands'], strict=True):
                actual = np.r_[obs['joints'], obs['joint_velocity']]
                assert np.array_equal(actual, expected)
                mujoco.mj_kinematics(arena._model, arena._data)
                ball.append(arena._ball_position.copy())
                blade.append(arena._data.geom_xpos[arena._blade].copy())
                obs, _, _ = arena.step(action)
            rebuilt.append(data | dict(ball=np.where(np.isfinite(data['ball']), ball, np.nan),
                                       blade=np.where(np.isfinite(data['blade']), blade, np.nan)))
    finally:
        arena.close()
    return rebuilt


def load_transferred(source_path, correction_path):
    pupil = WristPupil.load(source_path)
    with np.load(correction_path, allow_pickle=False) as values:
        if values['source_sha256'].item() != hashlib.sha256(Path(source_path).read_bytes()).hexdigest():
            raise ValueError('correction requires the original immutable stock pupil')
        decoder = ResidualRig.__new__(ResidualRig)
        decoder.source = pupil.sensor.decoder
        decoder.default = decoder.source.default
        decoder.image_height = getattr(decoder.source, 'image_height', 144)
        for name in ('center', 'rotation', 'translation', 'blade_delta'):
            setattr(decoder, name, values[name])
        motion = pupil.sensor.motion
        motion.weights = values['motion_weights']
    pupil.sensor = WristState(decoder, motion)
    return pupil


def _deploy(arena, seed, source_path, collection):
    pupil = load_transferred(source_path, collection)
    obs = arena.reset(seed=seed, split='held', regime=TARGET_REGIME)
    commands = []
    for _ in range(arena.max_steps):
        action = pupil.command(obs)
        commands.append(action.tolist())
        obs, _, done = arena.step(action)
        if done:
            break
    return arena.judge_metrics() | dict(commands=commands)


def _export_and_match(arena, report, records, variants, source, args):
    selected = report['selected']
    decoder, motion = variants[selected]
    path = args.out / 'metric-transfer-map.npz'
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    np.savez_compressed(path, source_sha256=digest, center=decoder.center, rotation=decoder.rotation,
                        translation=decoder.translation, blade_delta=decoder.blade_delta, motion_weights=motion.weights)
    expected = max(report['cohorts'][selected]['episodes'], key=lambda row: row['longest_consecutive_rebounds'])
    actual = _deploy(arena, expected['seed'], args.source, path)
    assert json.loads(json.dumps(actual)) == json.loads(json.dumps(expected))
    report['export_replay'] = dict(exact=True, seed=expected['seed'], frames=len(actual['commands']),
                                   consecutive=actual['longest_consecutive_rebounds'], source_sha256=digest)

    matched = _matched_privileged_examples(records)
    privileged = variants_for(source, matched)[selected]
    run = _evaluate(arena, HELD, source.actor, *privileged)
    exact = max(run['episodes'], key=lambda row: row['longest_consecutive_rebounds'])
    _verify_tape(arena, exact, TARGET_REGIME)
    report['cohorts']['matched_privileged_position_labels'] = run | dict(exact_replay_seed=exact['seed'])
    report['matched_label_control'] = dict(exact_encoder_replay=True,
        labels='privileged ball and blade positions on the same ordinary trajectory frames with paired RGB coverage',
        fit='identical frozen-head residual model and causal motion choice')
    print('same ordinary exposure privileged label comparison', run['summary'], flush=True)


def _save(environment, path, source, records, eyes, args):
    fit_data = split_metric(records, eyes)
    variants = variants_for(source, fit_data)
    if args.finish_archive:
        with gzip.open(path, 'rt', encoding='utf-8') as input_file:
            report = json.load(input_file)
        assert report['calibration_frames'] == [len(record['wrist']) for record in records]
        _export_and_match(environment, report, fit_data, variants, source, args)
        working = path.with_name('metric-transfer-working.json.gz')
        with gzip.open(working, 'wt', encoding='utf-8') as output:
            json.dump(report, output, separators=(',', ':'))
        working.replace(path)
        return
    public = Stereo(eyes, records)
    fixed = np.concatenate([data['fixed'] for data in records])
    score = public.check(fixed)
    centre = source.sensor.decoder
    dev = {}
    for name, (decoder, fitted_motion) in variants.items():
        run = _evaluate(environment, DEVELOPMENT, source.actor, decoder, fitted_motion)
        dev[name] = run['summary']
        print('development', name, dev[name], flush=True)
    public_names = [name for name in variants if name.startswith('public_')]
    selected = max(public_names, key=lambda name: dev[name]['mean'])
    print('frozen public choice', selected, flush=True)
    report = dict(schema='tt24-metric-transfer-1', target='alternate arm and changed drive',
                  adaptation='paired public RGB and own encoders under original pupil plus zero-mean babble',
                  target_motor_label_provided=False, target_world_label_provided=False,
                  original_calibration_privileged=True, runtime='wrist RGB and own joint angles',
                  calibration_seeds=CALIBRATION, calibration_frames=[len(record['wrist']) for record in records],
                  optical_quality=score, development_seeds=DEVELOPMENT, development=dev,
                  selected=selected, held_seeds=HELD, cohorts={}, calibration={}, reference_pose_known=True)
    for name, (decoder, _) in variants.items():
        if decoder is not centre:
            report['calibration'][name] = decoder.metrics
    if not args.development_only:
        supervised = WristPupil.load(args.supervised)
        cohorts = report['cohorts']
        for name, (decoder, fitted_motion) in (('frozen', variants['frozen']), (selected, variants[selected]),
                                                  ('supervised', (supervised.sensor.decoder, supervised.sensor.motion))):
            result = _evaluate(environment, HELD, source.actor, decoder, fitted_motion)
            top = max(result['episodes'], key=lambda row: row['longest_consecutive_rebounds'])
            _verify_tape(environment, top, TARGET_REGIME)
            result['exact_replay_seed'] = top['seed']
            cohorts[name] = result
            print('held', name, result['summary'], flush=True)
        _export_and_match(environment, report, fit_data, variants, source, args)
    with gzip.open(path, 'wt', encoding='utf-8') as output:
        json.dump(report, output, separators=(',', ':'))


def _long_check(args):
    with gzip.open(args.out / 'metric-transfer.json.gz', 'rt', encoding='utf-8') as file:
        old = json.load(file)
    original = old['cohorts'][old['selected']]['episodes']
    previous = max((row for row in original if row['termination'] == 'timeout'),
                   key=lambda row: row['longest_consecutive_rebounds'], default=None)
    parameters = args.out / 'metric-transfer-map.npz'
    arena = BounceArena(view='wrist', width=192, height=144, arm='alternate', max_steps=600)
    try:
        witness = None
        if previous is not None:
            witness = _deploy(arena, previous['seed'], args.source, parameters)
            assert witness['commands'][:len(previous['commands'])] == previous['commands']
        fresh = [_deploy(arena, seed, args.source, parameters) for seed in LONG_HELD]
        report = dict(schema='tt24-metric-transfer-long-1', held_seeds=LONG_HELD, max_steps=arena.max_steps,
                      unchanged_parameters=True, witness_first_300_commands_exact=witness is not None, witness=witness,
                      witness_prior_count=previous['longest_consecutive_rebounds'] if previous else None,
                      fresh_episodes=fresh, fresh_summary=summarize(fresh))
    finally:
        arena.close()
    with gzip.open(args.out / 'metric-transfer-long.json.gz', 'wt', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
    if witness is not None:
        print('extended timeout witness', witness['seed'], witness['longest_consecutive_rebounds'], witness['elapsed_s'], flush=True)
    print('extended independent check', report['fresh_summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/tt24'))
    parser.add_argument('--source', type=Path, default=Path('artifacts/tt24/bounce-ident-pupil.npz'))
    parser.add_argument('--supervised', type=Path, default=Path('artifacts/tt24/bounce-ident-transfer-map.npz'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--development-only', action='store_true')
    mode.add_argument('--finish-archive', action='store_true', help='verify an existing held archive and add the matched control')
    mode.add_argument('--extend-horizon', action='store_true', help='evaluate the saved public calibration for up to ten seconds')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.extend_horizon:
        _long_check(args)
        return
    prior_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    records, eyes = capture(args.source)
    source = WristPupil.load(args.source)
    arena = BounceArena(view='wrist', width=192, height=144, arm='alternate')
    try:
        name = 'metric-transfer-development.json.gz' if args.development_only else 'metric-transfer.json.gz'
        _save(arena, args.out / name, source, records, eyes, args)
    finally:
        arena.close()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == prior_sha


if __name__ == '__main__':
    main()

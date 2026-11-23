"""Estimate paddle-face changes from camera-visible impacts in played solo drops."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import _ellipse
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_metric_stereo import triangulate
from .tt24_public_body import FUSED_OUTPUT, load_public, stationary, stock
from .tt26_grip_course import Grip


OUTPUT = Path('artifacts/tt26/optical-face.json.gz')
HELD_OUTPUT = OUTPUT.with_name('optical-face-held.json.gz')
MOUNTS = {'nominal': (0., (0., 0., 1.)), 'yaw_plus': (.22, (0., 0., 1.)),
          'yaw_minus': (-.22, (0., 0., 1.)), 'pitch_plus': (.26, (1., 0., 0.))}
PRACTICE = tuple(range(262500400, 262500416))
HELD = tuple(range(262500800, 262500808))
FRAME = 1. / 60.
NEIGHBORHOOD = np.arange(-5, 6)


def unit(vector):
    return vector / np.linalg.norm(vector)


def rotation_between(first, second):
    perpendicular = np.cross(first, second)
    magnitude = np.linalg.norm(perpendicular)
    if magnitude < 1.e-8:
        return np.eye(3)
    angle = np.arctan2(magnitude, float(first @ second))
    return Rotation.from_rotvec(perpendicular * (angle / magnitude)).as_matrix()


def separation(first, second):
    return float(np.degrees(np.arccos(np.clip(first @ second, -1., 1.))))


def collect(arena, eyes, policy, seed):
    obs = arena.reset(seed=seed, split='practice')
    policy.begin_release(obs)
    pixels, joints, actions = [], [], []
    for tick in range(arena.max_steps):
        if tick < 34:
            frames = [_ellipse(rgb) for rgb in arena.fixed_images()]
            pixels.append([sight[:2] if sight is not None else np.full(2, np.nan) for sight in frames])
            joints.append(obs['joints'])
        action = policy.act(obs)
        actions.append(action.tolist())
        obs, _, done = arena.step(action)
        if done:
            break
    points, gap, _ = triangulate(eyes, np.asarray(pixels), tolerance=.018)
    metrics = arena.judge_metrics()
    return dict(seed=seed, pixels=pixels, ball=points, joint_angles=np.asarray(joints), ray_gap=gap,
                actions=actions, rebounds=metrics['rebounds'], streak=metrics['longest_consecutive_rebounds'])


def free_acceleration(releases):
    samples = []
    for drop in releases:
        p = drop['ball']
        incoming, outgoing = p[2:-2] - p[:-4], p[4:] - p[2:-2]
        a = (outgoing - incoming) / (2. * FRAME) ** 2
        use = (incoming[:, 2] * outgoing[:, 2] > 0.) & (np.linalg.norm(a, axis=1) < 25.)
        samples.extend(a[use])
    if len(samples) < 12:
        raise ValueError('too few visible free-flight samples')
    return np.median(samples, axis=0), len(samples)


def impact_candidates(drop, acceleration, decoder):
    p, q = drop['ball'], drop['joint_angles']
    link_center, rotations = decoder.links.read(q)
    blade = link_center + decoder.offset[:3] + rotations @ decoder.offset[3:]
    candidates = []
    for tick in range(5, len(p) - 5):
        positions = p[tick + NEIGHBORHOOD]
        hits = np.isfinite(positions).all(axis=1)
        if ((hits & (NEIGHBORHOOD < 0)).sum() < 3 or (hits & (NEIGHBORHOOD > 0)).sum() < 3):
            continue
        for phase in (-.5, 0., .5):
            t = (NEIGHBORHOOD[hits] - phase) * FRAME
            design = np.column_stack((np.ones(len(t)), t, np.maximum(0., t)))
            adjusted = positions[hits] - .5 * t[:, None] ** 2 * acceleration
            center, incoming, jump = motion = np.linalg.lstsq(design, adjusted, rcond=None)[0]
            outgoing = incoming + jump
            residual = float(np.sqrt(np.mean((adjusted - design @ motion) ** 2)))
            distance = float(np.linalg.norm(center - blade[tick]))
            if (incoming[2] >= -.42 or outgoing[2] <= .29 or jump[2] < 1.1 or distance > .105
                    or np.linalg.norm(jump[:2]) > .64 * jump[2] or residual > .006):
                continue
            local = rotations[tick].T @ unit(jump)
            if local @ decoder.normal < 0.:
                local = -local
            if local @ decoder.normal < .78:
                continue
            candidates.append(dict(tick=tick, sub_tick=phase, position_m=center.tolist(),
                                   incoming_m_s=incoming.tolist(), outgoing_m_s=outgoing.tolist(),
                                   velocity_jump_m_s=jump.tolist(), raw_normal_link=local.tolist(),
                                   residual_m=residual, paddle_distance_m=distance, optical_samples=int(hits.sum())))
    events = []
    for event in sorted(candidates, key=lambda row: row['residual_m']):
        if all(abs(event['tick'] - kept['tick']) >= 8 for kept in events):
            events.append(event)
    return sorted(events, key=lambda row: row['tick'])


def first_directions(releases):
    return {drop['seed']: np.array(drop['optical_impacts'][0]['raw_normal_link'])
            for drop in releases if drop['optical_impacts']}


def serialized(drop):
    result = {key: value for key, value in drop.items() if key not in ('pixels', 'ball', 'joint_angles', 'ray_gap')}
    for key in ('pixels', 'ball', 'joint_angles', 'ray_gap'):
        data = np.array(drop[key], dtype=object)
        data[~np.isfinite(np.asarray(data, dtype=float))] = None
        result[key] = data.tolist()
    return result


def fit_mounts(releases, decoder):
    acceleration, free = free_acceleration(drop for rows in releases.values() for drop in rows)
    for rows in releases.values():
        for drop in rows:
            drop['optical_impacts'] = impact_candidates(drop, acceleration, decoder)
    nominal = first_directions(releases['nominal'])
    if len(nominal) < 4:
        raise ValueError('too few observed nominal contact reversals')
    reference = unit(np.median(list(nominal.values()), axis=0))
    fitted = {}
    for mount, rows in releases.items():
        directions = first_directions(rows)
        shared = sorted(nominal.keys() & directions.keys())
        if len(shared) < 4:
            raise ValueError(f'{mount} has only {len(shared)} paired observed contact reversals')
        # First impact per release keeps differences in survival, later lateral
        # motion and spin from giving one mount disproportionate influence.
        base = unit(np.median([nominal[seed] for seed in shared], axis=0))
        changed = unit(np.median([directions[seed] for seed in shared], axis=0))
        correction = rotation_between(base, changed)
        fitted[mount] = dict(normal_link=(correction @ reference).tolist(),
                             observed_impulse_normal_link=changed.tolist(), matched_nominal_normal_link=base.tolist(),
                             rotation_from_nominal=correction.tolist(), magnitude_deg=separation(base, changed),
                             paired_first_impacts=len(shared), matched_seeds=shared,
                             observed_impacts=sum(len(drop['optical_impacts']) for drop in rows),
                             physical_rebounds=sum(drop['rebounds'] for drop in rows),
                             drops=[serialized(drop) for drop in rows])
    return fitted, acceleration, free


def held(report):
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=180)
    grip = Grip(arena._model)
    policy = CarriedWristPolicy()
    practice_normal = policy.calibration[0]['normal']
    result = dict(schema='tt26-optical-face-held-1', source_sha256=_hash(OUTPUT),
                  inputs='wrist RGB and joint encoders', seeds=HELD, mounts={})
    try:
        for mount, (angle, axis) in MOUNTS.items():
            if not angle:
                continue
            grip.set(angle, axis)
            correction = np.asarray(report['mounts'][mount]['rotation_from_nominal'])
            cases = []
            for seed in HELD:
                outcomes = {}
                for mode in ('prior', 'optical_change'):
                    obs = arena.reset(seed=seed, split='held')
                    policy.begin_release(obs)
                    policy.face_normals = {0: correction @ practice_normal} if mode == 'optical_change' else {}
                    tape = []
                    while True:
                        action = policy.act(obs)
                        tape.append(action.tolist())
                        obs, _, done = arena.step(action)
                        if done:
                            break
                    metrics = arena.judge_metrics()
                    arena.reset(seed=seed, split='held', render=False)
                    for action in tape:
                        arena.step(action, render=False)
                    if arena.judge_metrics() != metrics:
                        raise ValueError(f'held physical replay diverged on {mount}/{seed}/{mode}')
                    outcomes[mode] = dict(rebounds=metrics['rebounds'], streak=metrics['longest_consecutive_rebounds'],
                                          calibration=policy.calibration_index, metrics=metrics, actions=tape)
                cases.append(dict(seed=seed, results=outcomes))
            summary = {mode: dict(rebounds=sum(case['results'][mode]['rebounds'] for case in cases),
                                  max_streak=max(case['results'][mode]['streak'] for case in cases))
                       for mode in ('prior', 'optical_change')}
            result['mounts'][mount] = dict(summary=summary, cases=cases)
            _write(HELD_OUTPUT, result)
            print('held wrist', mount, summary, flush=True)
    finally:
        arena.close()


def run(held_only):
    if held_only:
        held(_read(OUTPUT))
        return
    started = perf_counter()
    arena = stock()
    arena.max_steps = 100
    grip = Grip(arena._model)
    policy = CarriedWristPolicy()
    decoder = load_public(FUSED_OUTPUT).decoder
    gathered = {}
    try:
        arena.reset(seed=PRACTICE[0], render=False)
        eyes = stationary(arena)
        for mount, (angle, axis) in MOUNTS.items():
            grip.set(angle, axis)
            gathered[mount] = [collect(arena, eyes, policy, seed) for seed in PRACTICE]
            print('captured', mount, len(gathered[mount]), sum(row['rebounds'] for row in gathered[mount]), flush=True)
    finally:
        arena.close()
    mounts, acceleration, free = fit_mounts(gathered, decoder)
    # The evaluator mount changes are revealed only after the fit. The nominal
    # quaternion is used here strictly for face error diagnostics.
    truth = Rotation.from_quat(np.roll(grip.original, -1)).apply((0., 0., 1.))
    if truth @ decoder.normal < 0.:
        truth = -truth
    for mount, (angle, axis) in MOUNTS.items():
        row = mounts[mount]
        actual = Rotation.from_rotvec(angle * np.asarray(axis)).apply(truth)
        predicted = np.asarray(row['rotation_from_nominal']) @ truth
        row.update(evaluator_mount_angle_rad=angle, evaluator_mount_axis=axis,
                   evaluator_face_change_deg=separation(truth, actual),
                   evaluator_rotation_error_deg=separation(actual, predicted))
        print('face fit', mount, row['paired_first_impacts'], round(row['magnitude_deg'], 2),
              round(row['evaluator_rotation_error_deg'], 2), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write(OUTPUT, dict(schema='tt26-optical-face-1', mounts=mounts, practice_seeds=PRACTICE, held_seeds=HELD,
                        source_policy_sha256=_hash(BUNDLE), nominal_decoder_sha256=_hash(FUSED_OUTPUT),
                        inputs='stationary calibrated stereo RGB ball positions and public SO-101 joint encoders',
                        source_policy_inputs='wrist RGB and joint encoders',
                        estimator='minimum rotation between nominal and changed median first ball impulse directions on paired releases',
                        frame='local nominal gripper / welded paddle body frame',
                        free_flight_acceleration_m_s2=acceleration.tolist(), free_flight_samples=free,
                        practice_cap=arena.max_steps, seconds=perf_counter() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held', action='store_true', help='apply the frozen relative fit in separate wrist-only play')
    run(parser.parse_args().held)

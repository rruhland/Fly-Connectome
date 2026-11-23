"""Refit a frozen motor policy's wrist observer from generic public RGB probes."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _unit_camera_ray, summarize
from .tt24_bounce_ident_transfer import _verify_tape
from .tt24_metric_stereo import triangulate
from .tt24_predict_reference import HEIGHT, WIDTH, TrainingEyes
from .tt24_public_body import (DIRECTIONS, FOCAL, FUSED_OUTPUT, RIM, ConsensusWrist,
                               PublicDecoder, PublicMotion, VendorLinks, blade_profile,
                               body_fit, load_public, motion_fit, probe, shared_normal,
                               sight, stationary)
from .tt24_world_arm import PUBLIC, SOURCE, perception


ARTIFACTS = Path('artifacts/tt24')
OUTPUT = ARTIFACTS / 'public-morph.npz'
FIT_RECEIPT = ARTIFACTS / 'public-morph-calibration.json.gz'
EVALUATION = ARTIFACTS / 'public-morph.json.gz'
PROBE_SEEDS = tuple(range(406201, 406205))
HELD_SEEDS = tuple(range(406401, 406417))
STOCK_RETURN = tuple(range(406601, 406605))
HORIZON = 450


class ProbeEyes(BounceArena):
    """The original calibrated pair observes the alternate arm during probe collection."""
    fixed_images = TrainingEyes.fixed_images

    def __init__(self):
        super().__init__(view='wrist', width=WIDTH, height=HEIGHT, max_steps=72, arm='alternate')
        prototype = TrainingEyes()
        self.eyes = prototype.eyes
        prototype.close()
        self._model.vis.global_.fovy = 23.
        for eye, yaw, tilt in zip(self.eyes, (12., 102.), (-36., -33.), strict=True):
            eye.lookat[:] = (0., -.94, .93)
            eye.distance = 1.48
            eye.azimuth = yaw
            eye.elevation = tilt


def collect(seeds):
    """The fitter records only RGB summaries, encoders and its own probe commands."""
    arena = ProbeEyes()
    records = []
    try:
        for seed in seeds:
            obs = arena.reset(seed=seed, split='held', regime='changed')
            if not records:
                eyes = stationary(arena)
            home = obs['joints'].copy()
            trace = []
            done = False
            while not done:
                command = probe(seed, len(trace), home, obs['joints'])
                fixed = arena.fixed_images()
                trace.append((obs['joints'].copy(), sight(obs['rgb']),
                              [sight(frame) for frame in fixed],
                              [blade_profile(frame) for frame in fixed], command))
                obs, _, done = arena.step(command)
            enc, wrist, fixed, blade, actions = (np.asarray(value) for value in zip(*trace, strict=True))
            record = dict(seed=seed, enc=enc, wrist=wrist, fixed=fixed, blade=blade, commands=actions)
            records.append(record)
            ending = arena.judge_metrics()
            _verify_tape(arena, ending | dict(commands=actions.tolist()), 'changed')
            print('probe', seed, len(trace), flush=True)
    finally:
        arena.close()
    return eyes, records


class OpticalLinks(VendorLinks):
    """Stock transforms with small shifts and axis tilts inferred from public imagery."""

    def __init__(self, shifts, tilts):
        super().__init__()
        self.shifts, self.tilts = shifts, tilts
        axis = np.c_[tilts, np.ones(len(tilts))]
        self.axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)

    def read(self, q):
        position = np.zeros((len(q), 3))
        frame = np.broadcast_to(self.basis, (len(q), 3, 3)).copy()
        for index, ((shift, neutral), angle) in enumerate(zip(self.links, q.T, strict=True)):
            position += frame @ (shift + self.shifts[index])
            turning = Rotation.from_rotvec(angle[:, None] * self.axis[index]).as_matrix()
            frame = frame @ (neutral @ turning)
        return position, frame


def blade_fit(records, eyes):
    """Fit the full painted support across motion, with stock geometry as a prior."""
    offset, normal, initial = body_fit(records, eyes)
    normal, width = shared_normal(records, eyes, offset, normal, widths=True)
    enc = np.concatenate([r['enc'][::2] for r in records])
    image = np.concatenate([r['blade'][::2, :, 2:] for r in records])
    visible = np.isfinite(image).all(axis=(1, 2))
    enc, image = enc[visible], image[visible]
    phase = np.arctan2(normal[1], normal[0])
    # The first link offset is indistinguishable from platform translation.
    start = np.r_[offset, phase, width['radius_m'], width['half_thickness_m'], np.zeros(22)]
    limit = np.r_[[.035] * 6, .44, .018, .012, [.030] * 12, [.22] * 10]
    prior = np.r_[[.022] * 6, .22, .006, .006, [.009] * 12, [.085] * 10]
    lower, upper = start - limit, start + limit
    lower[7:9] = [.047, 0.]
    upper[7:9] = [.103, .016]
    def decode(params):
        shifts = np.r_[np.zeros(3), params[9:21]].reshape(5, 3)
        links = OpticalLinks(shifts, params[21:31].reshape(5, 2))
        normal = np.r_[np.cos(params[6]), np.sin(params[6]), 0.]
        return links, params[:6], normal

    def image_error(params):
        links, mount, direction = decode(params)
        pos, rot = links.read(enc)
        n = rot @ direction
        center = pos + mount[:3] + rot @ mount[3:]
        tangent = np.cross(n, (0., 1., 0.))
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        rim = center[:, None, :] + params[7] * (RIM @ np.stack((tangent, np.cross(n, tangent)), axis=1))
        breadth = params[8] * n[:, None, :]
        outline = np.concatenate((rim + breadth, rim - breadth), axis=1)
        projected = []
        for eye in eyes:
            px, _ = eye.project(outline.reshape(-1, 3))
            support = (px.reshape(len(outline), -1, 2) @ DIRECTIONS.T).max(axis=1)
            projected.append(support)
        return np.stack(projected, axis=1) - image

    def loss(params):
        residual = image_error(params)
        half = len(DIRECTIONS) // 2
        # Opposing supports form a size/attitude component and a center component.
        breadth = .5 * (residual[..., :half] + residual[..., half:])
        center = .5 * (residual[..., :half] - residual[..., half:])
        return np.r_[(breadth * 1.35).ravel(), center.ravel(), 2. * (params - start) / prior]

    params = least_squares(loss, start, bounds=(lower, upper), loss='soft_l1',
                           f_scale=.62, max_nfev=52).x
    links, mount, normal = decode(params)
    return links, mount, normal, dict(initial=initial, initial_width=width, silhouette_frames=len(enc),
                                     initial_support_rmse_px=float(np.sqrt(np.mean(image_error(start)**2))),
                                     final_support_rmse_px=float(np.sqrt(np.mean(image_error(params)**2))),
                                     radius_m=float(params[7]), half_thickness_m=float(params[8]))


def register_ball(links, mount, records, eyes):
    q = np.concatenate([r['enc'] for r in records])
    stereo = np.concatenate([r['fixed'] for r in records])
    wrist = np.concatenate([r['wrist'] for r in records])
    points, gap, _ = triangulate(eyes, stereo[:, :, :2], tolerance=.028)
    visible = np.isfinite(wrist).all(axis=1) & np.isfinite(points).all(axis=1)
    if visible.sum() < 24:
        raise ValueError('insufficient joint wrist and fixed image sightings')
    pos, rot = links.read(q[visible])
    expected = np.einsum('nij,ni->nj', rot, points[visible] - pos - mount[:3])
    optical = _unit_camera_ray(wrist[visible], FOCAL)
    centered_optical = optical - optical.mean(axis=0)
    u, s, vt = np.linalg.svd(centered_optical.T @ (expected - expected.mean(axis=0)))
    signs = np.r_[1., 1., np.linalg.det(vt.T @ u.T)]
    rig = vt.T @ np.diag(signs) @ u.T
    radius = np.clip((s @ signs) / np.sum(centered_optical**2), .012, .032)
    offset = np.mean(expected - radius * optical @ rig.T, axis=0)
    strength = np.sqrt(1. / (1. + (gap[visible] / .008)**2))

    def residual(params):
        estimate = Rotation.from_rotvec(params[:3]).apply(optical * params[-1]) + params[3:6]
        return ((estimate - expected) * strength[:, None]).ravel()

    start = np.r_[Rotation.from_matrix(rig).as_rotvec(), offset, radius]
    best = least_squares(residual, start, loss='soft_l1', f_scale=.006, max_nfev=75)
    error = np.linalg.norm(residual(best.x).reshape(-1, 3), axis=1)
    return best.x, points, dict(paired_ball_frames=int(visible.sum()), metric_radius_m=float(best.x[-1]),
                                median_rgb_registration_m=float(np.median(error)))


class MorphDecoder(PublicDecoder):
    def __init__(self, links, offset, normal, registration, default):
        super().__init__(offset, normal, registration, default)
        self.links = links


def contact_directions(links, decoder, records, points):
    """Estimate face direction from large, spatially nearby RGB velocity reversals."""
    sequences = np.split(points, np.cumsum([len(r['enc']) for r in records[:-1]]))
    free = []
    for ball in sequences:
        velocity = np.diff(ball, axis=0)
        jump = np.diff(velocity, axis=0) * 3600.
        acceptable = (velocity[:-1, 2] * velocity[1:, 2] > 0.) & (np.linalg.norm(jump, axis=1) < 28.)
        free.extend(jump[acceptable])
    if not free:
        return [], []
    acceleration = np.median(free, axis=0)
    directions, samples = [], []
    for record, ball in zip(records, sequences, strict=True):
        _, rotations = links.read(record['enc'])
        blade = np.array([decoder.body(q)[12:15] for q in record['enc']])
        found = []
        for tick in range(3, len(ball)-3):
            before = (ball[tick-1] - ball[tick-3]) * 30
            after = (ball[tick+3] - ball[tick+1]) * 30
            impulse = after - before - acceleration * (4. / 60.)
            nearby = np.linalg.norm(ball[tick] - blade[tick]) < .12
            if not (nearby and np.isfinite(impulse).all() and before[2] < -.37 and after[2] > .30
                    and impulse[2] > 1.05 and np.linalg.norm(impulse[:2]) < .60 * impulse[2]):
                continue
            local = rotations[tick].T @ (impulse / np.linalg.norm(impulse))
            if local @ decoder.normal < 0:
                local = -local
            found.append((float(impulse[2]), tick, local))
        selected = []
        for strength, tick, local in sorted(found, key=lambda event: -event[0]):
            if all(abs(tick-old) >= 7 for old in selected):
                selected.append(tick)
                directions.append(local)
                samples.append(dict(seed=record['seed'], tick=tick, strength=strength, local=local.tolist()))
    return directions, samples


def calibrate(path, receipt):
    eyes, records = collect(PROBE_SEEDS)
    links, mount, normal, blade = blade_fit(records, eyes)
    registration, points, optics = register_ball(links, mount, records, eyes)
    decoder = MorphDecoder(links, mount, normal, registration, np.zeros(3))
    directions, contacts = contact_directions(links, decoder, records, points)
    if directions:
        joined = np.median(directions, axis=0)
        joined[2] = 0.
        if np.linalg.norm(joined) > 0.:
            n = len(directions) * joined / np.linalg.norm(joined) + 2. * normal
            decoder.normal = n / np.linalg.norm(n)
    starts = np.r_[0, np.cumsum([len(row['enc']) for row in records[:-1]])]
    origins = np.array([decoder.body(row['enc'][0])[12:15] for row in records])
    default = np.nanmedian(points[starts] - origins, axis=0)
    if not np.isfinite(default).all():
        raise ValueError('no stereo support for initial ball location')
    decoder.default = default
    motion, samples = motion_fit(decoder, records, points, observer=2)
    if samples < 16 or not np.isfinite(motion.weights).all():
        raise ValueError('too few public motion samples')
    shifts, tilts = links.shifts, links.tilts
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema='tt24-public-morph-1', shifts=shifts, tilts=tilts, offset=mount,
                        unfused_normal=normal, normal=decoder.normal, registration=registration, default=default,
                        motion_mean=motion.mean, motion_scale=motion.scale, motion_weights=motion.weights)
    _write(receipt, dict(schema='tt24-public-morph-fit-1', arm='alternate', regime='changed',
                         prior_geometry='original stock vendor CAD with optically fitted corrections',
                         labels=['calibrated fixed RGB', 'wrist RGB', 'own encoders'],
                         new_teacher_actions=0, target_rewards_used=0, hidden_pose_labels=0,
                         checkpoint_sha256=_hash(path), motion_samples=samples, optics=optics,
                         blade=blade, contacts=contacts, normal=decoder.normal.tolist(),
                         link_shifts_m=shifts.tolist(), link_axis_slope=tilts.tolist(),
                         data=[dict(seed=row['seed'], ticks=len(row['enc']), commands=row['commands'].tolist())
                               for row in records]))
    print('frozen observer', path, optics, 'motion samples', samples, 'contacts', len(contacts), flush=True)


def load_morph(path=OUTPUT):
    """Start a fresh observer that needs only wrist RGB and joint positions."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if checkpoint['schema'].item() != 'tt24-public-morph-1':
            raise ValueError('unrecognized morphology checkpoint')
        links = OpticalLinks(checkpoint['shifts'], checkpoint['tilts'])
        decoder = MorphDecoder(links, checkpoint['offset'], checkpoint['normal'], checkpoint['registration'],
                               checkpoint['default'])
        motion = PublicMotion(checkpoint['motion_mean'], checkpoint['motion_scale'], checkpoint['motion_weights'])
    return ConsensusWrist(decoder, motion)


def play(arena, seed, regime, factory):
    pupil = None if factory is None else WristPupil.load(SOURCE)
    if pupil is not None:
        pupil.sensor = factory()
    obs = arena.reset(seed=seed, split='held', regime=regime, render=pupil is not None)
    trace = []
    done = False
    while not done:
        command = np.zeros(arena.action_size) if pupil is None else pupil.command(obs)
        trace.append(command.tolist())
        obs, _, done = arena.step(command, render=pupil is not None)
    row = arena.judge_metrics() | dict(commands=trace)
    _verify_tape(arena, row, regime)
    tracker = pupil.sensor if pupil is not None else None
    sighting = dict(gaps=getattr(tracker, 'gaps', None), rejections=getattr(tracker, 'rejections', None))
    return row | dict(tracking=sighting)


def measure(path, receipt, result):
    from .tt4_transfer_trial import _read

    fit = _read(receipt)
    if fit['checkpoint_sha256'] != _hash(path):
        raise ValueError('fit receipt differs from frozen calibration')
    if set(HELD_SEEDS + STOCK_RETURN) & {row['seed'] for row in fit['data']}:
        raise ValueError('a calibration drop appears in the evaluation')
    files = (SOURCE, FUSED_OUTPUT, PUBLIC, path)
    before = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    alternate = BounceArena(view='wrist', width=WIDTH, height=HEIGHT, max_steps=HORIZON, arm='alternate')
    stock = BounceArena(view='wrist', width=WIDTH, height=HEIGHT, max_steps=HORIZON)
    stock_observer = lambda: load_public(FUSED_OUTPUT)
    arms = dict(recalibrated_public=lambda: load_morph(path), frozen_stock_public=stock_observer,
                prior_alternate_optics=lambda: perception(True), stationary=None)
    try:
        stock_before = [play(stock, seed, 'standard', stock_observer) for seed in STOCK_RETURN]
        modes = {}
        for name, sensor in arms.items():
            rows = [play(alternate, seed, 'changed', sensor) for seed in HELD_SEEDS]
            modes[name] = dict(summary=summarize(rows), episodes=rows)
            print(name, modes[name]['summary'], flush=True)
        stock_after = [play(stock, seed, 'standard', stock_observer) for seed in STOCK_RETURN]
    finally:
        stock.close()
        alternate.close()
    after = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    if stock_before != stock_after or before != after:
        raise ValueError('a calibration or source checkpoint changed')
    _write(result, dict(schema='tt24-public-morph-qualification-1', fit_sha256=_hash(receipt),
                        checkpoints=before, arm='alternate', regime='changed', horizon=HORIZON,
                        observation='wrist RGB and encoders only', actor='frozen stock demonstrated motor head',
                        prior_alternate_optics='existing metric RGB residual on privileged-trained stock pupil',
                        held_seeds=HELD_SEEDS, modes=modes, exact_action_replay=True,
                        stock_return=dict(seeds=STOCK_RETURN, exact_before_after=True,
                                          summary=summarize(stock_after), episodes=stock_after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--calibration', type=Path, default=FIT_RECEIPT)
    parser.add_argument('--report', type=Path, default=EVALUATION)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--evaluate-only', action='store_true')
    stage.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    if not args.evaluate_only:
        calibrate(args.output, args.calibration)
    if not args.fit_only:
        measure(args.output, args.calibration, args.report)


if __name__ == '__main__':
    main()

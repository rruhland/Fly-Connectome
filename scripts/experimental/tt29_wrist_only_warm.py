"""Compare nominal wrist calibration with a fit from physical solo drops on the same mount."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .tt4_arena import BALL_RADIUS
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import WRIST_CAMERA
from .tt24_metric_stereo import triangulate
from .tt24_predict_reference import TrainingEyes
from .tt24_public_body import stationary
from .tt29_rally_physics import measured_launches
from .tt29_vision import EncoderCamera, FlightPractice, Lens, Rig, ball_pixels
from .tt29_wrist_only import APERTURE, BRACKET, OPPONENT, WristPartner, WristReceiver, capture_player, load_memory, mount


OUT = Path('artifacts/tt29')
DROP_SOURCE = OUT / 'tt29_bounce_course.json.gz'
MOTOR_SOURCE = OUT / 'camera-action-memory.json.gz'
CALIBRATION = OUT / 'tt29_wrist_only_solo_fit.json'
TRANSFER = OUT / 'tt29_wrist_only_warm.json.gz'


class SoloBoard(TrainingEyes):
    """Mount the same board on the stock clamped arm in the separate solo area."""

    def __init__(self, *, flipped=False):
        self.flipped = flipped
        super().__init__(fixture='table-edge')
        self._model.vis.global_.fovy = 23.
        for eye, azimuth, elevation in zip(self.eyes, (12., 102.), (-36., -33.), strict=True):
            eye.lookat[:] = (0., -.574, 1.1844)
            eye.distance = 1.48
            eye.azimuth = azimuth
            eye.elevation = elevation

    def _actuated_pose(self, pose):
        q = np.array(pose, dtype=float)
        if self.flipped:
            q[4] += np.pi if q[4] < 0. else -np.pi
        return q

    def _document(self, width, height, detail, arm):
        return mount(super()._document(width, height, detail, arm), APERTURE, *BRACKET)


def coordinates(camera):
    """Convert CameraFit's right/down/forward convention to the local camera rotation."""
    return camera.rotation.T @ np.diag([1., -1., -1.])


def local_board(camera):
    nominal = camera.at(np.zeros(6))
    gripper = camera.remount(np.zeros(3), np.eye(3)).at(np.zeros(6))
    g = coordinates(gripper)
    return g.T @ (nominal.eye - gripper.eye), g.T @ coordinates(nominal)


def gather(arena, camera, drops, horizon):
    arena.reset(seed=drops[0]['seed'], render=False)
    external = stationary(arena)
    center = (np.array((arena.width, arena.height)) - 1.) * .5
    gripper = camera.remount(np.zeros(3), np.eye(3))
    packets = []
    rows = []
    for i, drop in enumerate(drops):
        obs = arena.reset(seed=drop['seed'], regime=drop['regime'])
        accepted = 0
        for tick, command in enumerate(drop['actions'][:horizon]):
            sight = ball_pixels(obs['rgb'])
            if sight is not None and sight.radius > 2.1 and not sight.clipped:
                reads = [ball_pixels(image) for image in arena.fixed_images()]
                if all(read is not None and not read.clipped for read in reads):
                    fixed = np.stack([(read.center - center) * [1, -1] for read in reads])
                    ball, _, _ = triangulate(external, fixed[None], tolerance=.010)
                    if np.isfinite(ball[0]).all():
                        public = gripper.at(np.r_[obs['joints'], -.1])
                        local = coordinates(public).T @ (ball[0] - public.eye)
                        packets.append((i, local, sight.center, sight.radius))
                        accepted += 1
            obs, _, done = arena.step(command)
            if done:
                break
        reward = arena.judge_metrics()
        rows.append(dict(seed=drop['seed'], regime=drop['regime'], observations=accepted,
                         rebounds=reward['rebounds'], longest=reward['longest_consecutive_rebounds'],
                         steps=reward['steps']))
        print('solo board', i + 1, accepted, reward['rebounds'], flush=True)
    return packets, rows


def calibration(feeds, output, count, horizon, *, flipped=False):
    arena = SoloBoard(flipped=flipped)
    # A separate construction supplies the common nominal drawing to both branches.
    nominal = SoloBoard()
    try:
        lens = Lens(nominal.width, nominal.height, float(nominal._model.camera(WRIST_CAMERA).fovy[0]))
        camera = EncoderCamera.from_model(nominal._model, lens)
        base_pos, base_rotation = local_board(camera)
        nominal_params = np.r_[base_pos, Rotation.from_matrix(base_rotation).as_rotvec()]
        samples, rows = gather(arena, camera, _read(feeds)['practice'][:count], horizon)
        fit_rows = [row for row in samples if row[0] % 4 != 3]
        held_rows = [row for row in samples if row[0] % 4 == 3]
        if len(fit_rows) < 24 or len(held_rows) < 5:
            receipt = dict(status='insufficient native paired solo sightings', mount=BRACKET, horizon=horizon, flipped=flipped,
                           source_sha256=_hash(feeds), train=len(fit_rows), held=len(held_rows), drops=rows)
            _write(output, receipt)
            return receipt
        center = np.array(((lens.width - 1) / 2, (lens.height - 1) / 2))

        def project(params, points):
            local = Rotation.from_rotvec(params[3:]).inv().apply(points - params[:3])
            projected = center + lens.focal * local[:, :2] * [1., -1.] / np.maximum(.02, -local[:, 2:])
            return projected, local

        def unpack(data):
            _, position, pixel, radius = zip(*data, strict=True)
            return np.stack(position), np.stack(pixel), np.asarray(radius)

        def metrics(params, data):
            p, pixels, radius = unpack(data)
            reproj, local = project(params, p)
            distance = np.linalg.norm(local, axis=1)
            off_axis = distance / np.maximum(.025, -local[:, 2])
            area_factor = BALL_RADIUS * lens.focal * off_axis ** 1.5 / (distance * radius)
            residual = np.linalg.norm(reproj - pixels, axis=1)
            return dict(median_px=float(np.median(residual)), radius_factor=float(np.clip(np.median(area_factor), .83, 1.2)),
                        p90_px=float(np.quantile(residual, .90)), samples=len(p))

        points, pixels, _ = unpack(fit_rows)

        def residual(params):
            computed, _ = project(params, points)
            return np.r_[(computed - pixels).ravel(), (params - nominal_params) / [.010] * 1.7]

        fit = least_squares(residual, nominal_params, loss='soft_l1', f_scale=.6, max_nfev=50)
        warm, cold = metrics(fit.x, held_rows), metrics(nominal_params, held_rows)
        learned = metrics(fit.x, fit_rows)
        receipt = dict(status='fitted', source_sha256=_hash(feeds), mount=BRACKET, horizon=horizon, drops=rows, flipped=flipped,
                       holdout_policy='entire fourth drop in each block held out', focal_px=lens.focal,
                       height=lens.height, prior_position=nominal_params[:3].tolist(), prior_rotvec=nominal_params[3:].tolist(),
                       fitted_position=fit.x[:3].tolist(), fitted_rotvec=fit.x[3:].tolist(),
                       observations={'fit': learned, 'held_nominal': cold, 'held_fitted': warm},
                       training_observations=len(fit_rows), held_observations=len(held_rows))
        _write(output, receipt)
        return receipt
    finally:
        nominal.close()
        arena.close()


def transfer(feeds, source, motor, output, cases, *, target):
    checkpoint = _read(source)
    if checkpoint['status'] != 'fitted':
        raise ValueError('solo capture did not support a warm calibration')
    lookup = measured_launches(feeds)
    optics = dict(board=BRACKET, strike_ready='lateral', width=640, height=480, max_steps=1200,
                  clearance_guard=True, fast_search=True, clearance_margin=.003, robust_contact=True)
    partner = dict(targets=OPPONENT, guard_table=True)
    twin = WristReceiver(**optics)
    cad = WristPartner(**optics, **partner, detail='cad')
    arena = WristPartner(**optics, **partner)
    outcomes = dict(schema='tt29-matched-solo-wrist-transfer-1', source_sha256=_hash(feeds),
                    solo_checkpoint_sha256=_hash(source), motor_source_sha256=_hash(motor),
                    streams=['wrist'], cases=list(cases), matched_target=target, rows=[],
                    cold_calibration='independent nominal mount drawing shared by both receivers',
                    warm_calibration='only fit from separate physical solo drops')
    try:
        reference = Rig.from_arena(twin).wrist
        warm = reference.remount(checkpoint['fitted_position'], Rotation.from_rotvec(checkpoint['fitted_rotvec']).as_matrix(),
                                  focal_px=checkpoint['focal_px'], reference_height=checkpoint['height'])
        for sid in cases:
            for label, eye in (('nominal', reference), ('solo', warm)):
                flight = FlightPractice()
                if label == 'solo':
                    fit = checkpoint['observations']['fit']['radius_factor']
                    flight.radius_samples['wrist'].extend([fit] * 8)
                row = capture_player(arena, twin, cad.model, lookup[sid], course=flight, memory=load_memory(motor),
                                     confidence=.112, scan=False, seed=sid, max_rally=target, min_frames=7,
                                     refine=True, camera=eye)
                outcomes['rows'].append(dict(calibration=label, **row))
                _write(output, outcomes)
                print('transfer', sid, label, row['summary']['returns'], row['CAD'], flush=True)
    finally:
        arena.close()
        cad.close()
        twin.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path, nargs='?')
    parser.add_argument('--drops', type=Path, default=DROP_SOURCE)
    parser.add_argument('--fit', type=Path, default=CALIBRATION)
    parser.add_argument('--drop-count', type=int, default=12)
    parser.add_argument('--drop-frames', type=int, default=38)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--solo-flip', action='store_true', help='bounce on the opposite blade face')
    parser.add_argument('--reuse-fit', action='store_true')
    parser.add_argument('--motor', type=Path, default=MOTOR_SOURCE)
    parser.add_argument('--output', type=Path, default=TRANSFER)
    parser.add_argument('--ids', nargs='+', type=int, default=(7271,))
    parser.add_argument('--target', type=int, default=12)
    args = parser.parse_args()
    if args.feeds is None and not args.fit_only:
        parser.error('a measured launch file is required for the table comparison')
    if args.drop_count < 4 or args.drop_frames < 4 or args.target < 1:
        parser.error('specify at least four drops, four exposures and a positive return target')
    args.fit.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_fit:
        print(calibration(args.drops, args.fit, args.drop_count, args.drop_frames,
                          flipped=args.solo_flip)['status'], flush=True)
    if not args.fit_only:
        transfer(args.feeds, args.fit, args.motor, args.output, args.ids, target=args.target)

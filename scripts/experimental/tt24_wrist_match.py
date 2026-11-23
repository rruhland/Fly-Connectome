"""Transfer a frozen rally law through an encoder-grounded, wrist-only observer."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np
from scipy.optimize import least_squares

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import GRIP_BODY, WRIST_CAMERA
from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, REPLICATION, SequenceArena, course, verify
from .tt24_rally_bank import ARCHIVE, HOLDOUT
from .tt24_rally_improve import RIG, triangulate
from .tt24_rally_improve_combined import DATA as AUGMENTED, FRESH as COMBINED_STARTS, PhaseExtent
from .tt24_rally_improve_optical import CACHE as SOURCE, OpticalPolicy, component
from .tt24_rally_improve_phase import DEV as PHASE_CLIPS, PhasePolicy
from .tt24_rally_improve_recovery import HELD as RECOVERY_CLIPS
from .tt24_rally_improve_trial import SCHEDULES
from .tt24_rally_trial import COACHED, OUTPUT, Station, returns, summary


OPTICAL_CLIPS = OUTPUT / 'rally-improve-optical-held.json.gz'
SHOTS = ((PHASE_CLIPS, 'rate_blend', 146477391), (PHASE_CLIPS, 'rate_blend', 1682479719),
         (OPTICAL_CLIPS, 'interaction', 2091841011), (RECOVERY_CLIPS, 'corrected', 1041545090))
FRESH = {'u': 1025412171, 'v': 636397642}
DIMS = (320, 240)
CLIP_LENGTH = 96
DATA = OUTPUT / 'wrist-match-calibration.npz'
CALIBRATION = OUTPUT / 'wrist-match-calibration.json.gz'
OBSERVATION_AUDIT = OUTPUT / 'wrist-match-observation-audit.json.gz'
RECEIPT = OUTPUT / 'wrist-match-held.json.gz'
MODES = ('stereo', 'wrist', 'wrist_delay', 'hold')
TRIAL = None


class EncoderRig:
    """Use only static shape parameters and a fresh data buffer for encoder FK."""

    def __init__(self):
        fixture = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
        self.model = fixture.model
        self.data = mujoco.MjData(self.model)
        self.joints = fixture._arm_ids[1][0].copy()
        self.camera = self.model.camera(WRIST_CAMERA).id
        self.grip = self.model.body(GRIP_BODY).id
        lens = fixture.lens_intrinsics()['wrist']
        self.focal = lens['focal'] * 480/DIMS[0]
        self.center = np.array((239.5, 179.5))
        fixture.close()

    def pose(self, encoders):
        self.data.qpos[self.joints] = encoders
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        origin = self.data.cam_xpos[self.camera] - [0., 0., TABLE_HEIGHT]
        attitude = self.data.cam_xmat[self.camera].reshape(3, 3).copy()
        grip_origin = self.data.xpos[self.grip] - [0., 0., TABLE_HEIGHT]
        grip_rotation = self.data.xmat[self.grip].reshape(3, 3).copy()
        return origin, attitude, grip_origin, grip_rotation

    def bearing(self, pixels):
        sight = (pixels - self.center)/self.focal
        return np.r_[sight[0], -sight[1], -1.]


def articles():
    for path, name, seed in SHOTS:
        report = _read(path)
        if not report['verified_tapes']:
            raise ValueError('calibration requires strictly verified game packets')
        row = next(case['results'][name] for case in report['cases'] if case['seed'] == seed)
        yield path, row


def collect(path, report_path):
    arena = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
    arena.streams = ('wrist', 'left', 'right')
    rig = EncoderRig()
    labels, clips = [], []
    try:
        for index, (archive, row) in enumerate(articles()):
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            for near, far in zip(row['actions'][:CLIP_LENGTH], row['rival_actions'][:CLIP_LENGTH], strict=True):
                images = arena.render_cameras()
                wrist = component(images['wrist'], False)
                fixed = np.asarray([[component(images[key], blade) for key in ('left', 'right')]
                                    for blade in (False, True)])
                origin, attitude, grip, rotation = rig.pose(arena.joints)
                points = [triangulate(reads[0, :2], reads[1, :2]) for reads in fixed]
                labels.append(dict(wrist=wrist, points=np.asarray(points), origin=origin, attitude=attitude,
                                   grip=grip, rotation=rotation, clip=index))
                command = np.asarray(far)
                arena._aim = arena._command_to = lambda *args, command=command: command
                _, _, done, cap, _ = arena.step(near, render=False)
                if done or cap:
                    break
            expected = [event for event in row['events'] if event['step'] <= arena.steps]
            if digest_json(expected) != digest_json(arena.events):
                raise ValueError('camera calibration replay changed the physical score prefix')
            clips.append(dict(seed=row['seed'], frames=arena.steps, source=archive.name, sha256=_hash(archive)))
    finally:
        arena.close()
    packed = {key: np.asarray([row[key] for row in labels]) for key in labels[0]}
    np.savez_compressed(path, **packed)
    with np.load(path) as data:
        metrics = fit(data)[1]
    _write(report_path, dict(clips=clips, paired_frames=len(labels), dimensions=list(DIMS),
                             motor_labels_used=False, fit=metrics, sha256=_hash(path)))


def design(wrist):
    u = (wrist[..., 0]-239.5)/170.
    v = (wrist[..., 1]-179.5)/170.
    inv = 4./np.maximum(wrist[..., 2], 1.3)
    return np.stack((inv, np.ones_like(inv), inv*u*u, inv*v*v, inv*u, inv*v), axis=-1)


def fit(data):
    points, wrist = data['points'], data['wrist']
    local = np.einsum('nij,ni->nj', data['attitude'], points[:, 0]-data['origin'])
    depth = -local[:, 2]
    visible = np.isfinite(wrist).all(axis=1) & np.isfinite(local).all(axis=1) & (depth > .25) & (depth < 3.)
    # Reject clipped silhouettes, which understate size and bias the centroid.
    radius = wrist[:, 2]
    visible &= ((wrist[:, 0] > radius+1) & (wrist[:, 0] < 478-radius)
                & (wrist[:, 1] > radius+1) & (wrist[:, 1] < 358-radius))
    if int(visible.sum()) < 24:
        raise ValueError('insufficient synchronized wrist and outside ball detections')
    x = design(wrist[visible])
    measured = depth[visible]
    start = np.array((np.median(measured/x[:, 0]), 0., 0., 0., 0., 0.))
    prior = np.array((.15, .60, 1.4, 1.4, 1.1, 1.1))
    rd = least_squares(lambda w: np.r_[x@w-measured, prior*(w-start)], start, loss='soft_l1', f_scale=.025)
    rig = EncoderRig()
    nominal = np.asarray([rig.bearing(photo[:2]) for photo in wrist[visible]])
    actual = local[visible]/measured[:, None]
    bearing = np.median(actual[:, :2]-nominal[:, :2], axis=0)
    blade = points[:, 1] - data['grip']
    tools = np.einsum('nij,ni->nj', data['rotation'], blade)
    local_blade = np.nanmedian(tools, axis=0)
    if not np.isfinite(local_blade).all():
        raise ValueError('no paired visual paddle measurements')
    depth_error = np.abs(x@rd.x-measured)
    error = np.linalg.norm((nominal+np.r_[bearing, 0.])*(x@rd.x)[:, None]-local[visible], axis=1)
    return (rd.x, bearing, local_blade), dict(paired_ball=int(visible.sum()), paired_blade=int(np.isfinite(tools).all(axis=1).sum()),
                nominal_error_m=float(np.median(np.linalg.norm(nominal*(x@start)[:, None]-local[visible], axis=1))),
                fitted_depth_median_error_m=float(np.median(depth_error)),
                fitted_position_median_error_m=float(np.median(error)), wrist_focal=rig.focal,
                depth_weights=rd.x.tolist(), bearing_offset=bearing.tolist(), blade_local=local_blade.tolist())


def observation_audit(path, out):
    """Predict simultaneous optical labels in each clip from the other three."""
    rig = EncoderRig()
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    rows = []
    for clip in np.unique(arrays['clip']):
        train = arrays['clip'] != clip
        depth, shift, _ = fit({key: value[train] for key, value in arrays.items()})[0]
        index = ~train
        target, observed = arrays['points'][index, 0], arrays['wrist'][index]
        valid = np.isfinite(target).all(axis=1) & np.isfinite(observed).all(axis=1)
        px = observed[valid]
        origin, attitude = arrays['origin'][index][valid], arrays['attitude'][index][valid]
        local = np.asarray([rig.bearing(x[:2]) for x in px])+np.r_[shift, 0.]
        distance = np.clip(design(px)@depth, .03, 3.)
        projected = origin+np.einsum('nij,nj->ni', attitude, local*distance[:, None])
        old = target[valid]
        error = np.linalg.norm(projected-old, axis=1)
        rows.append(dict(clip=int(clip), frames=int(index.sum()), wrist_detections=int(np.isfinite(observed).all(axis=1).sum()),
                         paired=int(valid.sum()), metric_error_median_m=float(np.median(error)),
                         metric_error_p90_m=float(np.quantile(error,.90))))
    result = dict(observations_sha256=_hash(path), folds=rows)
    _write(out,result)
    return result


class WristHistory:
    """Convert one actual wrist image into the earlier controller's optical reference."""

    def __init__(self, rig, parameters, sphere, *, delayed=False):
        self.rig = rig
        self.depth, self.shift, self.blade = parameters
        self.sphere = sphere
        self.phase = PhaseExtent(sphere)
        self.delayed = delayed
        self.frames = deque(maxlen=7)
        self.position = None
        self.velocity = np.zeros(3)
        self.age = 0

    def sense(self, frame, joints):
        image = component(frame, False)
        eye, axes, grip, grip_rotation = self.rig.pose(joints)
        padded = np.full((2, 2, 3), np.nan)
        paddle = grip + grip_rotation@self.blade
        for index, camera in enumerate(RIG):
            x, y, depth = camera.project(paddle)
            padded[1, index] = [x, y, 15.0/max(.5, depth)]
        if not np.isfinite(image).all():
            self.age += 1
            return padded
        ray = axes@(self.rig.bearing(image[:2])+np.r_[self.shift, 0.])
        size_depth = float(np.clip(design(image)@self.depth, .03, 3.))
        ray_unit = ray/np.linalg.norm(ray)
        scale = float(ray_unit@ray)
        distance = size_depth*scale
        if self.position is not None and self.age <= 2:
            advance = self.position + (self.age+1)*self.velocity
            prior = float((advance-eye)@ray_unit)
            # A few distant pixels give a weak range measurement. Angle remains observed.
            receive = float(np.clip(.13+.055*image[2], .2, .82))
            if prior > .035 and abs(distance-prior) < .55:
                distance = (1-receive)*prior + receive*distance
        measured = eye + distance*ray_unit
        if self.position is not None:
            delta = np.clip((measured-self.position)/(self.age+1), -.19, .19)
            self.velocity = .60*delta + .40*self.velocity
        self.position, self.age = measured, 0
        if np.any(np.abs(measured) > [1.1, 2.3, 1.4]):
            return padded
        for index, camera in enumerate(RIG):
            x, y, depth = camera.project(measured)
            if depth > 0 and 0 <= x < 480 and 0 <= y < 360:
                padded[0, index] = [x, y, self.sphere[index]/depth]
        return padded

    def observe(self, frames, joints, position):
        optical = self.sense(frames['left'], joints)
        if self.delayed:
            self.frames.append(optical)
            optical = self.frames[0]
        state = self.phase.read(optical, joints, position)
        y, speed, wait = state[[39, 42, 49]]
        if wait <= 1 and y < -.50 and speed < -.024:
            self.phase.approached = True
        if self.phase.approached and wait <= 1 and y < -.40 and speed > .028:
            self.phase.reversed = True
        if self.phase.reversed:
            self.phase.amount = min(1., self.phase.amount+.31)
        return np.r_[state, self.phase.amount]

    def issued(self, command):
        self.phase.issued(command)


class WristStation(Station):
    def __init__(self, profile):
        super().__init__(profile, render_dims=DIMS)
        render = self.arena.render_cameras
        self.arena.streams = ('wrist',)
        def current():
            rgb = render()['wrist']
            # Two checksum slots in the evaluator, one rendered eye.
            return dict(left=rgb, right=rgb)
        self.arena.render_cameras = current


def prepare(data_path, profile):
    global TRIAL
    with np.load(data_path) as data, np.load(SOURCE) as original, np.load(AUGMENTED) as augmented:
        lens, _ = fit(data)
        sphere = original['sphere']
        actor = PhasePolicy(OpticalPolicy(original, 'interaction'), OpticalPolicy(augmented, 'interaction'), .88)
    TRIAL = Station(profile, render_dims=DIMS), WristStation(profile), EncoderRig(), actor, lens, sphere


def run_point(start, order):
    outside, wrist, rig, actor, lens, sphere = TRIAL
    out = {}
    for name in order:
        if name == 'stereo':
            row = outside.play(start, 'stereo', actor, lambda: PhaseExtent(sphere), live_opening=True)
        else:
            history = lambda: WristHistory(rig, lens, sphere, delayed=name == 'wrist_delay')
            row = wrist.play(start, name, actor, history, live_opening=True)
        out[name] = row
    wrist_rgb = {out[name]['prefix_rgb'] for name in ('wrist','wrist_delay','hold')}
    if len(wrist_rgb) != 1:
        raise ValueError('wrist branches received different starting images')
    return dict(start, results=out)


def fresh(limit):
    starts = course(FRESH)[:limit]
    new = {row['seed'] for row in starts}
    known = {row['seed'] for row in course(HOLDOUT)+course(REPLICATION)+course(COMBINED_STARTS)}
    known.update(shot[2] for shot in SHOTS)
    known.update(row['seed'] for roster in _read(SCHEDULES)['schedules'].values() for row in roster)
    for prior in OUTPUT.glob('rally-eye-*.json.gz'):
        known.update(row['seed'] for row in _read(prior)['cases'])
    with np.load(AUGMENTED) as data:
        known.update(map(int, data['seed']))
    if known.intersection(new):
        raise ValueError('wrist evaluation overlaps previously observed launches')
    literal = re.compile(rb'(?<![0-9.])(?:'+b'|'.join(str(seed).encode() for seed in sorted(new))+rb')(?![0-9.])')
    scanned = 0
    for directory in (ARCHIVE.with_name('tt22'),ARCHIVE):
        for item in directory.iterdir():
            if item.name.endswith('.json.gz'):
                raw = gzip.decompress(item.read_bytes())
            elif item.suffix == '.json':
                raw = item.read_bytes()
            else:
                continue
            if literal.search(raw):
                raise ValueError('wrist evaluation repeats an archived release')
            scanned += 1
    return starts, scanned


def evaluate(out, sensors, calibration, workers, limit):
    starts, scanned = fresh(limit)
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-wrist-reference-rally-1', sensor_data_sha256=_hash(sensors),
                source_sha256=_hash(SOURCE), corrective_sha256=_hash(AUGMENTED),
                lens_summary=_read(calibration), dimensions=list(DIMS), holds=starts,
                archives_checked=scanned, conditions=list(MODES), teacher_commands_at_evaluation=0)
    receipt = _read(out) if out.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if receipt['metadata'] != meta:
        raise ValueError('wrist comparison checkpoint changed')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(sensors, profile)) as pool:
        jobs = {}
        for index, row in enumerate(starts):
            if receipt['cases'][index] is None:
                offset = index%len(MODES)
                jobs[pool.submit(run_point, row, MODES[offset:]+MODES[:offset])] = index
        for job in as_completed(jobs):
            point = job.result()
            receipt['cases'][jobs[job]] = point
            _write(out, receipt)
            print('wrist held', point['seed'], {name: returns(r,1) for name,r in point['results'].items()}, flush=True)
    receipt['verified_tapes'] = verify(receipt['cases'])
    receipt['summary'] = {name: summary(receipt['cases'],name,0) for name in MODES}
    for counts in receipt['summary'].values():
        counts['reach_first'] = sum(n>0 for n in counts['own_returns'])
    _write(out, receipt)
    return receipt['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--output', type=Path, default=RECEIPT)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if not args.data.exists() or not args.calibration.exists():
        collect(args.data, args.calibration)
    if args.audit:
        print(observation_audit(args.data, OBSERVATION_AUDIT), flush=True)
    else:
        print(evaluate(args.output, args.data, args.calibration, args.workers,args.limit), flush=True)

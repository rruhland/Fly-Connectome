"""Learn visual extent and relative paddle feedback from paired optical observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, SequenceArena, course, verify
from .tt24_rally_bank import CACHE as WIDE_CACHE
from .tt24_rally_bank import DIMS, HOLDOUT, sample_ranges, source_schedule
from .tt24_rally_improve import BODY_SCALE, RIG, triangulate
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_policy import EYES, ViewHistory
from .tt24_rally_trial import COACHED, OUTPUT, STEP_SIZE, PublicBody, Station, returns, summary


CACHE = OUTPUT / 'rally-improve-optical-demos.npz'
DEV = OUTPUT / 'rally-improve-optical-development.json.gz'
HELD = OUTPUT / 'rally-improve-optical-held.json.gz'
FRESH = {'k': 829305407, 'l': 470550286}
MODES = ('broad', 'extent', 'interaction')
ARENA = TRIAL = None


def component(rgb, blade):
    r, g, b = rgb.transpose(2, 0, 1)
    if blade:
        mask = (r > 70) & (g < .38*r) & (b < .48*r)
    else:
        mask = (r > 90) & (g > .26*r) & (g < .64*r) & (b < .26*r)
    y, x = np.nonzero(mask)
    if not len(x):
        return np.full(3, np.nan)
    x0, y0 = x.min(), y.min()
    if x.max()-x0 > 27 or y.max()-y0 > 27:
        clusters, n = label(mask[y0:y.max()+1, x0:x.max()+1])
        if n:
            labels = np.bincount(clusters.ravel(), minlength=n+1)
            labels[0] = 0
            largest = int(labels.argmax())
            slices = find_objects(clusters)[largest-1]
            yy, xx = np.nonzero(clusters[slices] == largest)
            x = xx + x0 + slices[1].start
            y = yy + y0 + slices[0].start
    scale = 480/rgb.shape[1]
    return np.r_[(np.array((x.mean(), y.mean()))+.5)*scale-.5, scale*np.sqrt(len(x)/np.pi)]


def snapshot(frames):
    return np.asarray([[component(frames[eye], blade) for eye in EYES] for blade in (False, True)])


def eye_depth(position, eye):
    return (position-np.asarray(eye.eye)) @ eye.basis[2]


def learn_radius(observations):
    data = observations[:, 0]
    valid = np.isfinite(data).all(axis=(1, 2))
    two = data[valid]
    xyz = triangulate(two[:, 0, :2], two[:, 1, :2])
    scale = []
    for idx, eye in enumerate(RIG):
        extent = two[:, idx, 2]
        d = eye_depth(xyz, eye)
        assay = extent*d
        useful = (d > .8) & (d < 4.4) & (extent > 2.) & (np.abs(xyz[:, 0]) < .8)
        scale.append(float(np.median(assay[useful])))
    return np.asarray(scale)


class ExtentHistory:
    def __init__(self, sphere, *, delayed=False):
        self.sphere = sphere
        self.base = ViewHistory()
        self.ball = None
        self.blade = None
        self.velocity = np.zeros(3)
        self.slow = np.zeros(3)
        self.wait = 0
        self.delayed = delayed
        self.past = []

    def observe(self, frames, joints, position):
        return self.read(snapshot(frames), joints, position)

    def read(self, observation, joints, position):
        ball, blade = observation
        visible = np.isfinite(ball).all(axis=1)
        paddle = np.isfinite(blade).all(axis=1)
        raw = self.base.observe_pixels([slot[:2] if seen else None for slot, seen in zip(ball, visible, strict=True)],
                                       joints, position)
        estimated = self.ball if self.ball is None else self.ball + self.velocity
        measure = None
        if visible.all():
            measure = triangulate(ball[0, :2], ball[1, :2])
        elif visible.any():
            index = int(np.argmax(visible))
            eye = RIG[index]
            center, radius = ball[index, :2], ball[index, 2]
            apparent_depth = self.sphere[index]/max(2., radius)
            depth = apparent_depth if estimated is None else .22*apparent_depth+.78*eye_depth(estimated, eye)
            measure = eye.eye + depth*eye.ray(center)
        if measure is None:
            self.wait += 1
        else:
            self.wait = 0
        if measure is None:
            self.ball = estimated
            self.velocity *= .78
        else:
            measure = np.clip(measure, [-1.1, -2.3, -.6], [1.1, 2.3, 1.3])
            if self.ball is not None:
                change = np.clip(measure-self.ball, -.21, .21)
                self.velocity = .75*change+.25*self.velocity
            self.ball = measure
        self.slow = .63*self.slow+.37*self.velocity
        if paddle.all():
            self.blade = triangulate(blade[0, :2], blade[1, :2])
        b = np.zeros(3) if self.ball is None else self.ball
        p = np.array([0., -.90, .20]) if self.blade is None else self.blade
        features = np.r_[raw, b, self.velocity, self.slow, np.nan_to_num(ball[:, 2]), min(self.wait, 7), b-p,
                         float(paddle.all())]
        if self.delayed:
            self.past.append(features.copy())
            lag = self.past[max(0, len(self.past)-7)]
            features[:14] = lag[:14]
            features[38:] = lag[38:]
        return features

    def issued(self, action):
        self.base.issued(action)


class OpticalPolicy:
    """Read observed ball motion and ball/paddle displacement into direct rates."""

    def __init__(self, data, mode):
        self.mode = mode
        columns = np.r_[np.arange(38, 50), np.arange(14, 38)]
        optical = np.r_[.10, .23, .075, .038, .060, .039, .038, .060, .039, 3.1, 3.1, 2.3]
        if mode == 'interaction':
            columns = np.r_[columns, np.arange(50, 54)]
        self.columns = columns
        self.scale = np.r_[optical, BODY_SCALE/.63]
        if mode == 'interaction':
            self.scale = np.r_[self.scale, .19, .33, .17, 2.0]
        self.train = data['features'][:, columns]/self.scale
        self.tree = cKDTree(self.train)
        self.values = np.column_stack((data['postures'], data['rates']))
        self.ridge = np.eye(len(columns)+1) * .24
        self.ridge[0, 0] = 0.

    def command(self, feature, issued):
        query = feature[self.columns]/self.scale
        distance, idx = self.tree.query(query, k=44)
        root = np.exp(-.25*((distance-distance[0])/max(.12, .48*distance[-1]))**2)
        design = np.column_stack((np.ones(len(idx)), self.train[idx]-query))*root[:, None]
        target, issued_rate = np.split(np.linalg.solve(design.T@design+self.ridge,
                                                       design.T@(self.values[idx]*root[:, None]))[0], 2)
        return np.clip(.85*(target-issued)/STEP_SIZE+.15*issued_rate, -1., 1.)


def prepare_collection():
    global ARENA
    ARENA = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))


def collect_tape(item):
    source, row = item
    arena = ARENA
    arena.lane = row['lane']
    arena.reset(seed=row['seed'], server=-1, render=False)
    body = PublicBody(arena)
    if row.get('refit_recovery'):
        frames = np.arange(len(row['actions']))
        first = row['transfer']
        end = first + row['fit_frames']
        keep = (frames >= first) & (frames < end)
        render = (frames >= max(0, first-7)) & (frames < end)
    else:
        keep, render = sample_ranges(row)
    samples = []
    for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
        near, far = np.asarray(near), np.asarray(far)
        if render[tick]:
            image = snapshot(arena.render_cameras())
            samples.append((tick, keep[tick], image, arena.joints, body.target.copy(), near))
        arena._aim = arena._command_to = lambda *args, far=far: far
        _, reward, done, capped, _ = arena.step(near, render=False)
        body.issued(near)
        if bool(done or capped) != (tick + 1 == len(row['actions'])):
            raise ValueError('demonstration reset or duration changed')
    if arena.point_result() != row['outcome'] or reward != row['reward'] or digest_json(arena.events) != digest_json(row['events']):
        raise ValueError('demo scores differ under strict replay')
    print('optical tape', row['seed'], len(samples), flush=True)
    return samples


def collect(path, workers):
    items, _ = source_schedule(48, course(HOLDOUT) + course(FRESH))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=prepare_collection) as pool:
        blocks = list(pool.map(collect_tape, items))
    sphere = learn_radius(np.asarray([sample[2] for block in blocks for sample in block]))
    features, postures, rates, seeds = [], [], [], []
    for ((source, row), block) in zip(items, blocks, strict=True):
        history = None
        previous = -1
        for tick, keep, observations, joints, issued, command in block:
            if history is None or tick != previous+1:
                history = ExtentHistory(sphere)
            feature = history.read(observations, joints, issued)
            history.issued(command)
            if keep:
                features.append(feature)
                postures.append(issued + STEP_SIZE*command)
                rates.append(command)
                seeds.append(row['seed'])
            previous = tick
    np.savez_compressed(path, features=features, postures=postures, rates=rates, seed=seeds,
                        sphere=sphere, replayed=len(items), frames=sum(len(row['actions']) for _,row in items),
                        rendered=sum(map(len, blocks)))


def prepare_trial(data_path, profile, modes, selected):
    global TRIAL
    station = Station(profile, render_dims=DIMS)
    with np.load(data_path) as data, np.load(WIDE_CACHE) as wide:
        policies = {name: OpticalPolicy(data, name) for name in modes if name in ('extent', 'interaction')}
        policies['broad'] = policy_for('broad', wide)
        sphere = data['sphere']
    TRIAL = station, policies, sphere, selected


def play(start, modes):
    station, policies, sphere, selected = TRIAL
    rows = {}
    for name in modes:
        if name in ('broad', 'hold'):
            rows[name] = station.play(start, name, policies.get(name), live_opening=True)
        else:
            controller = policies[selected if name == 'optical_delay' else name]
            history = lambda: ExtentHistory(sphere, delayed=name == 'optical_delay')
            rows[name] = station.play(start, name, controller, history, live_opening=True)
    if len({row['prefix_rgb'] for row in rows.values()}) != 1:
        raise ValueError('optical branches saw different starting frames')
    return dict(start, results=rows)


def evaluate(path, data, workers, *, confirm):
    selected = None
    if confirm:
        report = _read(DEV)
        if not report.get('verified_tapes'):
            raise ValueError('development comparison is incomplete')
        scores = report['summary']
        selected = max(('extent', 'interaction'), key=lambda name: (scores[name]['reach_fifth'], scores[name]['reach_second'],
                                                                    scores[name]['total_own_returns']))
        modes = ('broad', selected, 'optical_delay', 'hold')
        starts = course(FRESH)
    else:
        modes, starts = MODES, course(HOLDOUT)[:12]
    with np.load(data) as old:
        if set(map(int, old['seed'])) & {row['seed'] for row in starts}:
            raise ValueError('held exposure overlaps optical demonstrations')
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    meta = dict(schema='tt24-optical-extent-and-paddle-1', data_sha256=_hash(data),
                release=starts, modes=list(modes), selection=selected, confirmation=confirm)
    report = _read(path) if path.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if report['metadata'] != meta:
        raise ValueError('different test or data checkpoint')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare_trial,
                             initargs=(data, profile, modes, selected)) as pool:
        tasks = {pool.submit(play, row, modes): i for i, row in enumerate(starts) if report['cases'][i] is None}
        for task in as_completed(tasks):
            row = task.result()
            report['cases'][tasks[task]] = row
            _write(path, report)
            print('optical', row['seed'], {name: returns(result, 1) for name,result in row['results'].items()}, flush=True)
    report['verified_tapes'] = verify(report['cases'])
    report['summary'] = {name: summary(report['cases'], name, 0) for name in modes}
    for entry in report['summary'].values():
        entry['reach_first'] = sum(n > 0 for n in entry['own_returns'])
    _write(path, report)
    return report['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--data', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    if not args.data.exists():
        collect(args.data, workers=args.workers)
    print(evaluate(args.output or (HELD if args.confirm else DEV), args.data, args.workers, confirm=args.confirm), flush=True)

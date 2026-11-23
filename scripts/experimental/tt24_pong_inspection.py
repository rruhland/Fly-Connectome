"""Select when to inspect the actuator from actual native Pong point rewards."""

import argparse
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import HEIGHT, ROWS, STRIDE, WARM, WIDTH, field


PLAN = ROOT / 'pong-inspection-plan.json'
PILOT = ROOT / 'pong-inspection-pilot.json.gz'
OUTPUT = ROOT / 'pong-inspection-held.json.gz'
BASELINE_PLAN = ROOT / 'pong-inspection-high-plan.json'
BASELINE_OUT = ROOT / 'pong-inspection-high-held.json.gz'
MASTER = 1410359187
CAP = 12 * 120
DT = 1. / 120.
CAMERA_DT = STRIDE * DT
EXCURSION = .7 * CAMERA_DT
SHARED_FIXED = .068 / (5.2 / 60.)
THRESHOLDS = np.asarray((.20, .35, .50, .65, .80, 1.0, 1.3, 1.7, 2.3, 3.3))
PROBE = .91 * np.r_[np.ones(4), -np.ones(7), np.ones(4)]
PERIODIC = 30
CHARGED_STEPS = len(PROBE) * STRIDE


@dataclass(frozen=True)
class Plant:
    rise: float
    strength: float
    drag: float

    def advance(self, q, v, activation, u):
        blend = -np.expm1(-DT / self.rise)
        for _ in range(STRIDE):
            activation += blend * (u - activation)
            v += DT * (-self.strength * activation - self.drag * v)
            v = np.clip(v, -.7, .7)
            q = q + DT * v
            v = np.where((q <= .10) | (q >= .90), 0., v)
            q = np.clip(q, .10, .90)
        return q, v, activation


def actor_pixels(frame):
    pixels = frame[:, :, 1]
    return (pixels @ ROWS) / np.maximum(1, pixels.sum(axis=1))


@torch.no_grad()
def babble(seed, count, camera_frames):
    rng = np.random.default_rng(seed)
    launches = rng.choice(2**31 - 2, count, replace=False).tolist()
    game = Pong(launches)
    drive = rng.uniform(-1., 1., count)
    next_change = rng.integers(5, 16, count)
    actions, observations = [], [actor_pixels(game.render(HEIGHT, WIDTH).numpy())]
    for tick in range(camera_frames):
        alter = tick == next_change
        drive[alter] = rng.uniform(-1., 1., int(alter.sum()))
        next_change[alter] += rng.integers(5, 16, int(alter.sum()))
        actions.append(drive.copy())
        for _ in range(STRIDE):
            game.step(torch.from_numpy(drive.astype('float32')), auto_reset=False)
        observations.append(actor_pixels(game.render(HEIGHT, WIDTH).numpy()))
    return np.asarray(actions), np.asarray(observations)


def fit_plant():
    train = babble(1154893985, 20, 112)
    test = babble(789210041, 8, 112)

    def prediction(plant, actions, observations):
        positions = [observations[0].copy()]
        q = positions[0].copy()
        v, activation = np.zeros(len(q)), np.zeros(len(q))
        for u in actions:
            q, v, activation = plant.advance(q, v, activation, u)
            positions.append(q)
        return np.asarray(positions) - observations

    initial = (.14, 2., 3.)
    fit = least_squares(lambda p: prediction(Plant(*p), *train).ravel(), initial,
                        bounds=((.02, .6, .9), (.4, 6., 12.)), loss='soft_l1', f_scale=.008)
    plant = Plant(*fit.x)
    residual = prediction(plant, *test) / EXCURSION
    return plant, dict(babble_frames=sum(observations.size for _, observations in (train, test)),
                       train_motion_frames=train[1].size, held_motion_frames=test[1].size,
                       held_rms_excursion_units=float(np.sqrt(np.mean(residual**2))),
                       held_p95_abs_excursion_units=float(np.quantile(np.abs(residual), .95)))


def roster(seed, count, *, held=False):
    rng = np.random.default_rng(seed)
    launches = rng.choice(2**31 - 2, count, replace=False)
    profile = rng.permutation(np.arange(count) % 5)
    rows = []
    for start, regime in zip(launches, profile, strict=True):
        schedule = []
        if regime == 1:
            schedule = [(int(rng.integers(12, 23)), -1)]
        elif regime == 2:
            change = int(rng.integers(34, 78))
            schedule = [(change, -1), (change + int(rng.integers(60, 123)), 1)]
        elif regime == 3:
            change = int(rng.integers(48, 110))
            schedule = [(change, -1), (change + int(rng.integers(11, 23)), 1)]
        elif regime == 4:
            change = int(rng.integers(70, 140))
            schedule = [(0, -1), (change, 1)]
        gain = rng.uniform(.86, 1.14) if held else rng.uniform(.93, 1.07)
        rows.append(dict(seed=int(start), profile=int(regime), gain=float(gain), schedule=[list(x) for x in schedule]))
    return rows


def wiring(rows):
    result = np.ones((CAP // STRIDE, len(rows)))
    for i, row in enumerate(rows):
        for a, sign in row['schedule']:
            result[a:, i] = sign
    return result


class Inspector:
    """Run the same evidence accumulator as the arm in excursion-normalized units."""

    def __init__(self, plant, thresholds, modes, count):
        self.plant = plant
        self.threshold = np.repeat(thresholds, count)
        self.mode = np.repeat(modes, count)
        size = len(self.threshold)
        self.assumed = np.ones(size)
        self.evidence = np.zeros(size)
        self.position, self.velocity, self.activation = [np.zeros(size) for _ in range(3)]
        self.probing = np.full(size, -1, dtype=int)
        self.since = np.full(size, PERIODIC, dtype=int)
        self.phase_sign = np.ones(size)
        shape = (size, 2, 3)
        self.probe_q, self.probe_v, self.probe_activation, self.errors = [np.zeros(shape) for _ in range(4)]

    def decide(self, tick, optical, speed, playing, live):
        if tick == 0:
            self.position[:] = optical
        ongoing = (self.probing >= 0) & live
        err = optical - self.position
        measuring = live & ~ongoing
        self.evidence[measuring] = np.maximum(0., self.evidence[measuring] +
                                              np.where(np.abs(err[measuring]) / EXCURSION > self.threshold[measuring],
                                                       1., -.5))
        completed = ongoing & (self.probing == len(PROBE))
        if ongoing.any():
            self.errors[ongoing] += (self.probe_q[ongoing] - optical[ongoing, None, None]) ** 2
        if completed.any():
            losses = self.errors[completed].min(axis=-1)
            self.assumed[completed] = np.where(losses[:, 0] <= losses[:, 1], 1., -1.)
            self.position[completed] = optical[completed]
            self.velocity[completed] = speed[completed]
            slot = np.argmin(self.errors[completed].reshape((-1, 6)), axis=1)
            self.activation[completed] = self.probe_activation[completed].reshape((-1, 6))[np.arange(len(slot)), slot]
            self.evidence[completed] = 0.
            self.probing[completed] = -1
            self.since[completed] = 0
        idle = live & (self.probing < 0)
        trigger = idle & ((self.mode == 'always') | ((self.mode == 'periodic') & (self.since >= PERIODIC)) |
                          ((self.mode == 'threshold') & (self.evidence >= 4.)))
        if trigger.any():
            self.probing[trigger] = 0
            self.phase_sign[trigger] = np.where(optical[trigger] >= .5, 1., -1.)
            self.probe_q[trigger] = optical[trigger, None, None]
            self.probe_v[trigger] = speed[trigger, None, None]
            self.probe_activation[trigger] = np.array((-1., 0., 1.))[None, None, :]
            self.errors[trigger] = 0.
        active = live & (self.probing >= 0)
        command = self.assumed * playing
        command[active] = self.phase_sign[active] * PROBE[self.probing[active]]
        self.probing[active] += 1
        self.since[idle & ~active] += 1
        return command, trigger, completed, active, np.abs(err) / EXCURSION

    def advance(self, command, probing):
        self.position, self.velocity, self.activation = self.plant.advance(self.position, self.velocity,
                                                                           self.activation, self.assumed * command)
        if probing.any():
            state = self.plant.advance(self.probe_q[probing], self.probe_v[probing], self.probe_activation[probing],
                                       command[probing, None, None] * np.array((1., -1.))[None, :, None])
            self.probe_q[probing], self.probe_v[probing], self.probe_activation[probing] = state


@torch.no_grad()
def points(weights, plant, rows, settings):
    count, variants = len(rows), len(settings)
    size = count * variants
    native = Pong([r['seed'] for r in rows] * variants, Physics(opponent_speed=.45))
    inspector = Inspector(plant, [x[1] for x in settings], [x[0] for x in settings], count)
    transmissions = wiring(rows)
    magnitudes = np.tile([r['gain'] for r in rows], variants)
    live = np.ones(size, dtype=bool)
    optical_ball = np.zeros((size, 2))
    mover_speed = np.zeros_like(optical_ball)
    visibility = np.zeros(size, dtype=bool)
    score, elapsed, paddle_speed, desired = [np.zeros(size) for _ in range(4)]
    counts = {name: np.zeros(size, dtype=int) for name in ('checks', 'false_checks', 'paused_ticks', 'identified',
                                                           'correct', 'wrong_map_frames', 'settled_identified',
                                                           'settled_correct')}
    histories = [[] for _ in range(size)]
    actual_at_check = np.ones(size)
    probe_clean = np.ones(size, dtype=bool)
    for frame in range(CAP):
        if frame % STRIDE == 0:
            tick = frame // STRIDE
            actual = np.tile(transmissions[tick], variants)
            optical, seen = field(native.render(HEIGHT, WIDTH).numpy(), optical_ball, mover_speed, visibility)
            if tick == 0:
                home, prior = optical.copy(), optical.copy()
            gap = np.clip((optical_ball[:, 1] - optical) * .76 / .065, -3., 3.)
            tangent = np.clip(mover_speed[:, 1] * 3. / .38, -3., 3.)
            pose = np.clip((optical - home) * .76 / .04, -3., 3.)
            normal = np.clip(mover_speed[:, 0] * 3. / 1.5, -3., 3.)
            x = np.stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)), axis=1)
            offset = .36 * np.tanh((x * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (optical - prior) * (120 / STRIDE)
            requested = np.clip(-11. * (home + offset - optical) + 1.35 * paddle_speed, -1., 1.)
            desired[seen] = requested[seen]
            prior = optical
            old_map = inspector.assumed.copy()
            probe_clean &= actual == actual_at_check
            command, begins, completed, active, error = inspector.decide(tick, optical, paddle_speed, desired, live)
            matching = inspector.assumed == actual
            counts['checks'] += begins
            counts['false_checks'] += begins & (inspector.assumed == actual)
            counts['identified'] += completed
            counts['correct'] += completed & matching
            counts['settled_identified'] += completed & probe_clean
            counts['settled_correct'] += completed & matching & probe_clean
            counts['wrong_map_frames'] += live & ~matching
            for i in np.flatnonzero(begins | completed):
                histories[i].append([tick, bool(begins[i]), bool(completed[i]), float(old_map[i]),
                                     float(inspector.assumed[i]), float(actual[i]), float(error[i])])
            probe_clean[begins] = True
            actual_at_check[begins] = actual[begins]
            inspector.advance(command, active)
            drive = torch.from_numpy((magnitudes * actual * command).astype('float32'))
        result = native.step(drive, auto_reset=False)
        scored = result.scores.numpy()
        counts['paused_ticks'] += active & live
        elapsed += live
        score += scored * live
        live &= scored == 0.
        if not live.any():
            break
    def row(i):
        return dict(score=int(score[i]), live_ticks=int(elapsed[i]), **{key: int(value[i]) for key, value in counts.items()},
                    events=histories[i])
    return [[row(i * count + j) for j in range(count)] for i in range(variants)]


def summarize(results, scenarios):
    scores = np.asarray([row['score'] for row in results])
    keys = ('checks', 'false_checks', 'paused_ticks', 'identified', 'correct', 'wrong_map_frames',
            'settled_identified', 'settled_correct', 'live_ticks')
    totals = {key: sum(r[key] for r in results) for key in keys}
    stable = [r for r, scenario in zip(results, scenarios, strict=True) if scenario['profile'] == 0]
    changed = np.asarray([scenario['profile'] != 0 for scenario in scenarios])
    return dict(points=len(results), wins=int((scores > 0).sum()), losses=int((scores < 0).sum()),
                timeouts=int((scores == 0).sum()), net=int(scores.sum()), changed_net=int(scores[changed].sum()),
                stable_net=int(scores[~changed].sum()), stable_points=len(stable),
                stable_points_with_check=sum(r['checks'] > 0 for r in stable),
                stable_alarm_count=sum(r['checks'] for r in stable), **totals)


def effect(left, right):
    difference = np.array([a['score'] - b['score'] for a, b in zip(left, right, strict=True)])
    rng = np.random.default_rng(80172999)
    interval = np.quantile(rng.choice(difference, (16000, len(difference))).mean(axis=1), (.025, .975))
    return dict(mean=float(difference.mean()), interval=interval.tolist(),
                wins=int(sum(difference > 0)), ties=int(sum(difference == 0)), losses=int(sum(difference < 0)))


def select(rewards):
    curve = rewards.mean(axis=1)
    # The same kernel pools adjacent dose evaluations for real and shuffled points.
    padding = np.pad(curve, 1, mode='edge')
    smooth = .20 * padding[:-2] + .60 * padding[1:-1] + .20 * padding[2:]
    best = np.flatnonzero(np.isclose(smooth, smooth.max(), atol=1.e-10, rtol=0.))
    winner = int(best[(len(best) - 1) // 2])
    return dict(threshold=float(THRESHOLDS[winner]), curve=curve.tolist(), smoothed=smooth.tolist())


def make_plan():
    rng = np.random.default_rng(MASTER)
    pilot = roster(int(rng.integers(1, 2**31)), 30, held=True)
    runs = []
    used = {r['seed'] for r in pilot}
    for _ in range(4):
        practice = roster(int(rng.integers(1, 2**31)), 60)
        held = roster(int(rng.integers(1, 2**31)), 50, held=True)
        for group in (practice, held):
            ids = {r['seed'] for r in group}
            if used.intersection(ids):
                raise ValueError('an inspection course reuses a physical launch')
            used.update(ids)
        runs.append(dict(practice=practice, held=held, shuffle=int(rng.integers(1, 2**31))))
    doc = dict(schema='tt24-pong-inspection-1', pilot=pilot, streams=runs, master=MASTER,
               motor_weights_sha256=_hash(WARM), max_live_native_ticks=CAP, evidence_threshold=4.,
               negative_evidence_increment=-.50, normalizer=EXCURSION, per_check_native_ticks=CHARGED_STEPS,
               fixed_arm_equivalent=SHARED_FIXED, periodic_wait_camera_ticks=PERIODIC,
               reward='native terminal point score only', thresholds=THRESHOLDS.tolist(),
               selection='center .60 plus adjacent .20; midpoint of maximal discrete tolerance',
               fit_inputs='game score selects threshold; predictor and mapping use optical paddle coordinates and own actions',
               monitor='RMS prediction error divided by permitted per-observation actuator excursion',
               no_table_or_wrist_experience=True)
    if PLAN.exists() and doc != _read(PLAN):
        raise ValueError('inspection course differs from its registered plan')
    _write(PLAN, doc)
    return doc


def controls():
    return dict(fixed_arm=('threshold', SHARED_FIXED), fixed_low=('threshold', .5), fixed_unit=('threshold', 1.),
                never=('never', 999.), always=('always', 999.), periodic=('periodic', 999.))


def pilot(plan, weights):
    if PILOT.exists():
        doc = _read(PILOT)
        if doc['plan_sha256'] != _hash(PLAN):
            raise ValueError('previous pilot has a different protocol')
        return Plant(**doc['plant']), doc
    plant, calibration = fit_plant()
    print('inspection optical plant', asdict(plant), calibration, flush=True)
    settings = controls() | dict(low=('threshold', .20), high=('threshold', 1.3))
    roster = plan['pilot']
    rows = points(weights, plant, roster, list(settings.values()))
    stats = {name: summarize(result, roster) for name, result in zip(settings, rows, strict=True)}
    checked = [value for name, value in stats.items() if settings[name][0] == 'threshold']
    denominator = sum(x['settled_identified'] for x in checked)
    accurate = sum(x['settled_correct'] for x in checked) / max(1, denominator)
    doc = dict(plan_sha256=_hash(PLAN), plant=asdict(plant), calibration=calibration, settled_accuracy=accurate,
               modes=settings, results=dict(zip(settings, rows, strict=True)), summary=stats)
    _write(PILOT, doc)
    print('inspection pilot', accurate, {name: (r['net'], r['checks'], r['stable_alarm_count'])
                                         for name, r in stats.items()}, flush=True)
    return plant, doc


def run(plan, weights, plant):
    doc = dict(plan_sha256=_hash(PLAN), pilot_sha256=_hash(PILOT), streams=[])
    if OUTPUT.exists():
        saved = _read(OUTPUT)
        if any(saved[k] != doc[k] for k in ('plan_sha256', 'pilot_sha256')):
            raise ValueError('source retention receipt has a different sequence or plant')
        doc = saved
    for i, stream in enumerate(plan['streams'][len(doc['streams']):], start=len(doc['streams'])):
        fits = points(weights, plant, stream['practice'], [('threshold', t) for t in THRESHOLDS])
        roi = np.array([[row['score'] for row in results] for results in fits])
        shuffler = np.random.default_rng(stream['shuffle'])
        order = np.column_stack([shuffler.permutation(len(THRESHOLDS)) for _ in stream['practice']])
        learned = dict(pong=select(roi), shuffled=select(np.take_along_axis(roi, order, axis=0)))
        settings = {k: ('threshold', row['threshold']) for k, row in learned.items()} | controls()
        held_rows = points(weights, plant, stream['held'], list(settings.values()))
        held = dict(zip(settings, held_rows, strict=True))
        summaries = {role: summarize(rows, stream['held']) for role, rows in held.items()}
        doc['streams'].append(dict(stream=i, chosen=learned, settings=settings, summaries=summaries, held=held,
                                   source_training_terminal_scores=roi.tolist(),
                                   source_training_live_ticks=sum(row['live_ticks'] for group in fits for row in group),
                                   training_points=roi.size, source_training_checks=sum(row['checks'] for group in fits for row in group)))
        pooled = {role: [row for stream in doc['streams'] for row in stream['held'][role]] for role in settings}
        doc['paired'] = {role: effect(pooled['pong'], rows) for role, rows in pooled.items() if role != 'pong'}
        scenarios = [case for stream in plan['streams'][:len(doc['streams'])] for case in stream['held']]
        doc['summary'] = {role: summarize(rows, scenarios) for role, rows in pooled.items()}
        _write(OUTPUT, doc)
        print('inspection stream', i, {role: row['threshold'] for role, row in learned.items()},
              {role: (row['net'], row['checks'], row['stable_alarm_count']) for role, row in summaries.items()}, flush=True)


def register_high():
    doc = dict(parent_plan_sha256=_hash(PLAN), thresholds=[1.3, 2.3, 3.3],
               rationale='explicit fixed high-threshold controls following the first boundary selection',
               held_point_rosters='reuse main independent held launches with no selection or threshold retuning')
    if BASELINE_PLAN.exists() and doc != _read(BASELINE_PLAN):
        raise ValueError('explicit high threshold plan changed')
    _write(BASELINE_PLAN, doc)
    return doc


def high_baseline(plan, weights, plant):
    baseline = register_high()
    primary = _read(OUTPUT)
    if len(primary['streams']) != len(plan['streams']):
        raise ValueError('all registered source fits must finish first')
    cohorts = []
    combined = {str(t): [] for t in baseline['thresholds']}
    pong = []
    all_scenarios = []
    for stream, scenarios in zip(primary['streams'], plan['streams'], strict=True):
        scenarios = scenarios['held']
        cached = {t: stream['held'][name] for name, (tag, t) in stream['settings'].items() if tag == 'threshold'}
        new = [t for t in baseline['thresholds'] if t not in cached]
        if new:
            results = points(weights, plant, scenarios, [('threshold', t) for t in new])
            cached.update(dict(zip(new, results, strict=True)))
        for t in baseline['thresholds']:
            combined[str(t)].extend(cached[t])
        all_scenarios.extend(scenarios)
        pong.extend(stream['held']['pong'])
        cohorts.append({str(t): dict(new_live_points=t in new, raw=cached[t], summary=summarize(cached[t], scenarios))
                        for t in baseline['thresholds']})
    summary = {name: summarize(rows, all_scenarios) for name, rows in combined.items()}
    paired = {name: effect(pong, rows) for name, rows in combined.items()}
    _write(BASELINE_OUT, dict(baseline_plan_sha256=_hash(BASELINE_PLAN), primary_sha256=_hash(OUTPUT),
                              summary=summary, paired=paired, cohorts=cohorts))
    print('inspection fixed high thresholds', {k: (v['net'], v['checks'], v['stable_alarm_count'])
                                                for k, v in summary.items()}, 'paired', paired, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--pilot-only', action='store_true')
    stages.add_argument('--high-plan', action='store_true')
    stages.add_argument('--high-baselines', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = make_plan()
    if args.high_plan:
        register_high()
    else:
        actor = np.asarray(_read(WARM)['weights'])
        plant, receipt = pilot(protocol, actor)
        if args.high_baselines:
            high_baseline(protocol, actor, plant)
        elif not args.pilot_only:
            run(protocol, actor, plant)

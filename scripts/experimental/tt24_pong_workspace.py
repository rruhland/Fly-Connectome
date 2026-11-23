"""Learn a signed mover-to-reference displacement from official native Pong."""

import argparse
from collections import deque
from hashlib import sha256

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import HEIGHT, PLAN as PARENT, STRIDE, WARM, WIDTH, field


PLAN = ROOT / 'pong-workspace-source-plan.json'
PILOT = ROOT / 'pong-workspace-source-pilot.json.gz'
LOCK = ROOT / 'pong-workspace-source-lock.json.gz'
HELD = ROOT / 'pong-workspace-source-held.json.gz'
MASTER = 191108226
CAP = 12 * 120
PILOT_GRID = np.linspace(-1., 1., 5)
GRID = np.linspace(-1., 1., 9)


def initial(seeds):
    frame = Pong(seeds).render(HEIGHT, WIDTH).numpy()
    p, v = [np.zeros((len(seeds), 2)) for _ in range(2)]
    paddle, seen = field(frame, p, v, np.zeros(len(seeds), dtype=bool))
    if not seen.all():
        raise ValueError('a native official opening raster lacks its mover')
    pupil = frame[:, :, 1]
    return paddle, p[:, 1], pupil.sum(axis=1) / (2. * HEIGHT)


def reference():
    prior = _read(PARENT)
    warm = [seed for seeds in prior['warm_train'] for seed in seeds]
    paddle, mover, half = initial(warm)
    return dict(offset=float(np.median(mover-paddle)), optical_half_height=float(np.median(half)),
                data_frames=len(warm), exposures=warm, cap=float(.75 * np.median(half)),
                agreement_limit=float(.675 * np.median(half)))


def trial_roster(rng, count, calibration, used, *, wide):
    pool = rng.choice(2**31 - 2, 9 * count if wide else count + 20, replace=False) + 1
    pool = [x for x in pool.tolist() if x not in used]
    if wide:
        paddle, ball, _ = initial(pool)
        delta = ball-paddle-calibration['offset']
        half = count // 2
        left = np.flatnonzero(delta <= -.15)[:half]
        right = np.flatnonzero(delta >= .15)[:half]
        selected = [pool[i] for i in rng.permutation(np.r_[left, right])]
    else:
        selected = pool[:count]
    if len(selected) != count:
        raise ValueError('official raster seed selection did not fill its predeclared roster')
    used.update(selected)
    return selected


def register():
    rng = np.random.default_rng(MASTER)
    parent = _read(PARENT)
    used = {x for row in parent['warm_train'] for x in row} | set(parent['warm_held'])
    used |= {x for row in parent['streams'] for x in row['held_native']}
    used |= {x for row in parent['streams'] for batch in row['native'] for x in batch}
    origin = _read(ROOT / 'tt24_pong_origin-plan.json')['rosters']
    used |= {x for v in origin.values() for x in v['seeds']}
    calibration = reference()
    pilot = dict(wide=trial_roster(rng, 72, calibration, used, wide=True),
                 ordinary=trial_roster(rng, 32, calibration, used, wide=False))
    cohorts = []
    for _ in range(4):
        cohorts.append(dict(practice=trial_roster(rng, 72, calibration, used, wide=True),
                            held_wide=trial_roster(rng, 48, calibration, used, wide=True),
                            held_ordinary=trial_roster(rng, 40, calibration, used, wide=False),
                            shuffle=int(rng.integers(1, 2**31))))
    plan = dict(schema='tt24-pong-workspace-source-1', selection_seed=MASTER,
                scorer='terminal native point score only, 12 second timeouts count zero',
                actor_sha256=_hash(WARM), actor_plan_sha256=_hash(PARENT), calibration=calibration,
                official_serves='untouched native ball positions, launch motion, opponent and paddle reset',
                visual_wide='balanced at +/-0.15 initial displacement from earlier raster median',
                movable_axis='native camera row, the axis shared by the own actuator and the mover',
                transform='origin plus gain times clip(mover-query minus original origin minus familiar relative offset, +/-0.75 paddle half-height)',
                pilot=pilot, cohorts=cohorts, pilot_grid=PILOT_GRID.tolist(), grid=GRID.tolist(),
                pilot_gate='a signed fixed gain improves wide net by >=4 over zero and >=3 over opposite sign, improves at least 3 wide points, and loses <=2 net on ordinary',
                fit='bounded quadratic fit to matched terminal-score means with fixed .05 ridge on slope and curvature')
    if PLAN.exists() and _read(PLAN) != plan:
        raise ValueError('registered source workspace course changed')
    _write(PLAN, plan)
    return plan


@torch.no_grad()
def points(weights, seeds, gains, calibration):
    """Run only real native points with the same learned motor and optical readout."""
    repeats, cases = len(gains), len(seeds)
    beta = np.repeat(gains, cases)
    n = len(beta)
    game = Pong(seeds * repeats, Physics(opponent_speed=.45))
    live = np.ones(n, dtype=bool)
    old, velocity = [np.zeros((n, 2)) for _ in range(2)]
    visible = np.zeros(n, dtype=bool)
    score, hits, elapsed, action, paddle_speed = [np.zeros(n) for _ in range(5)]
    tape = np.zeros((CAP // STRIDE, n), dtype='<f4')
    trajectory = deque(maxlen=3)
    selected = np.zeros(n, dtype=bool)
    query_tick = np.full(n, -1, dtype=int)
    pad = np.zeros(n)
    for step in range(CAP):
        if step % STRIDE == 0:
            tick = step // STRIDE
            paddle, seen = field(game.render(HEIGHT, WIDTH).numpy(), old, velocity, visible)
            if tick == 0:
                base = paddle.copy()
                origin = base.copy()
                previous_paddle = base.copy()
                starts = old[:, 1].copy()
            if tick < 10 and not selected.all():
                trajectory.append(np.where(seen, old[:, 1], np.nan))
                if len(trajectory) == 3:
                    history = np.asarray(trajectory)
                    valid = ~selected & np.isfinite(history).all(axis=0)
                    valid &= (abs(np.diff(history, axis=0)) <= calibration['agreement_limit']).all(axis=0)
                    observed = np.median(history[:, valid], axis=0)
                    displacement = observed-base[valid]-calibration['offset']
                    pad[valid] = np.clip(displacement, -calibration['cap'], calibration['cap'])
                    origin[valid] = base[valid]+beta[valid]*pad[valid]
                    selected[valid] = True
                    query_tick[valid] = tick
            gap = np.clip((old[:, 1] - paddle) * .76 / .065, -3., 3.)
            tangent = np.clip(velocity[:, 1] * 3. / .38, -3., 3.)
            pose = np.clip((paddle - origin) * .76 / .04, -3., 3.)
            normal = np.clip(velocity[:, 0] * 3. / 1.5, -3., 3.)
            features = np.column_stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)))
            correction = .36 * np.tanh((features * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (120 / STRIDE)
            request = np.clip(-11. * (origin + correction - paddle) + 1.35 * paddle_speed, -1., 1.)
            action[seen] = request[seen]
            tape[tick, live] = action[live]
            previous_paddle = paddle
        outcome = game.step(torch.from_numpy(action.astype(np.float32)), auto_reset=False)
        scored = outcome.scores.numpy()
        score += scored * live
        hits += outcome.hits.numpy() * live
        elapsed += live
        live &= scored == 0.
        if not live.any():
            break
    tape_hash = np.array([sha256(np.ascontiguousarray(col, dtype='<f4').tobytes()).hexdigest()
                          for col in tape.T]).reshape(repeats, cases)
    initial_state = np.column_stack((starts, base, pad, query_tick)).reshape(repeats, cases, 4)
    if not np.all(initial_state == initial_state[0]):
        raise ValueError('candidate gains observed different release queries')
    return np.column_stack((score, hits, elapsed)).reshape(repeats, cases, 3), tape_hash, initial_state[0]


def summary(points):
    scores, hits, elapsed = points.T
    return dict(points=len(points), net=int(scores.sum()), wins=int((scores > 0).sum()), losses=int((scores < 0).sum()),
                timed_out=int((scores == 0).sum()), any_return=int((hits > 0).sum()),
                total_returns=int(hits.sum()), live_native_frames=int(elapsed.sum()))


def pilot(plan, weights):
    receipt = dict(plan_sha256=_hash(PLAN), by_course={})
    if PILOT.exists():
        old = _read(PILOT)
        if old['plan_sha256'] != receipt['plan_sha256']:
            raise ValueError('stored pilot is for another course')
        return old
    observed = {}
    for name, seeds in plan['pilot'].items():
        values, tapes, cues = points(weights, seeds, PILOT_GRID, plan['calibration'])
        observed[name] = values[..., 0]
        receipt['by_course'][name] = dict(outcomes=values.tolist(), action_sha256=tapes.tolist(), cue=cues.tolist(),
                                           stats={str(b): summary(v) for b, v in zip(PILOT_GRID, values, strict=True)})
    wide, ordinary = (observed[name] for name in ('wide', 'ordinary'))
    accepted = []
    for i, beta in enumerate(PILOT_GRID):
        if beta == 0:
            continue
        difference = wide[i]-wide[2]
        useful = difference.sum() >= 4 and (difference > 0).sum() >= 3
        direction_distinguished = (wide[i]-wide[-i-1]).sum() >= 3
        ordinary_retained = (ordinary[i]-ordinary[2]).sum() >= -2
        if useful and direction_distinguished and ordinary_retained:
            accepted.append(float(beta))
    receipt.update(gate=bool(accepted), eligible_fixed_values=accepted)
    _write(PILOT, receipt)
    print('native workspace pilot', {name: np.sum(scores, axis=1).astype(int).tolist() for name, scores in observed.items()},
          'eligible fixed values', accepted, flush=True)
    return receipt


def select(points):
    mean = points.mean(axis=1)
    design = np.column_stack((np.ones(len(GRID)), GRID, GRID ** 2))
    coefficients = np.linalg.solve(design.T @ design + np.diag((0., .05, .05)), design.T @ mean)
    slope, bend = coefficients[1:]
    if bend < 0.:
        gain = np.clip(-slope / (2. * bend), -1., 1.)
    else:
        gain = np.sign(slope)
    return dict(gain=float(gain), grid_means=mean.tolist(), coefficients=coefficients.tolist())


def learn(plan, weights):
    doc = dict(plan_sha256=_hash(PLAN), pilot_sha256=_hash(PILOT), source_streams=[])
    if LOCK.exists():
        old = _read(LOCK)
        if old['plan_sha256'] != doc['plan_sha256'] or old['pilot_sha256'] != doc['pilot_sha256']:
            raise ValueError('stored workspace fit differs from source protocol')
        doc = old
    for index, cohort in enumerate(plan['cohorts'][len(doc['source_streams']):], start=len(doc['source_streams'])):
        results, tapes, cues = points(weights, cohort['practice'], GRID, plan['calibration'])
        outcomes = results[..., 0]
        rng = np.random.default_rng(cohort['shuffle'])
        permutation = np.column_stack([rng.permutation(len(GRID)) for _ in cohort['practice']])
        fits = dict(pong=select(outcomes), shuffled=select(np.take_along_axis(outcomes, permutation, axis=0)))
        doc['source_streams'].append(dict(cohort=index, fits=fits, practice=results.tolist(), cue=cues.tolist(),
                                          action_sha256=tapes.tolist(), points=int(outcomes.size),
                                          live_native_frames=int(results[..., 2].sum())))
        _write(LOCK, doc)
        print('native workspace fit', index, fits, flush=True)
    return doc


def paired(a, b):
    delta = a[:, 0] - b[:, 0]
    rng = np.random.default_rng(355039949)
    interval = np.quantile(rng.choice(delta, (18000, len(delta))).mean(1), (.025, .975))
    return dict(mean=float(delta.mean()), interval=interval.tolist(), improve=int((delta > 0).sum()),
                ties=int((delta == 0).sum()), worsen=int((delta < 0).sum()))


def test(plan, weights, fitted):
    lock = _hash(LOCK)
    doc = dict(plan_sha256=_hash(PLAN), lock_sha256=lock, cohorts=[], courses={})
    if HELD.exists():
        old = _read(HELD)
        if old['plan_sha256'] != doc['plan_sha256'] or old['lock_sha256'] != lock:
            raise ValueError('stored workspace test differs from sealed source')
        doc = old
    for protocol, learned in zip(plan['cohorts'][len(doc['cohorts']):], fitted['source_streams'][len(doc['cohorts']):], strict=True):
        gains = {name: fit['gain'] for name, fit in learned['fits'].items()} | dict(zero=0., plus=1., minus=-1.)
        row = dict(index=learned['cohort'], gains=gains, courses={})
        for name in ('held_wide', 'held_ordinary'):
            results, tapes, cues = points(weights, protocol[name], list(gains.values()), plan['calibration'])
            rows = dict(zip(gains, results, strict=True))
            row['courses'][name] = dict(outcome={k: v.tolist() for k, v in rows.items()}, cue=cues.tolist(),
                                        action_sha256=dict(zip(gains, tapes.tolist(), strict=True)),
                                        summary={k: summary(v) for k, v in rows.items()})
        doc['cohorts'].append(row)
        for name in ('held_wide', 'held_ordinary'):
            pooled = {key: np.concatenate([c['courses'][name]['outcome'][key] for c in doc['cohorts']]) for key in gains}
            doc['courses'][name] = dict(summary={k: summary(v) for k, v in pooled.items()},
                                        paired={k: paired(pooled['pong'], v) for k, v in pooled.items() if k != 'pong'})
        _write(HELD, doc)
        print('native workspace held', row['index'], {name: {k: x['net'] for k, x in data['summary'].items()}
                                                     for name, data in row['courses'].items()}, flush=True)
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = register()
    weights = np.asarray(_read(WARM)['weights'])
    feasibility = pilot(protocol, weights)
    if feasibility['gate'] and not args.pilot_only:
        test(protocol, weights, learn(protocol, weights))

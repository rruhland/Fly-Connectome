"""Check whether wrist-only observed flight changes identify face-normal error."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from scipy.spatial.transform import Rotation

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy, commands
from .tt24_donor_trial import ROOT


TAG = 'wrist-impulse'
PLAN = ROOT / f'{TAG}-plan.json'
SCOUT = ROOT / f'{TAG}-scout.json.gz'
FIT = ROOT / f'{TAG}-fit.json.gz'
HELD = ROOT / f'{TAG}.json.gz'
REPLICATE_PLAN = ROOT / f'{TAG}-local-z-plan.json'
REPLICATE_SCOUT = ROOT / f'{TAG}-local-z-scout.json.gz'
REPLICATE_FIT = ROOT / f'{TAG}-local-z-fit.json.gz'
REPLICATE_HELD = ROOT / f'{TAG}-local-z.json.gz'
SEED = 1517054381
REPLICATE_SEED = 1088676920
CAP = 240
TILT = {'minus': -6., 'ordinary': 0., 'plus': 6.}
WINDOW = np.arange(-6, 7)
TIME = WINDOW / 60.
DESIGN = np.column_stack((np.ones(len(TIME)), TIME, np.maximum(0., TIME)))


def register(local_z=False):
    seed, count, path = (REPLICATE_SEED, 4, REPLICATE_PLAN) if local_z else (SEED, 2, PLAN)
    draws = (np.random.default_rng(seed).choice(2**31 - 2, count+12, replace=False) + 1).tolist()
    data = dict(schema='tt24-wrist-impulse-1', seed=seed, physical_scout=TILT, scouts=draws[:count],
                alternate_fit=draws[count:count+4], alternate_held=draws[count+4:count+10], stock_held=draws[-2:], cap=CAP,
                policy_sha256=_hash(BUNDLE), input='wrist RGB, own joint encoders and frozen optical calibration',
                scout_gate='at least three accepted optic jumps over both releases for each control, correct sign on either side of ordinary, >= .075 total world y separation',
                transfer_gate='at least two altered practice launches with events and at least five accepted events total',
                fixture_tilt='stock paddle geometry rotates about world x during controls through local body registration')
    if local_z:
        data.update(pilot_sha256=_hash(SCOUT), tangent='local z perpendicular to known normal',
                    stock_offset='subtract ordinary matched stock scout residual in local angle')
    if path.exists():
        if _read(path) != data:
            raise ValueError('wrist-impulse registration changed')
    else:
        _write(path, data)
    return data


def face_tangent(normal, local_z=False):
    if local_z:
        tangent = np.array((0., 0., 1.)) - normal[2] * normal
        return tangent / np.linalg.norm(tangent)
    return np.array((-normal[1], normal[0], 0.))


def rotate_face(normal, angle, local_z=False):
    c, s = np.cos(angle), np.sin(angle)
    return c * normal + s * face_tangent(normal, local_z)


def engineering_tilt(arena, seed, regime, degrees, original):
    """The environment changes the real blade on scout controls, outside inference."""
    model = arena._model
    model.geom_quat[arena._blade] = original
    if degrees:
        arena.reset(seed=seed, split='held', regime=regime, render=False)
        mujoco.mj_kinematics(model, arena._data)
        frame = arena._data.xmat[model.geom_bodyid[arena._blade]].reshape(3, 3)
        delta = Rotation.from_rotvec(np.deg2rad(degrees) * frame[0])
        local = Rotation.from_quat(np.roll(original, -1))
        model.geom_quat[arena._blade] = np.roll((delta * local).as_quat(), 1)


def observe_drop(arena, seed, regime, *, degrees=0., correction=0., collect=False, local_z=False):
    blade = arena._model.geom_quat[arena._blade].copy()
    engineering_tilt(arena, seed, regime, degrees, blade)
    try:
        obs = arena.reset(seed=seed, split='held', regime=regime)
        policy = CarriedWristPolicy()
        policy.begin_release(obs)
        eye = policy.sensor
        if correction and (not local_z or policy.calibration_index == 1):
            eye.decoder.normal = rotate_face(eye.decoder.normal, correction, local_z)
        traces, actions = [], []
        while True:
            gap = eye.gaps
            state = eye.observe(obs)
            if collect:
                point = (state[:3] + eye.home).tolist() if eye.gaps == gap else [None] * 3
                _, rotation = eye.decoder.links.read(obs['joints'][None])
                traces.append(dict(ball=point, blade=(state[6:9] + eye.home).tolist(), rotation=rotation[0].tolist()))
            action = commands(state, policy.weights, policy.issued)
            obs, _, done = arena.step(action)
            actions.append(action.tolist())
            if done:
                break
        judgment = arena.judge_metrics()
        arena.reset(seed=seed, split='held', regime=regime, render=False)
        for action in actions:
            arena.step(action, render=False)
        if arena.judge_metrics() != judgment:
            raise ValueError('wrist impulse action replay changed the actual referee')
        return dict(seed=seed, calibration_index=policy.calibration_index, traces=traces, normal=eye.decoder.normal.tolist(),
                    evaluation=judgment, actions=actions)
    finally:
        arena._model.geom_quat[arena._blade] = blade


def apparent_acceleration(rolls):
    increments = []
    for roll in rolls:
        position = np.asarray([row['ball'] for row in roll['traces']], dtype=float)
        rates = (position[4:] - 2 * position[2:-2] + position[:-4]) * (30.**2)
        usable = np.isfinite(rates).all(axis=1) & (np.linalg.norm(rates, axis=1) < 26.)
        increments.extend(rates[usable])
    return np.median(increments, axis=0) if len(increments) >= 20 else None


def witnesses(roll, acceleration, local_z=False):
    if acceleration is None:
        return []
    positions = np.asarray([row['ball'] for row in roll['traces']], dtype=float)
    blades = np.asarray([row['blade'] for row in roll['traces']])
    n = np.asarray(roll['normal'])
    tangent = face_tangent(n, local_z)
    candidates = []
    for tick in range(6, len(positions) - 6):
        window = positions[tick + WINDOW]
        valid = np.isfinite(window).all(axis=1)
        if (valid & (WINDOW < 0)).sum() < 4 or (valid & (WINDOW > 0)).sum() < 4:
            continue
        clipped = DESIGN[valid]
        shifted = window[valid] - .5 * TIME[valid, None]**2 * acceleration
        position, incoming, jump = np.linalg.lstsq(clipped, shifted, rcond=None)[0]
        outgoing = incoming + jump
        relative = position - blades[tick]
        residual = float(np.sqrt(np.mean((shifted - clipped @ np.stack((position, incoming, jump)))**2)))
        if (incoming[2] >= -.44 or outgoing[2] <= .35 or jump[2] < 1.15 or residual > .018
                or abs(relative[2]) > .095 or np.linalg.norm(relative[:2]) > .068
                or np.linalg.norm(jump[:2]) > .52 * jump[2]):
            continue
        rotation = np.asarray(roll['traces'][tick]['rotation'])
        sign = 1. if (rotation @ n)[2] >= 0. else -1.
        local = sign * rotation.T @ (jump / np.linalg.norm(jump))
        blade_speed = (blades[tick+2] - blades[tick-2]) * 15.
        slip = (incoming - blade_speed) @ (rotation @ tangent)
        if local @ n < .7 or abs(slip) > .35 * np.linalg.norm(jump):
            continue
        angle = np.arctan2(local @ tangent, local @ n)
        before = sign * rotation @ n
        inferred = sign * rotation @ rotate_face(n, angle, local_z)
        candidates.append(dict(tick=tick, angle=float(angle), fit_residual_m=residual, slip=float(slip),
                               delta_y=float(inferred[1]-before[1]), impulse=jump.tolist()))
    events = []
    for candidate in sorted(candidates, key=lambda row: row['fit_residual_m']):
        if all(abs(row['tick']-candidate['tick']) >= 11 for row in events):
            events.append(candidate)
    return sorted(events, key=lambda row: row['tick'])


def optical_fit(rolls, local_z=False):
    acceleration = apparent_acceleration(rolls)
    impulse = [witnesses(roll, acceleration, local_z) for roll in rolls]
    # Each release receives the same weight even if it survives longer.
    per_drop = np.array([[np.median([r['angle'] for r in group]), np.median([r['delta_y'] for r in group])]
                         for group in impulse if group])
    value = np.median(per_drop, axis=0).tolist() if len(per_drop) else [None, None]
    return dict(acceleration=acceleration.tolist() if acceleration is not None else None, events=impulse,
                count=sum(map(len, impulse)), positive_drops=len(per_drop), raw_angle=value[0], raw_world_y=value[1])


def scout_task(task):
    name, degrees, seeds, local_z = task
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=CAP)
    try:
        data = [observe_drop(arena, seed, 'standard', degrees=degrees, collect=True) for seed in seeds]
        result = optical_fit(data, local_z)
        print('single wrist physical control', name, result['count'], result['raw_world_y'], flush=True)
        return name, dict(optic=result, data=data)
    finally:
        arena.close()


def scout(plan, workers, local_z):
    registration, path = (REPLICATE_PLAN, REPLICATE_SCOUT) if local_z else (PLAN, SCOUT)
    if path.exists():
        result = _read(path)
        if result['plan'] != _hash(registration):
            raise ValueError('wrist scout uses a different plan')
        return result
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        controls = dict(pool.map(scout_task, [(name, degrees, plan['scouts'], local_z) for name, degrees in TILT.items()]))
    fits = {name: row['optic'] for name, row in controls.items()}
    support = all(row['count'] >= 3 and row['positive_drops'] >= 2 for row in fits.values())
    sign = False
    if support:
        lower, zero, higher = (fits[name]['raw_world_y'] for name in ('plus', 'ordinary', 'minus'))
        sign = lower + .015 <= zero <= higher - .015 and higher - lower >= .075
    result = dict(plan=_hash(registration), controls=controls, support=support, signed_effect=sign,
                  advance_to_altered=support and sign)
    _write(path, result)
    print('WRIST IMPULSE SCOUT', support, sign, flush=True)
    return result


def fit(plan, source, local_z):
    registration, scout_path, path = (REPLICATE_PLAN, REPLICATE_SCOUT, REPLICATE_FIT) if local_z else (PLAN, SCOUT, FIT)
    if path.exists():
        data = _read(path)
        if data['plan'] != _hash(registration) or data['scout'] != _hash(scout_path):
            raise ValueError('frozen orientation fit has a different scout')
        return data
    arm = BounceArena(view='wrist', width=192, height=144, max_steps=CAP, arm='alternate')
    try:
        rows = [observe_drop(arm, seed, 'changed', collect=True) for seed in plan['alternate_fit']]
    finally:
        arm.close()
    estimate = optical_fit(rows, local_z)
    eligible = (estimate['positive_drops'] >= 2 and estimate['count'] >= 5 and
                all(row['calibration_index'] == 1 for row in rows))
    offset = source['controls']['ordinary']['optic']['raw_angle'] if local_z else 0.
    angle = 0.
    if eligible:
        shrink = estimate['positive_drops'] / (1 + estimate['positive_drops'])
        angle = float(np.clip(shrink * (estimate['raw_angle'] - offset), -.14, .14))
    record = dict(plan=_hash(registration), scout=_hash(scout_path), optic=estimate, rows=rows,
                  stock_offset_rad=offset, correction_rad=angle, advance_to_held=eligible and abs(angle) >= .02)
    _write(path, record)
    print('FROZEN WRIST IMPULSE CORRECTION', estimate['count'], angle, flush=True)
    return record


def game(task):
    body, seed, correction, local_z = task
    arena = BounceArena(view='wrist', width=192, height=144, max_steps=CAP, arm=body)
    try:
        regime = 'changed' if body == 'alternate' else 'standard'
        modes = dict(frozen=0., learned=correction)
        if body == 'alternate':
            modes['reversed'] = -correction
        values = {name: observe_drop(arena, seed, regime, correction=angle, local_z=local_z)
                  for name, angle in modes.items()}
        print('wrist matched direction', body, seed,
              {n: row['evaluation']['longest_consecutive_rebounds'] for n, row in values.items()}, flush=True)
        return dict(body=body, seed=seed, results=values)
    finally:
        arena.close()


def held(plan, data, workers, local_z):
    registration, fit_path, path = (REPLICATE_PLAN, REPLICATE_FIT, REPLICATE_HELD) if local_z else (PLAN, FIT, HELD)
    tasks = [(body, seed, data['correction_rad'], local_z) for body in ('alternate', 'stock') for seed in plan[body+'_held']]
    output = dict(plan=_hash(registration), correction=_hash(fit_path), trials=[])
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for row in pool.map(game, tasks):
            output['trials'].append(row)
            _write(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--local-z', action='store_true', help='run independent, registered local z replication')
    args = parser.parse_args()
    plan = register(args.local_z)
    source = scout(plan, args.workers, args.local_z)
    if source['advance_to_altered']:
        corrected = fit(plan, source, args.local_z)
        if corrected['advance_to_held']:
            held(plan, corrected, args.workers, args.local_z)


if __name__ == '__main__':
    main()

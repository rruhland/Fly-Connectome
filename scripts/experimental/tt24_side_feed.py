"""Test frozen native-Pong lateral carry on physical side-fed paddle bounces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena, mujoco
from .tt24_bounce_ident import CleanState, _INC
from .tt24_donor_trial import Native
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_world_arm import DIR, Grounding
from .tt24_world_prior import WorldPrior


TAG = 'tt24_side_feed'
PLAN = DIR / f'{TAG}-plan.json'
FEASIBILITY = DIR / f'{TAG}-feasible.json.gz'
OUTPUT = DIR / f'{TAG}.json.gz'
DONOR = DIR / 'tt24_donor-refined.npz'
DONOR_RECEIPT = DIR / 'tt24_donor-refined.json.gz'
BANK = DIR / 'pong-world-strong-bank.json'
MODES = ('pong', 'shuffled', 'cold', 'matched_cold')
CONTROLS = ('passive', 'clean_donor', 'wrist_donor')
BANDS = ((.082, .099), (.068, .083), (.096, .113))
DIRECTIONS = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))
SEED = 247760663
STREAMS = 6
CAL_SIZE = 8
STEPS = 240
OPEN_LIMIT = 20
JOINT_BLEND = .40
STOCK_SUPPORT = 2


def register():
    bank = _read(BANK)
    donor = _read(DONOR_RECEIPT)
    if not donor['donor_gate_passed'] or _hash(DONOR) != donor['frozen_sha256']:
        raise ValueError('use the locked, independently qualified donor-only head')
    rng = np.random.default_rng(SEED)
    # The indexed direction is assigned in advance, with one drop from each
    # direction per held source-bank stream and two per feasibility band.
    launches = rng.choice(2**31 - 2, len(BANDS) * CAL_SIZE + STREAMS * 4, replace=False) + 1
    feasibility = [[[int(seed), index % 4] for index, seed in enumerate(group)]
                   for group in launches[:len(BANDS) * CAL_SIZE].reshape(-1, CAL_SIZE)]
    held = [[[int(seed), index] for index, seed in enumerate(group)]
            for group in launches[len(BANDS) * CAL_SIZE:].reshape(-1, 4)]
    bank_files = []
    for stream in bank['streams'][:STREAMS]:
        files = stream['policy_files']
        if any(_hash(DIR / files[mode]['file']) != files[mode]['sha256'] for mode in MODES):
            raise ValueError('an original historical checkpoint changed')
        bank_files.append(files)
    plan = dict(schema='tt24-side-feed-1', seed=SEED, feasibility=feasibility, held=held,
                modes=list(MODES), references=list(CONTROLS), bands=[list(b) for b in BANDS],
                directions=[list(d) for d in DIRECTIONS], judge='unchanged stock BounceArena',
                horizon=STEPS, opening_limit=OPEN_LIMIT, decay=JOINT_BLEND, donor=_hash(DONOR),
                donor_receipt=_hash(DONOR_RECEIPT), observer=_hash(FUSED_OUTPUT), bank=bank_files,
                qualification='first band with passive opening catches <= 4/8, clean donor >= 3/8 and above passive, wrist donor catches >= 2 cases missed by passive and two impact-1 consecutive catches >= 2/8',
                primary='a physically qualified rebound from the first blade impact',
                secondary='consecutive qualified rebounds anchored at that first impact', new_target_motor_practice=0)
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('the locked side-feed schedule changed')
    else:
        _write(PLAN, plan)
    return plan


class SideFeed(BounceArena):
    """Change the release only. Keep the stock plant, collisions and judge."""

    def reset(self, *, seed=None, direction, band, render=True):
        super().reset(seed=seed, split='held', regime='standard', render=False)
        rng = np.random.default_rng(seed + 3176881)
        heading = np.array(DIRECTIONS[direction])
        side = np.array((-heading[1], heading[0]))
        angle = rng.uniform(-.10, .10)
        axis = np.cos(angle) * heading + np.sin(angle) * side
        height = rng.uniform(.22, .30)
        speed = rng.uniform(.16, .25)
        falling = rng.uniform(-.12, -.015)
        drift = rng.uniform(-.015, .015)
        # Set a distribution of lateral landings with a free-flight estimate.
        # Only the environment uses this estimate. Contacts use the native judge.
        floor = .023
        flight = (falling + np.sqrt(falling**2 + 2. * 9.81 * (height - floor))) / 9.81
        landing = rng.uniform(*band)
        offset = np.r_[axis * (landing + speed * flight), height]
        velocity = np.r_[-axis * speed + side * drift, falling]
        self._drop = replace(self._drop, offset=tuple(offset), velocity=tuple(velocity))
        self._ball_position[:] = self._data.geom_xpos[self._blade] + offset
        self._ball_velocity[:] = velocity
        mujoco.mj_forward(self._model, self._data)
        return self._observation(render)


def first_run(events):
    next_impact = 1
    for event in events:
        if event['impact_index'] != next_impact:
            break
        next_impact += 1
    return next_impact - 1


def lateral_projection(sensor, home):
    ground = Grounding(sensor.decoder, home)
    jacobian = ground.jacobian(ground.home) / ground.scale[:, None]
    return np.linalg.solve(jacobian.T @ jacobian + .24 * np.eye(5), jacobian[:2].T) / ground.scale[:2]


def play(arena, native, weights, release, band, mode, prior=None):
    seed, direction = release
    public = mode not in ('passive', 'clean_donor')
    reset = dict(seed=seed, direction=direction, band=band)
    obs = arena.reset(**reset, render=public)
    sensor = load_public(FUSED_OUTPUT) if public else CleanState(arena) if mode == 'clean_donor' else None
    projection = lateral_projection(sensor, obs['joints']) if prior is not None else None
    virtual = np.zeros(5)
    issued = np.zeros(5)
    joint_lateral = np.zeros(5)
    donor_action = np.zeros(5)
    descending, rising = False, False
    tape = []
    for tick in range(arena.max_steps):
        if sensor is not None:
            state = np.ascontiguousarray(sensor.observe(obs), dtype=np.float64)
            native.command(state.ctypes.data, weights.ctypes.data, virtual.ctypes.data, donor_action.ctypes.data)
        move = donor_action
        if prior is not None:
            descending |= state[5] < -.26
            rising |= tick > 5 and descending and state[5] > .24
            active = tick < OPEN_LIMIT and not rising
            desired = np.clip(projection @ prior.target(state)[:2], -.18, .18) if active else np.zeros(5)
            joint_lateral += JOINT_BLEND * (desired - joint_lateral)
            move = np.clip((virtual + joint_lateral - issued) / _INC, -.65, .65)
            issued += _INC * move
        tape.append(move.tolist())
        obs, _, done = arena.step(move, render=public)
        if done:
            break
    judge = arena.judge_metrics()
    arena.reset(**reset, render=False)
    for move in tape:
        arena.step(move, render=False)
    if arena.judge_metrics() != judge:
        raise ValueError(f'physical replay differs for seed {seed} and treatment {mode}')
    opening = first_run(judge['rebound_events'])
    return dict(release=release, first=opening >= 1, second=opening >= 2, opening_run=opening,
                score=judge['rebounds'], longest=judge['longest_consecutive_rebounds'], judge=judge, actions=tape)


def summarize(rows):
    return {mode: dict(first=sum(row['first'] for row in values), second=sum(row['second'] for row in values),
                       opening_run=sum(row['opening_run'] for row in values), longest=sum(row['longest'] for row in values),
                       total=sum(row['score'] for row in values), drops=len(values)) for mode, values in rows.items()}


def evaluate(task):
    releases, band, modes, files = task
    native = Native()
    arena = SideFeed(view='wrist', width=192, height=144, max_steps=STEPS)
    try:
        with np.load(DONOR, allow_pickle=False) as ckpt:
            weights = np.ascontiguousarray(ckpt['weights'], dtype=np.float64)
        output = {}
        for mode in modes:
            prior = WorldPrior.load(DIR / files[mode]['file']) if mode in files else None
            output[mode] = [play(arena, native, weights, release, band, mode, prior) for release in releases]
        return output
    finally:
        arena.close()
        native.working.cleanup()


def feasible(plan):
    if FEASIBILITY.exists():
        receipt = _read(FEASIBILITY)
        if receipt['plan'] != _hash(PLAN):
            raise ValueError('the feasibility cohort was changed')
        return receipt['chosen']
    receipt = dict(plan=_hash(PLAN), tested=[], chosen=None)
    for index, (releases, band) in enumerate(zip(plan['feasibility'], plan['bands'], strict=True)):
        rows = evaluate((releases, band, CONTROLS, {}))
        result = summarize(rows)
        missed_passive = sum(learned['first'] and not passive['first'] for learned, passive in
                             zip(rows['wrist_donor'], rows['passive'], strict=True))
        clean = result['clean_donor']['first']
        passive = result['passive']['first']
        passed = (passive <= 4 and clean >= 3 and clean > passive and missed_passive >= STOCK_SUPPORT
                  and result['wrist_donor']['second'] >= STOCK_SUPPORT)
        receipt['tested'].append(dict(index=index, band=band, results=result, qualified=passed,
                                      catches_beyond_passive=missed_passive, rows=rows))
        print('feasible?', index, band, result, 'public catches beyond passive', missed_passive, passed, flush=True)
        if passed:
            receipt['chosen'] = index
            break
    _write(FEASIBILITY, receipt)
    return receipt['chosen']


def test(plan, band_index, workers):
    receipt = (_read(OUTPUT) if OUTPUT.exists() else
               dict(plan=_hash(PLAN), feasibility=_hash(FEASIBILITY), band_index=band_index, groups=[]))
    if (receipt['plan'], receipt['feasibility'], receipt['band_index']) != (_hash(PLAN), _hash(FEASIBILITY), band_index):
        raise ValueError('the frozen side-feed benchmark changed')
    offset = len(receipt['groups'])
    tasks = [(row, plan['bands'][band_index], (*MODES, *CONTROLS), files)
             for row, files in zip(plan['held'][offset:], plan['bank'][offset:], strict=True)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for index, rows in enumerate(pool.map(evaluate, tasks), start=offset):
            receipt['groups'].append(dict(stream=index, summary=summarize(rows), rows=rows))
            _write(OUTPUT, receipt)
            print('held source stream', index, receipt['groups'][-1]['summary'], flush=True)
    combined = {mode: [r for group in receipt['groups'] for r in group['rows'][mode]] for mode in (*MODES, *CONTROLS)}
    receipt['summary'] = summarize(combined)
    receipt['passive_misses'] = summarize({mode: [row for row, passive in zip(rows, combined['passive'], strict=True)
                                                  if not passive['first']] for mode, rows in combined.items()})
    receipt['primary_paired_streams'] = {
        mode: [g['summary']['pong']['first'] - g['summary'][mode]['first'] for g in receipt['groups']]
        for mode in ('shuffled', 'cold', 'matched_cold', 'wrist_donor')}
    _write(OUTPUT, receipt)
    print('paired summary', receipt['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--feasibility-only', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    plan = register()
    if args.plan_only:
        return
    selected = feasible(plan)
    if selected is None:
        print('no registered release band passed physical feasibility', flush=True)
    elif not args.feasibility_only:
        test(plan, selected, args.workers)


if __name__ == '__main__':
    main()

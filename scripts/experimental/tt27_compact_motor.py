"""Drive live compact-court points through a copied regulation encoder-probe map."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES, RATES
from .tt22_rally_trial import alternating
from .tt24_rally_trial import EYES, PublicBody, STEP_SIZE
from .tt25_arm_replay import NOMINAL_NO_LOAD
from .tt25_reference_receiver import ReferenceReceiver
from .tt27_compact import MOVIE_SEED, OUTPUT, CompactStation, Options, audit, brief
from .tt27_motor_interface import AXIS_COUNT, IDENTITY, Interface


SOURCE = OUTPUT / 'motor-active-serial.json.gz'
ROSTER_SEED = 997441215
PROBE_STREAM = 53017817
SETTINGS = (2, 5)
MODES = ('estimated', 'stale', 'known', 'intact')


def interface(record):
    return Interface(tuple(record['permutation']), tuple(record['signs']), record['delay'])


def hardware(arena):
    model = arena.model
    actuators = [model.actuator('learner_' + key).id for key in AXES]
    joints = [model.joint('learner_' + key).id for key in AXES]
    return dict(axes=list(AXES), nominal_rates=list(RATES), tick=arena.control_dt,
                integrator_step=model.opt.timestep, home=list(arena._arm_home(1)[0]),
                limits=model.jnt_range[joints].tolist(), stiffness=model.actuator_gainprm[actuators].tolist(),
                bias=model.actuator_biasprm[actuators].tolist(),
                force_ranges=model.actuator_forcerange[actuators].tolist())


def confirm_hardware():
    full = ReferenceReceiver()
    compact = CompactStation.make_arena('rebound')
    try:
        matching = hardware(full)
        if matching != hardware(compact):
            raise ValueError('regulation and compact learner motor hardware differ')
        return matching
    finally:
        full.close()
        compact.close()


class PhysicalRoute:
    """External simulator receiver of wire requests, including a real one-tick buffer."""

    def __init__(self, actual):
        self.actual = actual
        self.previous = np.zeros(AXIS_COUNT)

    def execute(self, request):
        wire = request[:AXIS_COUNT]
        previous = self.previous
        self.previous = wire.copy()
        driven = self.actual.apply(previous if self.actual.delay else wire)
        return np.r_[driven, request[-1]]


class Decoder:
    """Infer pending physical action from the supplied estimate and own wire history."""

    def __init__(self, arena, option, estimate, lead):
        self.body = PublicBody(arena)
        self.history, self.actor = option.history, option.actor
        self.estimate = estimate
        self.previous = np.zeros(AXIS_COUNT)
        self.lead = lead * estimate.delay

    def issue(self, frames, joints):
        state = self.history.observe(frames, joints, self.body.target)
        pending = self.estimate.apply(self.previous) * self.estimate.delay
        next_position = self.body.target.copy()
        next_position[:AXIS_COUNT] = np.clip(next_position[:AXIS_COUNT] + STEP_SIZE[:AXIS_COUNT] * pending,
                                             self.body.lower[:AXIS_COUNT], self.body.upper[:AXIS_COUNT])
        if self.lead:
            for eye in (0, 7):
                if state[eye+6] <= 1:
                    state[eye:eye+2] += self.lead * state[eye+2:eye+4]
            if state[49] <= 1:
                advance = self.lead * state[41:44]
                state[38:41] += advance
                state[50:53] += advance
        wanted = self.actor.command(state, next_position)
        wire = self.estimate.invert(wanted[:AXIS_COUNT])
        inferred_applied = pending if self.estimate.delay else self.estimate.apply(wire)
        applied = np.r_[inferred_applied, wanted[-1]]
        self.history.issued(applied)
        self.body.issued(applied)
        self.previous = wire
        return np.r_[wire, wanted[-1]], wanted, applied


def play(station, options, start, actual, estimate, lead=0.):
    arena = station.arena
    arena.lane = start['lane']
    arena.reset(seed=start['seed'], server=-1, render=False)
    station.tape.steps.clear()
    decoder = Decoder(arena, options.new('combined'), estimate, lead)
    route = PhysicalRoute(actual)
    actions, requests, desired, expected, speeds = [], [], [], [], []
    first_rgb = sha256()
    for tick in range(arena.max_steps):
        frames = arena.render_cameras()
        if tick == 0:
            for eye in EYES:
                first_rgb.update(frames[eye])
        request, wanted, projected = decoder.issue(frames, arena.joints)
        applied = route.execute(request)
        _, reward, done, cap, _ = arena.step(applied, render=False)
        actions.append(applied.tolist())
        requests.append(request.tolist())
        desired.append(wanted.tolist())
        expected.append(projected.tolist())
        speeds.append(arena.data.qvel[station.axes].copy())
        if done or cap:
            break
    replay_route = PhysicalRoute(actual)
    reconstructed = [replay_route.execute(np.asarray(q)).tolist() for q in requests]
    if reconstructed != actions:
        raise ValueError('wire-tape replay changed the executed joint rates')
    if estimate == actual and expected != actions:
        raise ValueError('matched estimate did not reconstruct its true actuation')
    active = np.asarray(actions)[:, :AXIS_COUNT]
    signs = np.where(abs(active) < .1, 0, np.sign(active))
    measured = np.abs(np.asarray(speeds))
    result = dict(start, reward=reward, outcome=arena.point_result(), longest=alternating(arena.events),
                  events=arena.events.copy(), actions=actions, rival_actions=station.tape.tape(len(actions)),
                  first_rgb=first_rgb.hexdigest(), wire_requests=requests, desired_rates=desired,
                  expected_joint_rates=expected, naturalness=dict(
                      reversals=int(np.count_nonzero(signs[1:]*signs[:-1] < 0.)),
                      mean_rate_change=float(np.abs(np.diff(active, axis=0)).mean()),
                      peak_motor_speed=float(measured.max()), joint_speed_p95=float(np.quantile(measured, .95)),
                      over_nominal=int(np.count_nonzero(measured.max(axis=1) > NOMINAL_NO_LOAD))))
    station.verify(result)
    return result


def course(count):
    rng = np.random.default_rng(ROSTER_SEED)
    releases = {}
    for setting in SETTINGS:
        cases = []
        for index in range(count):
            offset = rng.uniform(.07, .30) if index % 2 == 0 else rng.uniform(.27, .59)
            lane = .16 + (-1 if index % 3 < 2 else 1)*offset
            cases.append(dict(seed=int(rng.integers(1, 2**31 - 1)), lane=float(lane), role='fresh'))
        releases[setting] = cases
    paths = ('compact-assessment.json.gz', 'compact-development.json.gz', 'compact_camera-assessment.json.gz',
             'compact_camera-tracking-held.json.gz')
    proposed = {row['seed'] for cases in releases.values() for row in cases}
    for name in paths:
        data = _read(OUTPUT / name)
        if 'cases' in data:
            seeds = {row['seed'] for stream in data['cases'].values() for block in stream.values()
                     for case in block for row in case.values()}
        else:
            seeds = {row['seed'] for block in data['blocks'] for row in block['rows']}
        if proposed & seeds:
            raise ValueError('new motor course overlaps a compact assessment')
    archives = audit([releases])
    previous = _read(OUTPUT / paths[0])
    diagnostic = next(case['retained'] for stream in previous['cases'].values() for case in stream['rebound']
                      if case['retained']['seed'] == MOVIE_SEED)
    releases[SETTINGS[0]].append(dict(seed=diagnostic['seed'], lane=diagnostic['lane'], role='prior_7_plus_7'))
    return releases, archives


def score(block):
    fit, starts = block
    actual = interface(fit['actual'])
    estimates = {name: interface(fit['estimates'][label]['interface'])
                 for name, label in (('estimated', 'active'), ('stale', 'frozen'))}
    estimates.update(known=actual, intact=IDENTITY)
    setting = fit['number']
    station = CompactStation('rebound')
    options = Options()
    cases = []
    try:
        for start in starts:
            results = {}
            for mode in MODES:
                physical = IDENTITY if mode == 'intact' else actual
                results[mode] = play(station, options, start, physical, estimates[mode])
                row = results[mode]
                print(setting, start['seed'], start['role'], mode, row['outcome']['legal_returns'],
                      row['outcome']['rival_returns'], row['longest'], flush=True)
            if len({row['first_rgb'] for row in results.values()}) != 1:
                raise ValueError('remap changed initial optical data')
            if estimates['estimated'] == actual and results['estimated'] != results['known']:
                raise ValueError('exact learned map diverged from known-map closed-loop control')
            if actual.delay == 0 and estimates['estimated'] == actual:
                for key in ('actions', 'rival_actions', 'events', 'outcome'):
                    if results['estimated'][key] != results['intact'][key]:
                        raise ValueError(f'zero-delay inverse failed to reproduce native {key}')
            cases.append(dict(start, results=results))
        return dict(setting=setting, fixture=fit['actual'], estimates={mode: asdict(value)
                    for mode, value in estimates.items()}, confirm_tick=fit['active_trace']['confirm_tick'], cases=cases)
    finally:
        station.close()


def summary(blocks, role):
    return {mode: brief([case['results'][mode] for block in blocks for case in block['cases'] if case['role'] == role])
            for mode in MODES}


def experiment(path, count, workers):
    source = _read(SOURCE)
    stream = next(item for item in source['streams'] if item['seed'] == PROBE_STREAM)
    releases, checked = course(count)
    tasks = [(next(fit for fit in stream['settings'] if fit['number'] == i), releases[i]) for i in SETTINGS]
    meta = dict(schema='tt27-compact-motor-1', regulation_receipt=SOURCE.name, regulation_sha256=_hash(SOURCE),
                motor_hardware=confirm_hardware(), receipt_stream=PROBE_STREAM, settings=list(SETTINGS),
                source_probe_ticks=source['prep_ticks'], source_home_resets=source['restarts_each_setting'],
                source_clock_s=source['receiver_clock_s'], releases=releases, checked_archives=checked,
                target_exposure='no compact probing or updating from point outcomes',
                motor_delay='one temporal FIFO per remapped actuator interface; pending command projected forward')
    saved = []

    def save(result):
        saved.append(result)
        saved.sort(key=lambda block: block['setting'])
        report = dict(metadata=meta, blocks=saved, fresh=summary(saved, 'fresh'),
                      verified_tapes=sum(len(block['cases']) for block in saved)*len(MODES))
        if any(case['role'] == 'prior_7_plus_7' for block in saved for case in block['cases']):
            report['diagnostic'] = summary(saved, 'prior_7_plus_7')
        _write(path, report)
        return report

    if workers == 1:
        for task in tasks:
            report = save(score(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(score, task) for task in tasks]
            for future in as_completed(futures):
                report = save(future.result())
    print(report['fresh'], flush=True)


def delay_diagnostic(path):
    """Retry two known responsive intact points with causal stereo extrapolation."""
    source = _read(SOURCE)
    stream = next(item for item in source['streams'] if item['seed'] == PROBE_STREAM)
    fit = next(item for item in stream['settings'] if item['number'] == SETTINGS[-1])
    actual = interface(fit['actual'])
    inferred = interface(fit['estimates']['active']['interface'])
    releases, _ = course(3)
    cases = [releases[SETTINGS[-1]][0], releases[SETTINGS[0]][-1]]
    station = CompactStation('rebound')
    options = Options()
    trials = []
    try:
        for start in cases:
            results = {}
            for lead in (.5, 1.):
                row = play(station, options, start, actual, inferred, lead)
                results[str(lead)] = row
                print('timing diagnostic', start['seed'], lead, row['longest'], flush=True)
            trials.append(dict(start, results=results))
    finally:
        station.close()
    _write(path, dict(regulation_sha256=_hash(SOURCE), regulation_setting=SETTINGS[-1],
                      estimated_map=asdict(inferred), cases=trials, verified_tapes=2*len(trials),
                      scope='diagnostic retries chosen because the intact controller returned'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', type=Path, default=OUTPUT / 'compact_motor.json.gz')
    p.add_argument('--delay-diagnostic', action='store_true')
    args = p.parse_args()
    if args.count < 1:
        p.error('--count must be positive')
    if args.delay_diagnostic:
        delay_diagnostic(OUTPUT / 'compact_motor-delay.json.gz')
    else:
        experiment(args.output, args.count, args.workers)

"""Compare continuous joint feedback with a frozen near-arm path after a flight kick."""

import argparse
import gzip
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from . import tt24_rally_bank as bank
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, SequenceArena, course
from .tt24_rally_from_release import OPENING
from .tt24_rally_trial import COACHED, DATA, OUTPUT, alternating, contacts, returns


DESTINATION = OUTPUT / 'tt24_disturbance.json.gz'
COHORTS = {'g': 1486380259, 'h': 706208491}
ASSIGNMENT_SEED = 86801942
KICKS = (-.30, -.22, -.14, .14, .22, .30)
OPENING_KICK_TICK = 6
REPLY_SEPARATION_TICKS = 2
EARLY_MOTOR_WINDOW = 21
RATE_THRESHOLD = 1.e-4


def _design():
    starts = course(COHORTS)
    rng = np.random.default_rng(ASSIGNMENT_SEED)
    for cohort in COHORTS:
        for stratum in sorted({row['stratum'] for row in starts}):
            group = [row for row in starts if (row['cohort'], row['stratum']) == (cohort, stratum)]
            for row, kick in zip(group, rng.permutation(KICKS), strict=True):
                row['velocity_delta_x'] = float(kick)
    return starts


def _check_fresh(starts):
    seeds = {row['seed'] for row in starts}
    if len(starts) != len(seeds):
        raise ValueError('disturbance releases repeat a seed')
    for path in (bank.CACHE, OPENING, DATA):
        with np.load(path) as arrays:
            if seeds.intersection(map(int, arrays['seed'])):
                raise ValueError(f'disturbance release occurs in the teacher data at {path}')
    pattern = re.compile(rb'"seed"\s*:\s*(?:' + b'|'.join(str(seed).encode() for seed in seeds) + rb')\b')
    for folder in (bank.ARCHIVE.with_name('tt22'), bank.ARCHIVE, OUTPUT):
        for path in folder.iterdir():
            if path.name.startswith('tt24_disturbance'):
                continue
            if path.name.endswith('.json.gz'):
                content = gzip.decompress(path.read_bytes())
            elif path.suffix == '.json':
                content = path.read_bytes()
            else:
                continue
            if pattern.search(content):
                raise ValueError(f'disturbance release occurs in a prior evaluation at {path}')


def _state(arena):
    data = arena.data
    return sha256(np.r_[data.time, data.qpos, data.qvel, data.ctrl].astype('<f8').tobytes()).hexdigest()


class FlightTrace:
    """Audit sampled states and apply a single physical lateral impulse."""

    def __init__(self, kick=None):
        self.kick = kick
        self.packet = None
        self.states = []
        self.positions = []

    def before_step(self, arena):
        tick = arena.steps
        if tick != len(self.states):
            raise ValueError('expected one audit sample before each control step')
        self.states.append(_state(arena))
        self.positions.append(arena.ball.tolist())
        if self.kick is None or tick != self.kick['tick']:
            return
        if arena._previous_touches or arena.velocity[1] >= -.25:
            raise ValueError(f'disturbance at tick {tick} was outside a separated inbound flight')
        velocity = arena.velocity.copy()
        arena.data.qvel[arena._ball_dof] += self.kick['velocity_delta_x']
        self.packet = dict(self.kick, position=arena.ball.tolist(), velocity_before=velocity.tolist(),
                           velocity_after=arena.velocity.tolist(), prestate=self.states[-1])

    def finish(self, arena):
        if self.kick is not None and self.packet is None:
            raise ValueError('point terminated before the declared kick')
        states = [*self.states, _state(arena)]
        return dict(kick=self.packet, state_sha256=sha256(''.join(states).encode()).hexdigest())


def _live(station, pupil, start, kick=None):
    arena = station.arena
    trace = FlightTrace(kick)
    render = arena.render_cameras

    def sampled_rgb():
        trace.before_step(arena)
        return render()

    arena.render_cameras = sampled_rgb
    try:
        row = station.play(start, 'broad', pupil, live_opening=True)
    finally:
        arena.render_cameras = render
    return row | trace.finish(arena), trace


def _fixed(station, start, nominal, kick=None):
    arena, tape = station.arena, station.tape
    trace = FlightTrace(kick)
    arena.lane = start['lane']
    arena.reset(seed=start['seed'], server=-1, render=False)
    tape.steps.clear()
    actions = []
    freeze = np.zeros(arena.action_size).tolist()
    for tick in range(arena.max_steps):
        trace.before_step(arena)
        action = nominal['actions'][tick] if tick < len(nominal['actions']) else freeze
        _, reward, done, cap, _ = arena.step(action, render=False)
        actions.append(action)
        if done or cap:
            break
    row = dict(seed=start['seed'], lane=start['lane'], reward=reward, outcome=arena.point_result(),
               longest=alternating(arena.events), events=arena.events.copy(), actions=actions,
               rival_actions=tape.tape(len(actions)), zero_suffix_steps=max(0, len(actions) - len(nominal['actions'])))
    return row | trace.finish(arena), trace


def _verify(arena, row, kick=None, *, capture=None):
    trace = FlightTrace(kick)
    arena.lane = row['lane']
    arena.reset(seed=row['seed'], server=-1, render=False)
    for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), 1):
        trace.before_step(arena)
        if capture is not None:
            capture(arena, tick - 1)
        rate = np.asarray(far)
        arena._aim = arena._command_to = lambda *args, rate=rate: rate
        _, score, done, cap, _ = arena.step(near, render=False)
        if bool(done or cap) != (tick == len(row['actions'])):
            raise ValueError(f'physical replay changed terminal tick on {row["seed"]}')
    if (trace.finish(arena) != {key: row[key] for key in ('kick', 'state_sha256')} or score != row['reward'] or
            arena.point_result() != row['outcome'] or digest_json(arena.events) != digest_json(row['events'])):
        raise ValueError(f'physical replay changed the trajectory or referee record on {row["seed"]}')
    if capture is not None:
        capture(arena, len(row['actions']))


def _reply_tick(nominal):
    legal = returns(nominal, 1)
    if not legal:
        return None
    far = next((tick for tick in contacts(nominal, -1) if tick > legal[0]), None)
    if far is not None and far + REPLY_SEPARATION_TICKS < len(nominal['actions']):
        return far + REPLY_SEPARATION_TICKS
    return None


def _affected_metrics(live, fixed, nominal, fixed_trace, nominal_trace, tick):
    live_contacts, fixed_contacts = [[step for step in contacts(row, 1) if step > tick] for row in (live, fixed)]
    end = min(tick + EARLY_MOTOR_WINDOW, len(live['actions']), len(fixed['actions']),
              *(steps[0] for steps in (live_contacts, fixed_contacts) if steps))
    delta = abs(np.asarray(live['actions'][tick:end]) - np.asarray(fixed['actions'][tick:end]))
    changed = np.flatnonzero(np.any(delta > RATE_THRESHOLD, axis=1)) if len(delta) else np.empty(0, int)
    flight_tick = min(end - 1, tick + 8, len(nominal_trace.positions) - 1)
    drift_x = fixed_trace.positions[flight_tick][0] - nominal_trace.positions[flight_tick][0]
    return dict(window_stop_tick=end, early_comparable_ticks=len(delta), changed_ticks=len(changed),
                first_rate_change_after_kick=int(changed[0]) if len(changed) else None,
                early_abs_rate_max_by_joint=delta.max(axis=0).tolist() if len(delta) else [],
                flight_sample_tick=flight_tick, lateral_offset_from_nominal_m=drift_x,
                near_contact_ticks={name: [step for step in contacts(row, 1) if step > tick]
                                    for name, row in [('live', live), ('fixed', fixed), ('nominal', nominal)]},
                near_return_ticks={name: [step for step in returns(row, 1) if step > tick]
                                   for name, row in [('live', live), ('fixed', fixed), ('nominal', nominal)]})


def _test_case(station, pupil, verification, start, index):
    nominal, nominal_trace = _live(station, pupil, start)
    repeat, _ = _fixed(station, start, nominal)
    keys = ('outcome', 'reward', 'actions', 'rival_actions', 'state_sha256')
    if (any(nominal[key] != repeat[key] for key in keys) or
            digest_json(nominal['events']) != digest_json(repeat['events'])):
        raise ValueError('the no-kick nominal near tape changed its live-opponent control')
    _verify(verification, nominal)
    base = dict(start, nominal=nominal, nominal_live_rival_repeat_exact=True, verified_tapes=1, disturbances={})
    for name, tick in [('opening', OPENING_KICK_TICK), ('reply', _reply_tick(nominal))]:
        if tick is None:
            continue
        kick = dict(tick=tick, velocity_delta_x=start['velocity_delta_x'])
        if index % 2:
            fixed, fixed_trace = _fixed(station, start, nominal, kick)
            live, live_trace = _live(station, pupil, start, kick)
        else:
            live, live_trace = _live(station, pupil, start, kick)
            fixed, fixed_trace = _fixed(station, start, nominal, kick)
        if (live_trace.states[:tick+1] != nominal_trace.states[:tick+1] or
                fixed_trace.states[:tick+1] != nominal_trace.states[:tick+1] or live['kick'] != fixed['kick'] or
                live['actions'][:tick+1] != fixed['actions'][:tick+1]):
            raise ValueError('a pupil got a different pre-kick prefix or immediately observed a velocity change')
        _verify(verification, live, kick)
        _verify(verification, fixed, kick)
        base['verified_tapes'] += 2
        metrics = _affected_metrics(live, fixed, nominal, fixed_trace, nominal_trace, tick)
        base['disturbances'][name] = dict(live=live, fixed=fixed, metrics=metrics)
    return base


def _summary(cases, phase):
    pairs = [case['disturbances'][phase] for case in cases if phase in case['disturbances']]
    legal = {key: [len(pair['metrics']['near_return_ticks'][key]) for pair in pairs]
             for key in ('nominal', 'live', 'fixed')}
    present = {key: np.asarray(value) > 0 for key, value in legal.items()}
    motions = [pair['metrics'] for pair in pairs]
    early = [item['first_rate_change_after_kick'] for item in motions if item['first_rate_change_after_kick'] is not None]
    return dict(pairs=len(pairs), at_least_one_near_return={key: int(value.sum()) for key, value in present.items()},
                total_near_returns={key: sum(value) for key, value in legal.items()},
                near_returns_by_pair=legal,
                live_only=int((present['live'] & ~present['fixed']).sum()),
                fixed_only=int((present['fixed'] & ~present['live']).sum()),
                both=int((present['fixed'] & present['live']).sum()),
                early_rate_changed_pairs=len(early), median_first_rate_change_ticks=float(np.median(early)) if early else None,
                changed_early_rate_ticks=sum(item['changed_ticks'] for item in motions),
                comparable_early_rate_ticks=sum(item['early_comparable_ticks'] for item in motions),
                median_abs_flight_x_offset_m=float(np.median([abs(item['lateral_offset_from_nominal_m'])
                                                             for item in motions])) if motions else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    args = parser.parse_args()
    starts = _design()
    source = _read(bank.DATA_RECEIPT)
    if (source['demos'] != 48 or source['samples_sha256'] != _hash(bank.CACHE) or
            tuple(source['dimensions']) != bank.DIMS):
        raise ValueError('this probe expects the frozen 48-release broad demonstration bank')
    metadata = dict(schema='tt24-rally-flight-disturbance-1', fit_sha256=_hash(bank.CACHE),
                    far_profile_sha256=_hash(COACHED), starts=starts, cohorts=COHORTS, assignment_seed=ASSIGNMENT_SEED,
                    shift_values_m_per_s=KICKS, opening_before_control_tick=OPENING_KICK_TICK,
                    reply_delay_ticks=REPLY_SEPARATION_TICKS, settings=bank.BROAD_SETTINGS, dimensions=bank.DIMS,
                    zero_suffix='hold the final position target if the fixed path extends past the nominal tape',
                    early_motor_window=EARLY_MOTOR_WINDOW, early_rate_threshold=RATE_THRESHOLD)
    if args.output.exists():
        receipt = _read(args.output)
        if digest_json(receipt['metadata']) != digest_json(metadata):
            raise ValueError('saved probe has different fit or planned treatment')
    else:
        _check_fresh(starts)
        receipt = dict(metadata=metadata, cases=[])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(args.output, receipt)
    began = perf_counter()
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    bank.prepare_held(bank.CACHE, OPENING, DATA, profile)
    station, models = bank.LIVE
    verification = SequenceArena(**dict(CONFIG, width=bank.DIMS[0], height=bank.DIMS[1]))
    try:
        for index, start in enumerate(starts[len(receipt['cases']):], len(receipt['cases'])):
            case = _test_case(station, models['broad'], verification, start, index)
            receipt['cases'].append(case)
            _write(args.output, receipt)
            print(index + 1, start['seed'], len(returns(case['nominal'], 1)),
                  {name: [len(returns(pair[key], 1)) for key in ('live', 'fixed')]
                   for name, pair in case['disturbances'].items()}, flush=True)
    finally:
        station.arena.close()
        verification.close()
    receipt['summary'] = {phase: _summary(receipt['cases'], phase) for phase in ('opening', 'reply')}
    receipt['verified_tapes'] = sum(case['verified_tapes'] for case in receipt['cases'])
    receipt['wall_seconds_last_invocation'] = perf_counter() - began
    _write(args.output, receipt)
    print(receipt['summary'], 'verified', receipt['verified_tapes'], flush=True)


if __name__ == '__main__':
    main()

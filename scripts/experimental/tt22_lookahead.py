"""Look past a physical far hit through the RGB pupil's next motor pulse."""

import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import HOME as FAR_HOME, TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt21_coach_oracle import PHASE
from .tt22_match import step_recorded
from .tt22_motion_object import MotionObject, OPENING
from .tt22_opponent import Incoming, install_rally_opponent
from .tt22_rally_trial import OUTPUT, alternating
from .tt22_servo import RecurrentServo
from .tt22_servo_trial import MACRO, PARTNER, ServoArena, metrics, verify


SOURCE = OUTPUT / 'servo-held.json.gz'
ASSESS = 922933013
FRAME = np.array([0., 0., TABLE_HEIGHT])
OPTIONS = 12


def _common_command(arena, instruction):
    arena._rival_attempt = True
    arena._aim = lambda *args: np.asarray(instruction)


def camera_error(source):
    """Measure a deterministic readout bias on already frozen native runs."""
    arena = ServoArena()
    accepted, observations = [], 0
    try:
        for pair in source['pairs']:
            row = pair['follow']
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            for point, near, far in zip(row['optical'], row['actions'], row['rival_actions'], strict=True):
                if point is not None and -.88 < arena.ball[1] < .65:
                    observations += 1
                    error = np.asarray(point) - (arena.ball - FRAME)
                    if np.linalg.norm(error) < .11:
                        accepted.append(error)
                _common_command(arena, far)
                arena.step(near, render=False)
            if digest_json(arena.events) != digest_json(row['events']) or arena.point_result() != row['outcome']:
                raise ValueError('the archived pupil point changed during physical replay')
    finally:
        arena.close()
    errors = np.asarray(accepted)
    return dict(frames=observations, accepted=len(accepted), bias=np.median(errors, axis=0).tolist(),
                q75=np.quantile(abs(errors - np.median(errors, axis=0)), .75, axis=0).tolist())


def shortlist(rows, limit=OPTIONS):
    """Keep central candidates and several other legal shapes from the same batch."""
    rows = [row for row in rows if row['reachable'] is not None]
    selected = rows[:5]
    locations = {tuple(np.round(np.asarray(row['reachable']) / [.065,.085,.055,.45,.30,.55]).astype(int))
                 for row in selected}
    kept = {row['index'] for row in selected}
    for row in rows[5:]:
        shape = tuple(np.round(np.asarray(row['reachable']) / [.065,.085,.055,.45,.30,.55]).astype(int))
        if shape in locations:
            continue
        locations.add(shape)
        kept.add(row['index'])
        selected.append(row)
        if len(selected) == limit:
            break
    selected.extend(row for row in rows if row['index'] not in kept)
    return selected[:limit]


class PhysicalLookahead:
    """Copy the current pupil into scratch physics. Live pupil inputs stay optical."""

    def __init__(self, rival, camera, deeper=OPTIONS):
        self.rival = rival
        self.bias = np.asarray(camera['bias'])
        self.jitter = np.minimum(np.asarray(camera['q75']), [.012, .016, .012])
        self.original = rival.confirm
        self.deeper = deeper
        self.check = ServoArena()
        self.flight = Incoming(self.check)
        self.snapshot = None
        rival.confirm = self.confirm

    def observe(self, pupil):
        arena = self.rival.arena
        if arena._last_hitter == 1 and arena.velocity[1] > .22 and arena.ball[1] > -.57:
            self.snapshot = deepcopy(pupil)

    def future(self, tape, noise):
        arena, check = self.rival.arena, self.check
        mujoco.mj_copyData(check.data, check.model, arena.data)
        for key in PHASE:
            if hasattr(arena, key):
                setattr(check, key, deepcopy(getattr(arena, key)))
        check.events = []
        check._rival_attempt = True
        check._choose_rival_response = lambda: None
        begin = arena.steps
        until = min(len(tape) + 24, arena.max_steps - begin)
        pupil = deepcopy(self.snapshot)
        far_contacts = check.rival_contacts
        pulse_start = len(pupil.pulses)

        def drive(*args):
            elapsed = check.steps - begin
            if check.rival_contacts == far_contacts and elapsed < len(tape):
                return tape[elapsed]
            return check._command_to(-1, FAR_HOME, .97)

        check._aim = drive
        far_reply, pupil_reply = None, None
        for _ in range(until):
            point = check.ball - FRAME + self.bias + noise
            command = pupil.command(check.steps, point)
            _, _, done, capped, info = check.step(command, render=False)
            for event in info['events']:
                if event['kind'] == 'legal_return' and event['side'] == -1 and far_reply is None:
                    far_reply = event
                elif event['kind'] == 'legal_return' and event['side'] == 1 and far_reply is not None:
                    pupil_reply = event
            if pupil_reply is not None or done or capped:
                break
        options = [] if pupil_reply is None else self.flight.windows(36)
        return dict(far=far_reply, near=pupil_reply, future_far_windows=len(options),
                    events=[e for e in check.events if e['kind'] in ('contact', 'legal_return', 'point')],
                    reason=check.reason, pupil_pulses=pupil.pulses[pulse_start:])

    def confirm(self, ranked, tapes, record):
        if self.snapshot is None:
            return self.original(ranked, tapes, record)
        attempts = []
        options = []
        def evaluate(candidate):
            tape = tapes[candidate['index']]
            outcome = self.future(tape, np.zeros(3))
            attempts.append(dict(index=candidate['index'], far=outcome['far'] is not None,
                                 near=outcome['near'] is not None, next_far=outcome['future_far_windows']))
            if outcome['far'] is not None and outcome['near'] is not None:
                y = outcome['near']['location'][1]
                x = outcome['near']['location'][0]
                merit = (.16 * min(3, outcome['future_far_windows']) - abs(y-.54)
                         - .62*abs(x) - .12*candidate['cost'])
                options.append((merit, candidate, outcome))
        for candidate in shortlist(ranked):
            evaluate(candidate)
        if not options and self.deeper > OPTIONS and self.rival.arena.contacts >= 2:
            tested = {item['index'] for item in attempts}
            for candidate in shortlist(ranked, self.deeper):
                if candidate['index'] not in tested:
                    evaluate(candidate)
        record['lookahead_trials'] = attempts
        record['pupil_confirmed'] = len(options)
        if not options:
            return self.original(ranked, tapes, record)
        ranked_results = sorted(options, key=lambda row: row[0], reverse=True)
        refined = []
        for merit, candidate, outcome in ranked_results[:3]:
            noises = (self.jitter*np.array([.45, -.8, .65]), self.jitter*np.array([-.45, .8, -.65]))
            alternate = [self.future(tapes[candidate['index']], noise) for noise in noises]
            stable = sum(result['far'] is not None and result['near'] is not None for result in alternate)
            refined.append((merit + .35*stable, candidate, outcome, stable))
        _, picked, outcome, stable = max(refined, key=lambda row: row[0])
        record['confirmed'] = True
        record['exact_trials'] = len(attempts)+2*len(refined)
        return dict(picked, predicted_next=outcome['near'], predicted_rival=outcome['far'],
                    predicted_after_windows=outcome['future_far_windows'], noise_successes=stable,
                    shadow=outcome, physical_lookahead=True)

    def close(self):
        self.check.close()


class Bench:
    def __init__(self, config, teacher):
        self.arena = ServoArena()
        self.rival = install_rally_opponent(self.arena, target_profile=PARTNER)
        self.teacher = PhysicalLookahead(self.rival, config['camera'], config.get('deeper', OPTIONS)) if teacher else None
        self.config = config

    def point(self, seed, lane):
        arena, rival = self.arena, self.rival
        arena.lane = lane
        rival.target_profile = (float(.066*lane), *PARTNER[1:])
        arena.reset(seed=seed, server=-1, render=False)
        if self.teacher is not None:
            self.teacher.snapshot = None
        view = MotionObject(arena.lens_intrinsics())
        pupil = RecurrentServo(self.config['macro'], self.config['weights'],
                               lead=self.config['settings'][0], amplitude=self.config['settings'][1])
        commands, far_tape, optical = [], [], []
        digest = sha256()
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            for stream, frame in frames.items():
                digest.update(stream.encode())
                digest.update(frame)
            position = view.update(frames, tick)
            if tick == OPENING-1 and view.prefix_positions:
                for at, value in view.prefix_positions[:-1]:
                    pupil.history[at] = at, value
            if self.teacher is not None:
                self.teacher.observe(pupil)
            command = pupil.command(tick, position)
            optical.append(None if position is None else position.tolist())
            commands.append(command.tolist())
            reward, done, cap, far = step_recorded(arena, command)
            far_tape.append(far)
            if done or cap:
                break
        return dict(seed=seed, lane=lane, longest=alternating(arena.events), actions=commands, rival_actions=far_tape,
                    optical=optical, pulses=pupil.pulses, initialized=view.seed_score is not None,
                    rgb=digest.hexdigest(), reward=reward, outcome=arena.point_result(),
                    events=arena.events.copy(), decisions=rival.decisions.copy())

    def close(self):
        if self.teacher is not None:
            self.teacher.close()
        self.arena.close()


def run(path, count, *, development=True):
    evidence = _read(SOURCE)
    macro = _read(MACRO)['parameters']
    camera = camera_error(evidence)
    configuration = dict(macro=macro, settings=evidence['settings'], weights=evidence['coefficients']['follow'],
                         camera=camera)
    seeds = np.random.default_rng(ASSESS).integers(2**31, size=count).tolist()
    lanes = (-.10, .08, -.09, .12)
    held = [(seed, lanes[index % len(lanes)]) for index,seed in enumerate(seeds)]
    prior = {row[0] for row in evidence['openings']}
    if prior.intersection(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('held release seeds must be fresh')
    dev = tuple(evidence['openings'][2])
    data = dict(schema='tt22-two-physical-strokes-1', configuration=configuration,
                sources={SOURCE.name: _hash(SOURCE), MACRO.name: _hash(MACRO)}, held_generator=ASSESS,
                development=[*dev] if development else None, held=held,
                rows={name: {'development':[], 'held':[]} for name in ('fixed','lookahead')})
    stations = {mode:Bench(configuration, mode=='lookahead') for mode in data['rows']}
    began = perf_counter()
    path.parent.mkdir(exist_ok=True, parents=True)
    try:
        stages = [('development', [dev]), ('held', held)] if development else [('held',held)]
        for stage, starts in stages:
            for index, (seed, lane) in enumerate(starts):
                names = tuple(stations)
                for mode in names if index % 2 == 0 else names[::-1]:
                    row = stations[mode].point(seed,lane)
                    data['rows'][mode][stage].append(row)
                    passes = sum(bool(record.get('physical_lookahead')) for record in row['decisions'])
                    print(stage, mode, seed, row['longest'], passes, row['outcome']['reason'], flush=True)
                    _write(path, data)
    finally:
        for station in stations.values():
            station.close()
    data['replayed'] = verify([row for variant in data['rows'].values() for rows in variant.values() for row in rows])
    data['summary'] = {mode: metrics(group['held']) for mode,group in data['rows'].items()}
    data['wall_seconds'] = perf_counter()-began
    _write(path, data)
    return data


def deeper_diagnostic(source, path):
    primary = _read(source)
    stalls = [row for row in primary['rows']['lookahead']['held'] if row['longest'] >= 4]
    config = dict(primary['configuration'], deeper=48)
    bench = Bench(config, True)
    data = dict(schema='tt22-two-stroke-coverage-diagnostic-1', source=source.name, sha256=_hash(source),
                configuration=config, previously_seen=[row['seed'] for row in stalls], rows=[])
    try:
        for before in stalls:
            row = bench.point(before['seed'], before['lane'])
            data['rows'].append(row)
            print('deeper diagnostic', before['seed'], before['longest'], row['longest'], flush=True)
            _write(path, data)
    finally:
        bench.close()
    data['replayed'] = verify(data['rows'])
    _write(path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/'tt22_lookahead.json.gz')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--skip-development', action='store_true')
    parser.add_argument('--deeper-diagnostic', type=Path, metavar='PRIMARY_RECEIPT')
    args = parser.parse_args()
    if args.deeper_diagnostic:
        deeper_diagnostic(args.deeper_diagnostic, args.output)
    else:
        print(run(args.output, args.count, development=not args.skip_development)['summary'])

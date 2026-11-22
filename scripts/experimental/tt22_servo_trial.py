"""Score matched direct-servo rallies using anonymous stereo and physical paddles."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt22_match import step_recorded
from .tt22_motion_object import MotionObject, OPENING
from .tt22_opponent import RallyArena, install_rally_opponent
from .tt22_rally_trial import CONFIG, GentleDraws, OUTPUT, alternating
from .tt22_servo import RecurrentServo


MACRO = OUTPUT/'openloop-practice.json'
SOURCE = ROOT/'artifacts/tt3/discovered-numeric-source.json'
DURATION_SOURCE = OUTPUT/'duration-pong-player.json'
SOURCE_CERTIFICATE = ROOT/'artifacts/tt3/discovered-source-refinement.json.gz'
PRACTICE_SEED = 1370170519
HELD_SEED = 481507604
MODES = ('follow', 'ignore', 'oppose')
OPTIONS = ((6.5, .84), (8., .84), (9.5, .84), (6.5, 1.06), (8., 1.06), (9.5, 1.06))
LOFT_OPTIONS = ((7., .95), (8.5, .95), (7., 1.12), (8.5, 1.12))
PRACTICE = 'servo-practice.json.gz'
HELD = 'servo-held.json.gz'
SPREAD = .18
PACE = 1.
PARTNER = (0., -.835, .17, 1.95, 0., 1.2)


class LaneDraws(GentleDraws):
    def __init__(self, rng, lane):
        super().__init__(rng, SPREAD, PACE)
        self.lane = lane

    def uniform(self, low, high):
        value = super().uniform(low, high)
        return value + high*self.lane if np.ndim(value) == 0 and (low, high) in ((-.30, .30), (-.28, .28)) else value


class ServoArena(RallyArena):
    """A stock pupil, a physically steerable practice partner, and a narrow opening."""

    def __init__(self):
        self.lane = 0.
        super().__init__(**CONFIG)

    def _opening(self):
        rng = self._rng
        if self.server == -1:
            self._rng = LaneDraws(rng, self.lane)
        try:
            super()._opening()
        finally:
            self._rng = rng


def scheduled(seed, lanes):
    starts = np.random.default_rng(seed).integers(2**31, size=len(lanes)).tolist()
    return list(zip(starts, lanes, strict=True))


def coefficients(loft=False):
    source = _read(DURATION_SOURCE if loft else SOURCE)
    if not loft:
        qualification = _read(SOURCE_CERTIFICATE)
        if source != qualification['portable_memory'] or _hash(SOURCE) != qualification['numeric_state_sha256']:
            raise ValueError('the source score-selected state changed')
    best = int(np.argmax(np.divide(source['totals'], source['counts'])))
    selected = np.asarray(source['coefficients'][best][1:])
    return dict(follow=selected.tolist(), ignore=[0., 0.], oppose=(-selected).tolist())


class Station:
    def __init__(self, macro, weights, *, loft=False):
        self.arena = ServoArena()
        self.partner = install_rally_opponent(self.arena, target_profile=PARTNER)
        self.macro, self.weights = macro, weights
        self.loft = loft

    def point(self, seed, lane, mode, settings):
        arena, partner = self.arena, self.partner
        arena.lane = lane
        partner.target_profile = (float(.066*lane), *PARTNER[1:])
        arena.reset(seed=seed, server=-1, render=False)
        view = MotionObject(arena.lens_intrinsics())
        receiver = RecurrentServo(self.macro, self.weights[mode], lead=settings[0], amplitude=settings[1],
                                  loft=.135 if self.loft else 0., height_gain=.45 if self.loft else .88,
                                  continuous_phase=self.loft)
        timeline, commands, opponent = [], [], []
        digest = sha256()
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            for stream, frame in frames.items():
                digest.update(stream.encode())
                digest.update(frame)
            position = view.update(frames, tick)
            # Once available, include the anonymously reconstructed prefix in
            # the local motion history. No commands are revised retroactively.
            if tick == OPENING-1 and view.prefix_positions:
                for sample, value in view.prefix_positions[:-1]:
                    receiver.history[sample] = sample, value
            command = receiver.command(tick, position)
            timeline.append(None if position is None else position.tolist())
            commands.append(command.tolist())
            reward, done, cap, far = step_recorded(arena, command)
            opponent.append(far)
            if done or cap:
                break
        return dict(seed=seed, lane=lane, mode=mode, settings=list(settings), longest=alternating(arena.events),
                    actions=commands, rival_actions=opponent, optical=timeline, pulses=receiver.pulses,
                    initialized=view.seed_score is not None, rgb=digest.hexdigest(), reward=reward,
                    outcome=arena.point_result(), events=arena.events.copy())

    def close(self):
        self.arena.close()


def verify(rows):
    arena = ServoArena()
    install_rally_opponent(arena, target_profile=PARTNER)
    try:
        for row in rows:
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            aim, direct = arena._aim, arena._command_to
            try:
                for command, far in zip(row['actions'], row['rival_actions'], strict=True):
                    far = np.asarray(far)
                    arena._aim = arena._command_to = lambda *args, rates=far: rates
                    _, reward, _, _, _ = arena.step(command, render=False)
            finally:
                arena._aim, arena._command_to = aim, direct
            if (row['outcome'] != arena.point_result() or row['reward'] != reward
                    or json.dumps(row['events']) != json.dumps(arena.events)):
                raise ValueError(f"direct servo physics replay changed at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def metrics(rows):
    return dict(starts=len(rows), visible=sum(row['initialized'] for row in rows),
                first=sum(row['outcome']['legal_returns'] > 0 for row in rows),
                continuing=sum(row['outcome']['legal_returns'] > 1 for row in rows),
                learner_returns=sum(row['outcome']['legal_returns'] for row in rows),
                rival_returns=sum(row['outcome']['rival_returns'] for row in rows),
                contacts=sum(row['outcome']['contacts'] for row in rows),
                wins=sum(row['reward'] > 0 for row in rows),
                lengths=[row['longest'] for row in rows],
                at_least_ten=sum(row['longest'] >= 10 for row in rows),
                fired=sum(len(row['pulses']) for row in rows),
                refired=sum(len(row['pulses']) > 1 for row in rows))


def run_path(name, loft):
    return name.replace('servo-', 'servo-loft-') if loft else name


def train(folder, *, loft):
    original = _read(MACRO)
    modes = coefficients(loft)
    practice_seed = PRACTICE_SEED+1 if loft else PRACTICE_SEED
    starts = scheduled(practice_seed, (-.115, .115) if loft else (-.24, .24))
    station = Station(original['parameters'], modes, loft=loft)
    path = folder/run_path(PRACTICE, loft)
    data = dict(schema='tt22-servo-practice-1', prior_stroke=_hash(MACRO),
                source_sha256=_hash(DURATION_SOURCE if loft else SOURCE), loft=loft,
                practice_seed=practice_seed, openings=starts, prior_motor=original['parameters'],
                coefficients=modes, variants=[])
    try:
        for i, settings in enumerate(LOFT_OPTIONS if loft else OPTIONS):
            rows = [station.point(seed, lane, 'ignore' if j == 0 else 'follow', settings)
                    for j, (seed, lane) in enumerate(starts)]
            data['variants'].append(dict(settings=list(settings), rows=rows, summary=metrics(rows)))
            print('practice', i, settings, metrics(rows), flush=True)
            _write(path, data)
    finally:
        station.close()
    candidates = data['variants']
    selection = max(candidates, key=lambda v: (v['summary']['learner_returns'], v['summary']['continuing'],
                                               v['summary']['first'], v['summary']['wins'],
                                               -abs(v['settings'][0]-8.), -abs(v['settings'][1]-1.)))
    data['selected'] = selection['settings']
    data['verified'] = verify([row for variant in candidates for row in variant['rows']])
    _write(path, data)
    print('selected', data['selected'], selection['summary'], flush=True)
    return data


def held(folder, practice, *, loft):
    source = DURATION_SOURCE if loft else SOURCE
    if practice['prior_stroke'] != _hash(MACRO) or practice['source_sha256'] != _hash(source):
        raise ValueError('the carried motor or visual state changed')
    held_seed = HELD_SEED+1 if loft else HELD_SEED
    starts = scheduled(held_seed, (-.15, .12, -.19, .16, -.14, .18) if loft else (-.28, .21, -.16, .31, -.32, .17))
    if {seed for seed, _ in starts} & {seed for seed, _ in practice['openings']}:
        raise ValueError('held points overlap servo practice')
    bench = Station(practice['prior_motor'], practice['coefficients'], loft=loft)
    path = folder/run_path(HELD, loft)
    data = dict(schema='tt22-servo-held-1', seed=held_seed, openings=starts,
                practice_sha256=_hash(folder/run_path(PRACTICE, loft)), loft=loft,
                settings=practice['selected'], pairs=[], coefficients=practice['coefficients'])
    try:
        for i, (seed, lane) in enumerate(starts):
            modes = MODES[i%len(MODES):] + MODES[:i%len(MODES)]
            pair = {mode: bench.point(seed, lane, mode, practice['selected']) for mode in modes}
            data['pairs'].append(pair)
            data['summary'] = {mode: metrics([pair[mode] for pair in data['pairs']]) for mode in MODES}
            _write(path, data)
            print('held', seed, {mode: (row['longest'], len(row['pulses'])) for mode, row in pair.items()}, flush=True)
    finally:
        bench.close()
    data['verified'] = verify([row for pair in data['pairs'] for row in pair.values()])
    _write(path, data)
    print(data['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('both', 'practice', 'held', 'verify'), default='both')
    parser.add_argument('--loft', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.stage == 'verify':
        practice, check = _read(args.folder/run_path(PRACTICE, args.loft)), _read(args.folder/run_path(HELD, args.loft))
        print('replayed practice', verify([row for group in practice['variants'] for row in group['rows']]))
        print('replayed held', verify([row for pair in check['pairs'] for row in pair.values()]))
    else:
        practice = train(args.folder, loft=args.loft) if args.stage in ('both', 'practice') else _read(args.folder/run_path(PRACTICE, args.loft))
        if args.stage in ('both', 'held'):
            held(args.folder, practice, loft=args.loft)

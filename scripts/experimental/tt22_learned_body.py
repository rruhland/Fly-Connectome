"""Calibrate a hidden port installation, then run the same optical rally player."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_motor_carry import COMMANDS, IDENTITY, READ_TICKS, WIDTH, Wiring, descriptor, identify
from .tt20_motor_carry_trial import installation
from .tt22_opponent_learn import JitteredOpening, PACE, SPREAD, _station
from .tt22_opponent_trial import FarTape
from .tt22_point_feedback import OutcomePlayer
from .tt22_rally_trial import CONFIG, OUTPUT, alternating


SOURCE = OUTPUT/'paired-course.json.gz'
INSTALLATION = 1728163107
PROBES = 588240953
EYES = ('left', 'wrist')
SOURCE_EYE = 'wrist'
MODES = ('learned', 'unadapted')
PROBE_SETTINGS = dict(CONFIG, view_mode='wrist+fixed')


def optical_visit(arena, wiring, seed, port, sign):
    """One physical pulse produces two pixel streams on the same clock."""
    arena.reset(seed=seed, server=-1, render=False)
    first = arena.render_cameras()
    images = {eye: [first[eye]] for eye in EYES}
    public = []
    for tick, magnitude in enumerate(COMMANDS, 1):
        action = np.zeros(WIDTH+1)
        action[port] = magnitude*sign
        _, _, done, capped, _ = arena.step(wiring.physical(action), render=False)
        if done or capped:
            raise ValueError('a probe ended early')
        public.append(action.tolist())
        if tick in READ_TICKS:
            frames = arena.render_cameras()
            for eye in EYES:
                images[eye].append(frames[eye])
    tape = dict(seed=seed, public_port=port, sign=sign, commands=public,
                rgb_sha256={eye: [sha256(frame).hexdigest() for frame in views]
                            for eye, views in images.items()})
    return {eye: descriptor(views) for eye, views in images.items()}, tape


def calibration(path):
    wiring = installation(INSTALLATION)
    seeds = iter(np.random.default_rng(PROBES).integers(2**31, size=3*WIDTH).tolist())
    known = JitteredOpening(**PROBE_SETTINGS)
    target = JitteredOpening(**PROBE_SETTINGS)
    observed, atlas = {eye: [] for eye in EYES}, {eye: [] for eye in EYES}
    target_tapes, model_tapes = [], []
    try:
        for port in range(WIDTH):
            view, row = optical_visit(target, wiring, next(seeds), port, 1)
            for eye in EYES:
                observed[eye].append(view[eye])
            target_tapes.append(row)
        for axis in range(WIDTH):
            visits = []
            for polarity in (-1, 1):
                view, row = optical_visit(known, IDENTITY, next(seeds), axis, polarity)
                visits.append(view)
                model_tapes.append(row)
            for eye in EYES:
                atlas[eye].append([view[eye] for view in visits])
    finally:
        target.close()
        known.close()
    measurements = {}
    for eye in EYES:
        inference, candidate_costs = identify(observed[eye], atlas[eye])
        measurements[eye] = dict(inferred=inference.state(), fits=candidate_costs)
        print('optics', eye, inference.state(), candidate_costs[-1]['next_assignment_margin'], flush=True)
    result = dict(schema='tt22-shared-body-probes-1', arena=PROBE_SETTINGS, installation_seed=INSTALLATION,
                  probe_seed=PROBES, evaluator_installation=wiring.state(), deployment_eye=SOURCE_EYE,
                  target_frames=WIDTH*len(COMMANDS), target_resets=WIDTH, nominal_frames=2*WIDTH*len(COMMANDS),
                  target_tapes=target_tapes, independent_nominal_tapes=model_tapes, measurements=measurements)
    _write(path, result)


class ActuatorBridge:
    """The policy requests nominal rates. Only this layer drives public ports."""

    def __init__(self, arena, evaluator_wiring, correction):
        self.wiring = evaluator_wiring
        self.correction = correction
        self.step = arena.step
        self.public = []
        self.realized = []
        arena.step = self.submit

    def submit(self, request, *, render=True):
        public = np.asarray(request) if self.correction is None else self.correction.inverse(request)
        actual = self.wiring.physical(public)
        self.public.append(public.tolist())
        self.realized.append(actual.tolist())
        return self.step(actual, render=render)

    def clear(self):
        self.public.clear()
        self.realized.clear()


def native_station(source, profile, wiring, correction):
    station = _station(profile)
    law = station.player.law
    station.player.close()
    # The original published frozen branch evaluates the same ranker at gain 0.
    station.player = OutcomePlayer(law, np.asarray(source['weights']), 0.)
    return station, FarTape(station.arena), ActuatorBridge(station.arena, wiring, correction)


def recode(requests, issued, realized):
    near = np.asarray(requests)
    public = np.asarray(issued)
    physical = np.asarray(realized)
    if not (near.shape == public.shape == physical.shape):
        raise ValueError('controller and actuator tapes have different dimensions')
    return dict(public_differs=sum(map(int, np.any(near != public, axis=1))),
                physical_differs=sum(map(int, np.any(near != physical, axis=1))),
                physical_differs_from_public=sum(map(int, np.any(public != physical, axis=1))),
                nonzero_steps=sum(map(int, np.any(near != 0., axis=1))))


def summary(rows):
    lengths = [alternating(row['events']) for row in rows]
    return dict(points=len(rows), lengths=lengths, longest=max(lengths, default=0),
                at_least_10=sum(n >= 10 for n in lengths), at_least_20=sum(n >= 20 for n in lengths),
                own_returns=sum(row['outcome']['legal_returns'] for row in rows),
                rival_returns=sum(row['outcome']['rival_returns'] for row in rows),
                wins=sum(row['reward'] > 0 for row in rows),
                control_frames=sum(len(row['actions']) for row in rows),
                changed_public_frames=sum(row['actuation']['public_differs'] for row in rows),
                changed_physical_frames=sum(row['actuation']['physical_differs'] for row in rows))


def replay_point(arena, row, wiring, correction, *, rgb=False, observe=None):
    arena.reset(seed=row['seed'], server=-1, render=False)
    digest = sha256()
    for tick, (request, public, resulting, far) in enumerate(zip(
            row['actions'], row['issued_public'], row['realized_near'], row['rival_actions'], strict=True)):
        translated = request if correction is None else correction.inverse(request).tolist()
        actual = wiring.physical(public)
        if translated != public or actual.tolist() != resulting:
            raise ValueError('the public request does not realize the recorded motor rates')
        if rgb:
            for eye, image in arena.render_cameras().items():
                digest.update(eye.encode())
                digest.update(image)
        if observe is not None:
            observe(arena)
        arena._rival_attempt = True
        arena._aim = lambda *args: np.asarray(far)
        _, reward, done, capped, _ = arena.step(actual, render=False)
        if bool(done or capped) != (tick+1 == len(row['actions'])):
            raise ValueError('the physical port replay changed point duration')
    if (arena.point_result() != row['outcome'] or reward != row['reward']
            or digest_json(arena.events) != digest_json(row['events'])
            or (rgb and digest.hexdigest() != row['rgb'])):
        raise ValueError(f'the physical port replay changed point {row["seed"]}')
    if observe is not None:
        observe(arena)


def verify(rows, wiring, correction, rgb=False):
    arena = JitteredOpening(**CONFIG)
    try:
        for row in rows:
            replay_point(arena, row, wiring, correction, rgb=rgb)
        return len(rows)
    finally:
        arena.close()


def score(path, probes):
    source = _read(SOURCE)
    expected = source['rows']['frozen']
    seeds = np.random.default_rng(source['seed']).integers(2**31, size=12).tolist()
    if seeds != [row['seed'] for row in expected] or set(seeds).intersection(source['training_seeds']):
        raise ValueError('the archived comparison is not the complete independent course')
    acquisition = _read(probes)
    wiring = Wiring.restore(acquisition['evaluator_installation'])
    if wiring != installation(INSTALLATION):
        raise ValueError('the body was reinstalled after calibration')
    correction = Wiring.restore(acquisition['measurements'][SOURCE_EYE]['inferred'])
    basis = np.eye(WIDTH+1)
    realized = np.stack([wiring.physical(correction.inverse(row)) for row in basis])
    data = dict(schema='tt22-optical-learned-body-1', cohort_sha256=_hash(SOURCE), cohort_seed=source['seed'],
                seeds=seeds, profile=source['profile'], arena=CONFIG, opening=dict(spread=SPREAD, pace=PACE),
                probe_sha256=_hash(probes), body_practice_frames=acquisition['target_frames'],
                calibration_eye=SOURCE_EYE, play_eyes=['left', 'right'],
                exact_basis=bool(np.array_equal(realized, basis)), realized_basis=realized.tolist(),
                evaluator_installation=wiring.state(), installed_correction=correction.state(),
                branches={name: [] for name in MODES}, archive_matches=[])
    stations = {name: native_station(source, source['profile'], wiring, correction if name == 'learned' else None)
                for name in MODES}
    start = perf_counter()
    try:
        for i, seed in enumerate(seeds):
            for name in MODES if i % 2 == 0 else MODES[::-1]:
                station, tape, bridge = stations[name]
                tape.steps.clear()
                bridge.clear()
                row = station.point(seed)
                row['issued_public'] = bridge.public.copy()
                row['realized_near'] = bridge.realized.copy()
                row['rival_actions'] = tape.tape(len(row['actions']))
                row['actuation'] = recode(row['actions'], row['issued_public'], row['realized_near'])
                row['public_sha256'] = digest_json(row['issued_public'])
                data['branches'][name].append(row)
                if name == 'learned':
                    ref = expected[i]
                    data['archive_matches'].append(dict(seed=seed, optics=row['rgb']==ref['rgb'],
                                                        nominal_actions=row['actions']==ref['actions'],
                                                        physical_actions=row['realized_near']==ref['actions'],
                                                        far_actions=row['rival_actions']==ref['rival_actions'],
                                                        outcome=row['outcome']==ref['outcome'],
                                                        events=digest_json(row['events'])==digest_json(ref['events'])))
                _write(path, data)
                print(name, i+1, seed, row['longest'], row['outcome']['legal_returns'],
                      row['actuation']['public_differs'], row['actuation']['physical_differs'], flush=True)
    finally:
        for station, _, _ in stations.values():
            station.close()
    data['summary'] = {name: summary(rows) for name, rows in data['branches'].items()}
    data['replayed'] = {name: verify(rows, wiring, correction if name == 'learned' else None)
                        for name, rows in data['branches'].items()}
    witness = max(data['branches']['learned'], key=lambda row: row['longest'])
    data['rgb_witness'] = dict(seed=witness['seed'], longest=witness['longest'],
                              replays=verify([witness], wiring, correction, rgb=True))
    data['wall_seconds'] = perf_counter()-start
    _write(path, data)
    print('completed', data['summary'], data['replayed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=('probe', 'score', 'all', 'verify'), default='all')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    probes = args.folder/'tt22_learned_body_probes.json.gz'
    result = args.folder/'tt22_learned_body_held.json.gz'
    if args.phase in ('all', 'probe'):
        calibration(probes)
    if args.phase in ('all', 'score'):
        score(result, probes)
    if args.phase == 'verify':
        data = _read(result)
        wiring = Wiring.restore(data['evaluator_installation'])
        correction = Wiring.restore(data['installed_correction'])
        print({name: verify(rows, wiring, correction if name == 'learned' else None)
               for name, rows in data['branches'].items()})

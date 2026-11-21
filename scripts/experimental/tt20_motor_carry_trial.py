"""Train a signed actuator map with one eye and carry the map into strict games."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_camera_carry import EYES, OUTPUT, SETTINGS, Station, summarize
from .tt20_curriculum import configuration
from .tt20_fixture import TT20Arena
from .tt20_motor_carry import COMMANDS, IDENTITY, WIDTH, Wiring, capture_pulse, identify


INSTALL = 381322771
OPTICAL = 514078231
GAME = 28505413
MOTOR = 'motor-carry'
RANGING = 'adaptive'


def installation(seed=INSTALL):
    rng = np.random.default_rng(seed)
    return Wiring(tuple(map(int, rng.permutation(WIDTH))), tuple(map(int, rng.choice((-1, 1), WIDTH))))


def probe(eye, wiring):
    # The target and independent CAD simulator have disjoint ball starts.
    seed = np.random.default_rng([OPTICAL, EYES.index(eye)])
    optical = iter(seed.integers(2**31, size=3 * WIDTH).tolist())
    target = TT20Arena(**SETTINGS, view_mode=eye)
    nominal = TT20Arena(**SETTINGS, view_mode=eye)
    observed, atlas, records, source = [], [], [], []
    try:
        for port in range(WIDTH):
            response, row = capture_pulse(target, eye, next(optical), port, 1, wiring)
            observed.append(response)
            records.append(row)
        for axis in range(WIDTH):
            options = []
            for sign in (-1, 1):
                response, row = capture_pulse(nominal, eye, next(optical), axis, sign, IDENTITY)
                options.append(response)
                source.append(row)
            atlas.append(options)
        recovered, fits = identify(observed, atlas)
    finally:
        nominal.close()
        target.close()
    result = dict(eye=eye, physical_calibration_frames=WIDTH * len(COMMANDS), calibration_resets=WIDTH,
                  dry_model_frames=2 * WIDTH * len(COMMANDS), target=records, independent_model=source,
                  equal_length_response_reductions=fits, inferred=recovered.state())
    print('optical', eye, recovered.state(), 'margin', fits[-1]['next_assignment_margin'], flush=True)
    return result


class Bridge:
    """Translate learner commands before they reach the remapped public interface."""

    def __init__(self, arena, wiring, correction):
        self.native = arena.step
        self.wiring = wiring
        self.correction = correction
        self.reset()
        arena.step = self.step

    def reset(self):
        self.commands = []

    def step(self, action, *, render=True):
        public = action if self.correction is None else self.correction.inverse(action)
        self.commands.append(np.asarray(public).tolist())
        return self.native(self.wiring.physical(public), render=render)


def scored(eye, wiring, correction, candidate):
    station = Station(eye, ranging=RANGING)
    bridge = Bridge(station.arena, wiring, correction)
    score = ServiceScore(first_server=-1)
    rng = np.random.default_rng(GAME)
    rows = []
    try:
        for _ in range(80):
            seed = int(rng.integers(2**31))
            bridge.reset()
            point = station.point(seed, score.server, candidate)
            point['public_commands'] = bridge.commands
            rows.append(point)
            score.record(point['outcome'])
            if score.finished:
                break
        if not score.finished:
            raise ValueError('the strict 11-point game did not finish')
    finally:
        station.close()
    result = dict(eye=eye, game_seed=GAME, candidate=candidate, score=list(score.score), rows=rows,
                  result=summarize(rows))
    print('scored', eye, 'unadapted' if correction is None else 'inferred', score.score, result['result'], flush=True)
    return result


def replay(result, wiring, correction, *, check_recorded=False):
    arena = TT20Arena(**SETTINGS, view_mode=result['eye'])
    bridge = Bridge(arena, wiring, correction)
    rng = np.random.default_rng(result['game_seed'])
    score = ServiceScore(first_server=-1)
    points = result['rows']
    frames = 0
    try:
        for point in points:
            seed = int(rng.integers(2**31))
            if score.finished or score.server != point['role'] or seed != point['seed']:
                raise ValueError('scoreboard or point schedule differs on replay')
            arena.reset(seed=seed, server=score.server, render=False)
            bridge.reset()
            for action in point['actions']:
                _, reward, _, _, _ = arena.step(action, render=False)
            if check_recorded and bridge.commands != point['public_commands']:
                raise ValueError('the inverse did not reproduce the recorded public actions')
            if (arena.point_result() != point['outcome'] or
                    reward != point['reward'] or
                    json.dumps(arena.events) != json.dumps(point['events'])):
                raise ValueError(f"replay diverged for {result['eye']} seed {point['seed']}")
            frames += len(bridge.commands)
            score.record(arena.point_result())
        if not score.finished or list(score.score) != result['score']:
            raise ValueError('replayed a different completed score')
    finally:
        arena.close()
    return dict(points=len(points), frames=frames)


def equivalence(wiring, first, second):
    """Certify that two public wrappers realize the same six-dimensional map."""
    effective = lambda correction: np.stack([wiring.physical(correction.inverse(axis))
                                              for axis in np.eye(WIDTH + 1)])
    return np.array_equal(effective(first), effective(second))


def evaluate(eye, wiring, calibrations, candidate):
    other_eye = next(key for key in EYES if key != eye)
    inverse = Wiring.restore(calibrations[other_eye]['inferred'])
    same = Wiring.restore(calibrations[eye]['inferred'])
    transfer = scored(eye, wiring, inverse, candidate)
    raw = scored(eye, wiring, None, candidate)
    conditions = dict(other_eye=dict(data=transfer, source=other_eye, acquisition_frames=0,
                                     source_acquisition_frames=calibrations[other_eye]['physical_calibration_frames'],
                                     physical_replay=replay(transfer, wiring, inverse, check_recorded=True)),
                      unadapted=dict(data=raw, source=None, acquisition_frames=0,
                                     physical_replay=replay(raw, wiring, None, check_recorded=True)))
    if equivalence(wiring, same, inverse):
        conditions['same_eye'] = dict(equivalent_to='other_eye', source=eye,
                                      acquisition_frames=calibrations[eye]['physical_calibration_frames'],
                                      physical_replay=replay(transfer, wiring, same))
    else:
        traced = scored(eye, wiring, same, candidate)
        conditions['same_eye'] = dict(data=traced, source=eye,
                                      acquisition_frames=calibrations[eye]['physical_calibration_frames'],
                                      physical_replay=replay(traced, wiring, same, check_recorded=True))
    if inverse == wiring:
        # Each exact corrected command yields the same physical input as the normal interface.
        conditions['normal'] = dict(equivalent_to='other_eye', source=None, acquisition_frames=0,
                                    physical_replay=replay(transfer, IDENTITY, None))
    else:
        traced = scored(eye, IDENTITY, None, candidate)
        conditions['normal'] = dict(data=traced, source=None, acquisition_frames=0,
                                    physical_replay=replay(traced, IDENTITY, None, check_recorded=True))
    return conditions


def collect_probes(path):
    wiring = installation()
    settings = dict(SETTINGS)
    for eye in EYES:
        arena = TT20Arena(**settings, view_mode=eye)
        try:
            if arena.action_size != WIDTH + 1 or tuple(arena.streams) != (eye,):
                raise ValueError('the station changed its public control or view interface')
        finally:
            arena.close()
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        jobs = {eye: pool.submit(probe, eye, wiring) for eye in EYES}
        measurements = {eye: job.result() for eye, job in jobs.items()}
    info = dict(schema='tt20-motor-carry-probes-v1', arena=settings, installation_seed=INSTALL,
                probe_seed=OPTICAL, private_evaluator_map=wiring.state(), measurements=measurements)
    _write(path, info)
    return info


def run(folder, phase, eyes):
    folder.mkdir(parents=True, exist_ok=True)
    calibration_path = folder / f'{MOTOR}-probes.json.gz'
    calibration = collect_probes(calibration_path) if phase in ('probe', 'all') else _read(calibration_path)
    if phase == 'probe':
        return
    wiring = installation()
    if calibration['private_evaluator_map'] != wiring.state() or calibration['arena'] != SETTINGS:
        raise ValueError('optical acquisition and scored evaluation use different stations')
    candidate = int(configuration(OUTPUT)[-1])
    receipt = dict(schema='tt20-motor-carry-games-v1', arena=SETTINGS, seed=GAME,
                   preparation_sha256=_hash(calibration_path), prior_serving_candidate=candidate,
                   private_evaluator_map=wiring.state(), ranging=RANGING, variants={})
    with ProcessPoolExecutor(max_workers=len(eyes), mp_context=get_context('spawn')) as pool:
        jobs = {eye: pool.submit(evaluate, eye, wiring, calibration['measurements'], candidate) for eye in eyes}
        for eye, job in jobs.items():
            receipt['variants'][eye] = job.result()
            _write(folder / f'{MOTOR}-games.json.gz', receipt)


def verify_games(path):
    data = _read(path)
    calibration = path.parent / f'{MOTOR}-probes.json.gz'
    if _hash(calibration) != data['preparation_sha256'] or data['arena'] != SETTINGS:
        raise ValueError('the scored game and its optical preparation differ')
    fits = _read(calibration)['measurements']
    wiring = Wiring.restore(data['private_evaluator_map'])
    count = 0
    for variants in data['variants'].values():
        for label, condition in variants.items():
            reference = condition.get('equivalent_to')
            game = variants[reference]['data'] if reference is not None else condition['data']
            source = condition['source']
            correction = None if source is None else Wiring.restore(fits[source]['inferred'])
            result = replay(game, IDENTITY if label == 'normal' else wiring, correction,
                            check_recorded=reference is None)
            if result != condition['physical_replay']:
                raise ValueError('the verified point count differs from the stored game')
            count += result['points']
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=('probe', 'game', 'all'), default='all')
    parser.add_argument('--eye', choices=EYES, action='append')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print('replayed', verify_games(args.folder / f'{MOTOR}-games.json.gz'))
    else:
        run(args.folder, args.phase, args.eye or EYES)

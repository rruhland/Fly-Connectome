"""Fit SO-101 motion from optical probes and qualify trajectory-search games."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_continuous_trial import match_phases, prepare_point
from .tt6_memory import load
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore
from .tt8_service_trial import PointInterface
from .tt19_flight import FlightLaw
from .tt19_online_body import kernel_for
from .tt19_shoot import BOUND, WIDTH, OpticalBody, ShootingPlayer, observed_pose, qualification
from .tt19_stroke_trial import IdentityWiring
from .tt19_view import ViewTransport
from .tt19_view_trial import SO101ViewArena


SEED = 409676221
ARENA = dict(width=256, height=192, max_steps=115, frameskip=8, detail='preview',
             venue='hall', strict_endline=True, view_mode='wrist+fixed')
DONOR = ROOT / 'artifacts/tt17/motor-gantry-coupling-travel-player.json.gz'
SOURCE = ROOT / 'artifacts/tt19/flight-source.json.gz'
STEPS = 14
PROBE_SEED = 588931301
PHASES = ('ordinary', 'ordinary', 'soft', 'mixed')


def probe_tapes():
    rng = np.random.default_rng(PROBE_SEED)
    for index in range(20):
        action = np.zeros((STEPS, WIDTH + 1))
        if index < 10:
            axis, sign = index // 2, 1. if index % 2 else -1.
            action[:5, axis] = sign * .73
            action[7:11, axis] = -.54 * sign
            pick = (axis + 1 + index % 3) % WIDTH
            action[6:11, pick] = .34 * (1. if index % 4 < 2 else -1.)
        else:
            n = rng.integers((3, 7), (7, 11))
            action[:n[0], :WIDTH] = rng.uniform(-.74, .74, WIDTH)
            action[n[0]:n[1], :WIDTH] = rng.uniform(-.74, .74, WIDTH)
            action[n[1]:, :WIDTH] = rng.uniform(-.34, .34, WIDTH)
        request = np.clip(np.cumsum(action, axis=0), -BOUND, BOUND)
        action[:] = np.diff(request, axis=0, prepend=np.zeros_like(request[:1]))
        yield action


def _serial(pose):
    return [None] * 5 if pose is None else [None if not np.isfinite(value) else float(value) for value in pose]


def calibrate(output):
    old = _read(DONOR)
    body = load(DONOR)[0].models[old['gain_memory']['maps'][old['gain_memory']['active']]['model']]
    arena, transport = SO101ViewArena(**ARENA), ViewTransport('wrist+fixed')
    records = []
    try:
        seeds = _seeds(PROBE_SEED, 73094, 0, 20)['tests']
        for index, (tape, seed) in enumerate(zip(probe_tapes(), seeds, strict=True)):
            tracker = transport.tracker()
            tracker.update(arena.reset(seed=seed, server=-1))
            poses, actions = [_serial(observed_pose(tracker))], []
            for action in tape:
                frame, _, done, capped, _ = arena.step(action)
                tracker.update(frame)
                poses.append(_serial(observed_pose(tracker)))
                actions.append(action.tolist())
                if done or capped:
                    break
            records.append(dict(seed=seed, actions=actions, poses=poses, rgb_sha256=tracker.digest.hexdigest()))
            print('probe', index + 1, len(actions), sum(row[3] is not None for row in poses), flush=True)
    finally:
        arena.close()
    trials = {}
    training, validation = records[:16], records[16:]
    for mode, kernel in [('nominal', kernel_for()), ('gantry', kernel_for(body))]:
        for lag in (0, 1, 2):
            fitted = OpticalBody.fit(training, kernel, lag)
            trials[f'{mode}-{lag}'] = dict(fit=fitted.state(), training=qualification(fitted, training),
                                          validation=qualification(fitted, validation))
    key = min(trials, key=lambda name: trials[name]['validation']['median_m'] + .35 * trials[name]['validation']['p90_m'])
    result = dict(schema='tt19-optical-shoot-fit-v1', arena=ARENA, donor_sha256=_hash(DONOR),
                  training=training, validation=validation, selected=key, candidates=trials,
                  setup_frames=sum(len(row['actions']) for row in records))
    _write(output, result)
    print('fit',key,{name: row['validation'] for name,row in trials.items()},flush=True)


def parameters(mode):
    law = FlightLaw() if mode == 'nominal' else FlightLaw.from_state(_read(SOURCE)['source_law'])
    state = {} if mode == 'nominal' else _read(DONOR)['series']
    impulse = FlightMemory('impulse', state.get('flight_history', ()), state.get('flight_impulses', ()), frozen=True)
    return law, impulse.response_matrix


def _point(arena, wiring, model, law, response, transport, game_seed, seed, phase, server, count):
    interface = PointInterface(wiring, prepare_point(arena, phase), server=server, seed=game_seed)
    player = ShootingPlayer(transport.tracker(), model, law, response, server, seed, count=count)
    frame = interface.reset(seed=seed)
    actions = []
    while True:
        player.observe(frame)
        command = player.choose()
        actions.append(command.tolist())
        frame, reward, done, capped, _ = interface.step(command)
        if done or capped:
            player.observe(frame)
            break
    return dict(seed=seed, phase=phase, server=server, actions=actions, applied=interface.applied,
                action_sha256=sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest(),
                rgb_sha256=player.tracker.digest.hexdigest(), optical=player.optical, plans=player.shots,
                public_reward=reward, evaluator=arena.point_result(), events=arena.events.copy())


def summary(rows):
    cases = [(row, row['evaluator']) for row in rows]
    return dict(steps=sum(len(row['actions']) for row in rows),
                return_contacts=sum(stat['contacts'] for row, stat in cases if row['server'] == -1),
                legal_returns=sum(stat['legal_returns'] for row, stat in cases),
                return_points=sum(stat['legal_returns'] > 0 for row, stat in cases),
                multiple_returns=sum(stat['legal_returns'] > 1 for row, stat in cases),
                serve_attempts=sum(row['server'] == 1 for row, stat in cases),
                serve_contacts=sum(stat['service_contact'] for row, stat in cases if row['server'] == 1),
                legal_serves=sum(stat['legal_serve'] for row, stat in cases if row['server'] == 1),
                planned_points=sum(bool(row['plans']) for row in rows),
                predicted_hits=sum(any(p['predicted_contact'] for p in row['plans']) for row in rows))


def schedule(seed):
    points = _seeds(seed, 184790, 0, 60)['tests']
    phases = match_phases(PHASES, len(points), [seed, 912943])
    return zip(points, phases, strict=True)


def verify(report):
    arena = SO101ViewArena(**report['arena'])
    board = ServiceScore(first_server=-1)
    try:
        wiring = IdentityWiring(arena)
        for row, (seed, phase) in zip(report['points'], schedule(report['seed'])):
            if (row['seed'], row['phase'], row['server']) != (seed, phase, board.server):
                raise ValueError('saved opponent or serve schedule differs from replay')
            interface = PointInterface(wiring, prepare_point(arena, phase), server=board.server, seed=report['seed'])
            interface.reset(seed=seed, render=False)
            for action in row['actions']:
                interface.step(action, render=False)
            digest = sha256(np.asarray(row['actions'], dtype='<f8').tobytes()).hexdigest()
            if (digest != row['action_sha256'] or interface.applied != row['applied']
                    or json.dumps(arena.events, sort_keys=True) != json.dumps(row['events'], sort_keys=True)
                    or arena.point_result() != row['evaluator']):
                raise ValueError(f'physical replay failed for point {seed}')
            board.record(row['evaluator'])
        if list(board.score) != report['score'] or board.finished != report['finished']:
            raise ValueError('saved score differs from replay')
    finally:
        arena.close()
    return len(report['points'])


def verify_decisions(report, fit):
    if report['calibration_sha256'] != _hash(fit):
        raise ValueError('the optical motion calibration changed')
    setup = _read(fit)
    body = OpticalBody.restore(setup['candidates'][report['body']]['fit'])
    law, response = FlightLaw.from_state(report['law']), np.array(report['response'])
    for row in report['points']:
        player = ShootingPlayer(None, body, law, response, row['server'], row['seed'],
                                count=report['planner_candidates'])
        observations = row['optical']
        if len(observations) != len(row['actions']) + 1:
            raise ValueError('a control step lacks its optical observation')
        for tick, (expected, frame) in enumerate(zip(row['actions'], observations[:-1], strict=True)):
            player.tick = tick
            if frame['tick'] != tick:
                raise ValueError('optical clock differs from the action tape')
            pose = None if frame['blade'] is None else np.asarray(frame['blade'], dtype=float)
            player.observe_measurements(frame['ball'], pose)
            if player.choose().tolist() != expected:
                raise ValueError(f"optical action replay diverged on point {row['seed']} at tick {tick}")
        if player.shots != row['plans']:
            raise ValueError('recaptured shot decisions differ from the saved plan')
    return len(report['points'])


def game(folder, fit, mode, seed, count, limit=None):
    setup = _read(fit)
    model = OpticalBody.restore(setup['candidates'][setup['selected']]['fit'])
    arena = SO101ViewArena(**ARENA)
    law, response = parameters(mode)
    board = ServiceScore(first_server=-1)
    transport = ViewTransport('wrist+fixed')
    name = f'shoot-{seed}-{mode}' + ('-pilot' if limit is not None else '')
    output = folder / f'{name}.json.gz'
    rows = []
    try:
        wires = IdentityWiring(arena)
        for index, (point, phase) in enumerate(schedule(seed), 1):
            row = _point(arena, wires, model, law, response, transport, seed, point, phase, board.server, count)
            rows.append(row)
            board.record(row['evaluator'])
            faults = row['evaluator']
            last = row['plans'][-1] if row['plans'] else {}
            print(mode, index, row['server'], board.score, faults['reason'], faults['legal_returns'], faults['contacts'],
                  len(row['plans']), round(last.get('separation', 0),4), flush=True)
            if board.finished or (limit is not None and index >= limit):
                break
    finally:
        arena.close()
    if limit is None and not board.finished:
        raise ValueError('no finished game under the held start limit')
    description = dict(mode=mode, seed=seed, planner_candidates=count, law=law.state(), response=response.tolist(),
                       body=setup['selected'], calibration_sha256=_hash(fit), setup_frames=setup['setup_frames'])
    checkpoint = folder / f'{name}-player.json.gz'
    state = _read(DONOR)
    state['tt19_shoot'] = dict(**description, optical_body=model.state(), camera_fits=transport.state())
    _write(checkpoint, state)
    result = dict(schema='tt19-optical-shoot-game-v1', arena=ARENA, **description, points=rows,
                  score=list(board.score), finished=board.finished, metrics=summary(rows),
                  checkpoint=checkpoint.name, checkpoint_sha256=_hash(checkpoint))
    _write(output, result)
    result['replayed_points'] = verify(_read(output))
    result['replayed_optical_points'] = verify_decisions(_read(output), fit)
    _write(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('operation', choices=('fit', 'nominal', 'gantry', 'verify'))
    p.add_argument('--fit', type=Path)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--count', type=int, default=360)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    a.folder.mkdir(exist_ok=True, parents=True)
    fit = a.fit or a.folder / 'shoot-body-fit.json.gz'
    if a.operation == 'fit':
        calibrate(fit)
    elif a.operation == 'verify':
        for mode in ('nominal', 'gantry'):
            receipt = a.folder / f'shoot-{a.seed}-{mode}.json.gz'
            result = _read(receipt)
            result['replayed_points'] = verify(result)
            result['replayed_optical_points'] = verify_decisions(result, fit)
            _write(receipt, result)
            print(mode, result['replayed_points'], result['replayed_optical_points'])
    else:
        game(a.folder, fit, a.operation, a.seed, a.count, a.limit)


if __name__ == '__main__':
    main()

"""Collect charged teacher recoveries from pupil-induced states, test without teacher."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from hashlib import sha256
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt22_flight_inverse_trial import InversePlayer
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import teacher_observation
from .tt23_sequence import REPLICATION, course, verify
from .tt24_rally_bank import CACHE, DIMS, HOLDOUT
from .tt24_rally_improve_phase import PhaseHistory, PhasePolicy
from .tt24_rally_improve_trial import policy_for
from .tt24_rally_policy import EYES, JointPolicy, ViewHistory
from .tt24_rally_trial import (COACHED, OUTPUT, STEP_SIZE, PublicBody, Station, alternating,
                               difference_count, returns, summary)


PRACTICE = OUTPUT / 'rally-improve-recovery-practice.json.gz'
DATA = OUTPUT / 'rally-improve-recovery-demos.npz'
HELD = OUTPUT / 'rally-improve-recovery-held.json.gz'
FRESH = {'q': 1268857121, 'r': 610843651}
MODES = ('broad', 'corrected', 'sight_delay', 'hold')
WORKER = None


def actor(path):
    keys = ('features', 'postures', 'rates')
    with np.load(CACHE) as original, np.load(path) as corrective:
        arrays = [np.concatenate((original[key], corrective[key])) for key in keys]
    return JointPolicy(*arrays, STEP_SIZE, neighbors=28, body=.68, mix=0.)


def prepare(teacher, data_path=None):
    global WORKER
    opening = _read(COACHED)['metadata']['opening']
    station = Station(opening['far_profile'], render_dims=DIMS)
    with np.load(CACHE) as data:
        broad = policy_for('broad', data)
    if teacher:
        player = InversePlayer(FlightLaw())
        player.memory = ReturnMemory(opening['evidence'])
        advance = player.sim.twin.advance
        # The teaching arm must follow the actual learner's issued commands
        # before the transfer of authority, including during its first return.
        player.sim.twin.advance = lambda _: None
        WORKER = station, broad, player, advance
    else:
        WORKER = station, broad, PhasePolicy(broad, actor(data_path), .92), None


def practice_row(start, switch_after):
    station, broad, teacher, advance = WORKER
    arena = station.arena
    arena.lane = start['lane']
    arena.reset(seed=start['seed'], server=-1, render=False)
    station.tape.steps.clear()
    teacher.reset()
    history = ViewHistory()
    body = PublicBody(arena)
    digest = sha256()
    near = []
    features, postures, rates = [], [], []
    takeover = None
    for tick in range(arena.max_steps):
        frames = arena.render_cameras()
        for eye in EYES:
            digest.update(eye.encode())
            digest.update(frames[eye])
        observation = history.observe(frames, arena.joints, body.target)
        if takeover is None and arena.legal_returns >= switch_after:
            takeover = tick
        if takeover is None:
            command = broad.command(observation, body.target)
        else:
            physical = teacher_observation(arena)
            # This evaluator-only demonstrator may access simulator flight.
            command = teacher.command(tick, physical.center, teacher=physical)
        _, reward, done, capped, _ = arena.step(command, render=False)
        near.append(command.tolist())
        body.issued(command)
        history.issued(command)
        advance(command)
        if takeover is not None:
            features.append(observation)
            postures.append(body.target.copy())
            rates.append(command)
        if done or capped:
            break
    row = dict(start, reward=reward, outcome=arena.point_result(), longest=alternating(arena.events),
               events=arena.events.copy(), rgb=digest.hexdigest(), actions=near, rival_actions=station.tape.tape(len(near)),
               transfer=takeover, pupil_frames=len(near) if takeover is None else takeover,
               teacher_frames=0 if takeover is None else len(near)-takeover, transfer_after=switch_after)
    learned = 0 if takeover is None else sum(tick > takeover for tick in returns(row, 1))
    batch = dict(features=np.empty((0, 38)), postures=np.empty((0, 6)), rates=np.empty((0, 6)))
    if learned:
        last_legal = returns(row, 1)[-1]
        batch = dict(features=np.asarray(features[:last_legal-takeover]),
                     postures=np.asarray(postures[:last_legal-takeover]),
                     rates=np.asarray(rates[:last_legal-takeover]))
    row['extra_teacher_returns'] = learned
    row['fit_frames'] = len(batch['features'])
    return row, batch


def collect(path, data_path, workers):
    development = course(HOLDOUT)[:12]
    schedule = [(row, 1) for row in development]
    meta = dict(schema='tt24-mixed-initiative-recovery-1', body=list(DIMS),
                demonstrations_sha256=_hash(CACHE), development=development, teacher_origins='TT23 inverse-flight tutor',
                teacher_feedback=['ball position', 'ball velocity', 'spin', 'bounce status'],
                transition_rule='first pupil legal return')
    result = dict(metadata=meta, construction=[])
    blocks = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=prepare, initargs=(True,)) as pool:
        tasks = [pool.submit(practice_row, *visit) for visit in schedule]
        for task in tasks:
            row, arrays = task.result()
            blocks.append(arrays)
            result['construction'].append(row)
            print('correction', row['seed'], row['transfer'], row['extra_teacher_returns'], row['fit_frames'], flush=True)
            _write(path, result)
    # Exact verification treats all pupil and teacher commands as one physical tape.
    result['verified_tapes'] = verify([dict(results={'construction': row}) for row in result['construction']])
    result['cost'] = dict(episodes=len(schedule), teacher_frames=sum(row['teacher_frames'] for row in result['construction']),
                          pupil_frames=sum(row['pupil_frames'] for row in result['construction']),
                          fit_frames=sum(row['fit_frames'] for row in result['construction']))
    column = {key: np.concatenate([block[key] for block in blocks]) for key in ('features', 'postures', 'rates')}
    seeds = np.concatenate([np.full(len(block['features']), row['seed']) for block,row in zip(blocks, development)])
    np.savez_compressed(data_path, **column, seed=seeds)
    result['data_sha256'] = _hash(data_path)
    _write(path, result)
    return result


def held_case(start, order):
    station, broad, corrected, _ = WORKER
    branches = {}
    for name in order:
        if name in ('broad', 'hold'):
            branches[name] = station.play(start, name, broad, live_opening=True)
        else:
            history = lambda: PhaseHistory(delayed=name == 'sight_delay')
            branches[name] = station.play(start, 'correction', corrected, history, live_opening=True)
    if len({branch['prefix_rgb'] for branch in branches.values()}) != 1:
        raise ValueError('initial visual evidence differs')
    for branch in branches.values():
        branch['vs_broad'] = difference_count(branch, branches['broad'], 0)
    return dict(start, results=branches)


def evaluate(path, data_path, practice, workers):
    starts = course(FRESH)
    known = {row['seed'] for row in course(HOLDOUT)+course(REPLICATION)}
    with np.load(CACHE) as bank, np.load(data_path) as taught:
        known.update(map(int, set(bank['seed']) | set(taught['seed'])))
    if known & {row['seed'] for row in starts}:
        raise ValueError('held point overlaps training')
    meta = dict(schema='tt24-mixed-initiative-correction-held-1', practice_sha256=_hash(practice),
                correction_sha256=_hash(data_path), demos_sha256=_hash(CACHE), held=starts,
                active_teacher_commands=0, modes=list(MODES), resolution=list(DIMS))
    out = _read(path) if path.exists() else dict(metadata=meta, cases=[None]*len(starts))
    if out['metadata'] != meta:
        raise ValueError('different held test')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=prepare, initargs=(False, data_path)) as pool:
        tasks = {}
        for index, start in enumerate(starts):
            if out['cases'][index] is None:
                offset = index % len(MODES)
                tasks[pool.submit(held_case, start, MODES[offset:]+MODES[:offset])] = index
        for task in as_completed(tasks):
            row = task.result()
            out['cases'][tasks[task]] = row
            _write(path, out)
            print('recovery held', row['seed'], {mode: returns(result, 1) for mode,result in row['results'].items()}, flush=True)
    out['verified_tapes'] = verify(out['cases'])
    out['summary'] = {name: summary(out['cases'], name, 0) for name in MODES}
    for row in out['summary'].values():
        row['reach_first'] = sum(n > 0 for n in row['own_returns'])
    _write(path, out)
    return out['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, default=HELD)
    parser.add_argument('--practice', type=Path, default=PRACTICE)
    parser.add_argument('--dataset', type=Path, default=DATA)
    args = parser.parse_args()
    if not args.dataset.exists() or not args.practice.exists():
        evidence = collect(args.practice, args.dataset, args.workers)
    else:
        evidence = _read(args.practice)
    if evidence['cost']['fit_frames'] > 0:
        print(evaluate(args.output, args.dataset, args.practice, args.workers), flush=True)
    else:
        print(evidence['cost'], flush=True)

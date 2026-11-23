"""Test the frozen RGB joint pupil on fresh physical releases with a live teacher opening."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive, lateral_goal
from .tt23_sequence import HANDOVER, course, REPLICATION, verify
from .tt24_rally_trial import COACHED, DATA, OUTPUT, Station, contacts, difference_count, make_policy, returns, summary


CONDITIONS = ('hold', 'sight_delay', 'demo_full')
FRESH = OUTPUT / 'rally-fresh.json.gz'
SETTINGS = dict(neighbors=20, body=.22, mix=0.)
TRIAL = None


class OpeningTape:
    """Gather the shared teacher prefix and score its result with zero continuation."""

    def __init__(self, station):
        self.station = station
        self.arena = station.arena
        self.actions = []
        self.terminal = False
        self.reward = 0

    def lens_intrinsics(self):
        return self.arena.lens_intrinsics()

    def render_cameras(self):
        return self.arena.render_cameras()

    def reset(self, **kwargs):
        self.arena.lane = self.lane
        self.station.tape.steps.clear()
        self.actions.clear()
        self.terminal = False
        self.arena.reset(**kwargs)

    def step(self, command, *, render=False):
        _, self.reward, done, cap, _ = self.arena.step(command, render=render)
        self.actions.append(np.asarray(command).tolist())
        self.terminal = done or cap

    def try_opening(self, start, macro, gains):
        optical = prefix(self, start['seed'], start['lane'])
        primitive = MotorPrimitive(macro, None)
        if optical['mover'] is not None:
            mover, speed = np.asarray(optical['mover'])
            ready = primitive.ready_world
            target, _ = lateral_goal(ready, mover, speed, np.r_[0., gains[:2]])
            primitive._translate(target, ready)
        for tick in range(len(self.actions), self.arena.max_steps):
            if self.terminal:
                break
            command = primitive.command(tick) if tick < HANDOVER else np.zeros(self.arena.action_size)
            self.step(command)
        row = dict(start, actions=self.actions.copy(), rival_actions=self.station.tape.tape(len(self.actions)),
                   events=self.arena.events.copy(), outcome=self.arena.point_result(), optical_prefix=optical['rgb'])
        owned, partner = returns(row, 1), returns(row, -1)
        eligible = bool(owned and partner and contacts(row, 1)[0] <= HANDOVER)
        return row, eligible


def prepare(data_path, opening):
    global TRIAL
    station = Station(opening['far_profile'])
    with np.load(data_path) as data:
        model, _ = make_policy(data, 'demo_full', **SETTINGS)
    TRIAL = station, model, opening


def score(index, start):
    station, model, opening = TRIAL
    tape = OpeningTape(station)
    screen = {}
    case = dict(start, screening=screen, results={})
    for mode in ('neutral', 'duration'):
        row, candidate = tape.try_opening(start, opening['macro'], opening['gains'][mode])
        screen[mode] = dict(eligible=candidate, near=returns(row, 1), far=returns(row, -1),
                            contacts=contacts(row, 1), outcome=row['outcome'], optical_prefix=row['optical_prefix'])
        if candidate:
            case.update(opening_mode=mode, teacher_prefix=row['actions'][:HANDOVER])
            break
    else:
        return index, case
    order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
    for name in order:
        case['results'][name] = station.play(row, name, model)
    if len({result['prefix_rgb'] for result in case['results'].values()}) != 1:
        raise ValueError('fresh opening RGB differs across conditions')
    for result in case['results'].values():
        result['vs_hold'] = difference_count(result, case['results']['hold'])
    return index, case


def run(path, data_path, workers, limit):
    opening = _read(COACHED)['metadata']['opening']
    starts = course(REPLICATION)[:limit]
    metadata = dict(schema='tt24-fresh-rally-rgb-joint-1', handoff=HANDOVER, conditions=list(CONDITIONS),
                    schedules=REPLICATION, settings=SETTINGS, archive_sha256=_hash(COACHED),
                    demo_data_sha256=_hash(data_path), releases=starts)
    report = _read(path) if path.exists() else dict(metadata=metadata, cases=[None]*len(starts))
    if report['metadata'] != metadata:
        raise ValueError('existing fresh receipt uses a different schedule or pupil')
    path.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=prepare,
                             initargs=(data_path, opening)) as pool:
        work = [pool.submit(score, i, start) for i, start in enumerate(starts) if report['cases'][i] is None]
        for task in as_completed(work):
            index, case = task.result()
            report['cases'][index] = case
            outcomes = {name: returns(row, 1) for name, row in case['results'].items()}
            print('fresh', case['seed'], case.get('opening_mode', 'ineligible'), outcomes, flush=True)
            _write(path, report)
    report['verified_tapes'] = verify(report['cases'])
    report['eligible'] = sum(bool(case['results']) for case in report['cases'])
    report['summary'] = {name: summary(report['cases'], name) for name in CONDITIONS}
    _write(path, report)
    return report['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=FRESH)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    print(run(args.output, args.data, args.workers, args.limit), flush=True)

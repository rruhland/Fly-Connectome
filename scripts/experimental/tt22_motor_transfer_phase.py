"""Try direct reward-only phase adjustment before a harder matched holdout."""

import argparse
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_motor_transfer_trial import MACRO, OUTPUT as PILOT, MotorArena, metrics, point, schedule, verify


OUTPUT = PILOT.with_name('motor-transfer-phase.json.gz')
SCREEN = 217558016
HELD = 2136329085
PHASES = (-6, -3, 0, 3, 6)
SCREEN_LANES = (-.56, -.28, .28, .56)


def template(macro, phase):
    shifted = np.array(macro, dtype=float)
    shifted[10] += phase
    return shifted


def matched_point(arena, seed, lane, macro, coefficients, phase, prefix=None):
    tape = template(macro, phase)
    if prefix is not None and (prefix['decision'] is None or prefix['decision']['tick'] >= tape[10]):
        prefix = None
    row = point(arena, seed, lane, tape, coefficients, shared_prefix=prefix)
    return dict(row, phase_offset_frames=phase)


def choose(rows):
    return max(PHASES, key=lambda phase: (
        sum(row['outcome']['legal_returns'] > 0 for row in rows[str(phase)]),
        sum(row['outcome']['legal_returns'] for row in rows[str(phase)]),
        sum(row['reward'] for row in rows[str(phase)]), -abs(phase)))


def first_n(rows, threshold):
    returns = 0
    for index, row in enumerate(rows, start=1):
        returns += row['outcome']['legal_returns'] > 0
        if returns >= threshold:
            return index
    return None


def run(count):
    pilot = _read(PILOT)
    macro = _read(MACRO)['parameters']
    if pilot['macro_sha256'] != _hash(MACRO):
        raise ValueError('timing screen must inherit the original macro')
    candidates = {name: pilot['no_target_practice'][name] for name in ('scratch','pong')}
    screen = schedule(SCREEN, SCREEN_LANES)
    half = np.linspace(.28, .94, count//2).tolist()
    held = schedule(HELD, tuple([-lane for lane in half]+half))
    prior = {seed for stage in pilot['profiles'].values() for seed, _ in stage}
    starts = [seed for seed, _ in screen+held]
    if set(starts) & prior or len(set(starts))!=len(starts):
        raise ValueError('phase screen and harder held launches must be disjoint')
    data = dict(schema='tt22-motor-transfer-phase-1', source=PILOT.name, source_sha256=_hash(PILOT),
                macro_sha256=_hash(MACRO), phase_search=PHASES, phase_scores='actual scalar learner legal returns',
                candidates=candidates, screen_schedule=screen, held_schedule=held,
                screen={n: {str(phase): [] for phase in PHASES} for n in candidates}, held={n: [] for n in candidates})
    began = perf_counter()
    arena = MotorArena()
    try:
        for seed, lane in screen:
            common = matched_point(arena, seed, lane, macro, candidates['pong'], 0)
            for name, coefficients in candidates.items():
                for phase in PHASES:
                    row = (common if name == 'pong' and phase == 0
                           else matched_point(arena, seed, lane, macro, coefficients, phase, common))
                    data['screen'][name][str(phase)].append(row)
            print('screen', seed, lane, {name: [rs[-1]['outcome']['legal_returns'] for rs in rows.values()]
                                        for name, rows in data['screen'].items()}, flush=True)
            _write(OUTPUT, data)
        best = {name: choose(rows) for name, rows in data['screen'].items()}
        data['selected_phase'] = best
        data['screen_summary'] = {name: {phase: metrics(rows) for phase, rows in programs.items()}
                                  for name, programs in data['screen'].items()}
        selected_points = [data['screen_summary'][name][str(phase)]['first_returns'] for name, phase in best.items()]
        viable = max(selected_points) >= 2 and any(phase != 0 for phase in best.values())
        data['qualified_for_held'] = viable
        _write(OUTPUT, data)
        print('selected', best, selected_points, 'held', viable, flush=True)
        if viable:
            for i, (seed, lane) in enumerate(held):
                order = ('scratch','pong') if i % 2 == 0 else ('pong','scratch')
                captured = None
                for name in order:
                    row = matched_point(arena, seed, lane, macro, candidates[name], best[name], captured)
                    data['held'][name].append(row)
                    captured = row
                data['held_summary'] = {n: dict(metrics(rs), points_to_first=first_n(rs,1),
                                                points_to_four=first_n(rs,4)) for n,rs in data['held'].items()}
                print('held', i+1, seed, lane, [rows[-1]['outcome']['legal_returns'] for rows in data['held'].values()], flush=True)
                _write(OUTPUT, data)
        screen_rows = [row for programs in data['screen'].values() for rows in programs.values() for row in rows]
        data['verified'] = verify(arena, screen_rows+[row for rows in data['held'].values() for row in rows])
        data['wall_seconds'] = perf_counter()-began
        _write(OUTPUT, data)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held', type=int, choices=(12,24,48), default=24)
    args = parser.parse_args()
    run(args.held)

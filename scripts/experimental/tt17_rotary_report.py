"""Compare deadline weighting on held serves, exact replies, and completed games."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt6_body_arena import TT6Wiring
from .tt8_service_arena import TT8ServiceArena
from .tt13_rotary_report import same_incoming_events
from .tt13_rotary_trial import path
from .tt14_rotary_trial import matched_control
from .tt15_rotary_report import second_returns, sustained_pairs
from .tt17_rotary_diagnostics import measure, physics, second_events
from .tt17_rotary_player import replay_first_pace
from .tt17_rotary_trial import MODES, SEEDS


def summary(receipt):
    rows = receipt['points']
    roles = {'receive': [r for r in rows if r['server'] == -1], 'serve': [r for r in rows if r['server'] == 1]}
    return dict(score=receipt.get('score'), metrics=receipt['metrics'], replayed_points=receipt['replayed_points'],
                second={side: second_returns(points) for side, points in roles.items()},
                weighted_frames=sum(r['deadline_frames'] for r in rows),
                weighted_points=sum(r['deadline_frames'] > 0 for r in rows),
                speed_frames=sum(r.get('speed_frames', 0) for r in rows),
                speed_points=sum(r.get('speed_frames', 0) > 0 for r in rows),
                pace_references=sum(r.get('first_optical_pace') is not None for r in rows))


def describe_attempts(rows):
    windows = {}
    for lead in (12, 8, 4, 1):
        samples = [row['milestones'][str(lead)] for row in rows if 'milestones' in row
                   and row['milestones'][str(lead)] is not None]
        windows[str(lead)] = dict(observed=len(samples),
                                  median_absolute_frames=float(np.median([abs(r['error_frames']) for r in samples]))
                                  if samples else None,
                                  median_absolute_xz_m=np.median(np.abs([r['error_xz'] for r in samples]), axis=0).tolist()
                                  if samples else None)
    gaps = [row['actual_gap'] for row in rows if 'actual_gap' in row]
    blade_speeds = [row['blade_forward_mps'] for row in rows if 'blade_forward_mps' in row]
    return dict(attempts=len(rows), classes=dict(Counter(r['classification'] for r in rows)), milestones=windows,
                median_absolute_gap_m=np.median(np.abs(gaps), axis=0).tolist() if gaps else None,
                median_blade_forward_mps=float(np.median(blade_speeds)) if blade_speeds else None)


def check_recorded_pace(row):
    trace = row['optical_trace']
    measured = replay_first_pace(trace, row['timing_route']['first_receiver_frame'])
    weighted = sum(frame['speed_weighted'] is True for frame in trace)
    if measured != row['first_optical_pace'] or weighted != row['speed_frames']:
        raise ValueError('the stored speed reference or application cannot be replayed from camera data')


def inspect(receipt):
    arena = TT8ServiceArena(**receipt['arena'])
    wiring = TT6Wiring(arena, receipt['installation_seed'])
    results = {}
    try:
        for row in receipt['points']:
            events = second_events(row)
            if events is not None:
                if 'first_optical_pace' in row:
                    check_recorded_pace(row)
                ball, blade = physics(receipt, row, arena, wiring)
                results[row['seed']] = (events[1], dict(measure(row, events, ball, blade, arena.control_dt),
                                                       deadline_frames=row['deadline_frames'],
                                                       speed_frames=row.get('speed_frames', 0),
                                                       first_optical_pace=row.get('first_optical_pace')))
    finally:
        arena.close()
    return results


def matched_approaches(before, after):
    results = {}
    for seed in before.keys() & after.keys():
        (a, left), (b, right) = before[seed], after[seed]
        if same_incoming_events([a], [b]):
            results[seed] = left, right
    return results


def first_contact_prefix(base, revised):
    peer = {row['seed']: row for row in revised['points']}
    equal = 0
    for a in base['points']:
        b = peer.get(a['seed'])
        if a['server'] != -1 or b is None or b['server'] != -1:
            continue
        end = next((event['step'] for event in a['events']
                    if event['kind'] == 'contact' and event.get('side') == 1), a['steps'])
        equal += a['actions'][:end] == b['actions'][:end]
    return equal


def match_qualification(folder, include_repeat):
    sections = {}
    totals = {mode: [] for mode in MODES}
    common = {mode: [] for mode in MODES}
    for fixture in SEEDS:
        base, revised = [_read(path(folder, f'second-{mode}-{fixture}')) for mode in MODES]
        if base['schedule'] != revised['schedule'] or base['source_sha256'] != revised['source_sha256']:
            raise ValueError('the match schedules and starting controller must agree')
        paired = matched_control(base, revised)
        paired['sustained_points'] = sustained_pairs(base, revised)
        paired['exact_through_first_contact'] = first_contact_prefix(base, revised)
        earlier, later = map(inspect, (base, revised))
        matched = matched_approaches(earlier, later)
        for before, after in matched.values():
            common['base'].append(before)
            common['deadline'].append(after)
        paired['same_second_incoming'] = sorted(matched)
        for mode, measured in zip(MODES, (earlier, later), strict=True):
            totals[mode].extend(dict(row, fixture=fixture) for _, row in measured.values())
        sections[fixture] = dict(pairs=paired, branches={mode: summary(receipt)
                                                       for mode, receipt in zip(MODES, (base, revised), strict=True)})
    archived = _read(path(folder, 'second-diagnostics'))
    output = dict(fixtures=sections,
            baseline_archive=describe_attempts(archived['records']),
            second_attempts={mode: dict(summary=describe_attempts(rows), rows=rows) for mode, rows in totals.items()},
            matched_second_attempts={mode: describe_attempts(rows) for mode, rows in common.items()})
    if include_repeat:
        readings = {mode: _read(path(folder, f'second-{mode}-repeat')) for mode in (*MODES, 'speed')}
        measured = {mode: inspect(receipt) for mode, receipt in readings.items()}
        base = readings['base']
        repeat = dict(branches={mode: summary(receipt) for mode, receipt in readings.items()}, pairs={})
        for mode, receipt in readings.items():
            if mode == 'base':
                continue
            if base['schedule'] != receipt['schedule'] or base['source_sha256'] != receipt['source_sha256']:
                raise ValueError('the replication schedule or starting checkpoint changed')
            pairs = matched_control(base, receipt)
            matched = matched_approaches(measured['base'], measured[mode])
            repeat['pairs'][mode] = dict(pairs, exact_through_first_contact=first_contact_prefix(base, receipt),
                                         sustained_points=sustained_pairs(base, receipt),
                                         second_matches=matched)
        output['repeat'] = repeat
    _write(path(folder, 'second-comparison'), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    match_qualification(args.folder, args.repeat)

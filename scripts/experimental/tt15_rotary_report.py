"""Compare recovery work and second returns from matched rotary receipts."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt6_memory import load
from .tt13_rotary_report import same_incoming_events
from .tt13_rotary_trial import path
from .tt14_rotary_diagnostics import later_returns, matched_later, metrics as contact_metrics
from .tt14_rotary_trial import matched_control
from .tt15_rotary_player import reply_time
from .tt15_rotary_trial import FIXTURES, MODES, START, fixture_name


def second_returns(rows):
    chances = [r for r in rows if r['rival_returns'] > int(r['server'] == 1)]
    seconds = [r for r in chances if r['player_returns'] >= 2]
    return dict(opportunities=len(chances), landed=len(seconds), points_won=sum(r['points_for'] for r in seconds),
                seeds=[r['seed'] for r in seconds], longest=max((r['player_returns'] for r in rows), default=0))


def recovery_work(rows, ready_depth):
    work = np.zeros(3)
    timed_frames = planned_frames = optical_frames = reply_windows = 0
    for row in rows:
        active = []
        received = row['timing_route']['first_receiver_frame']
        last_attempt = -100.
        for frame in row['optical_trace']:
            tick, horizon = frame['tick'], frame['horizon']
            if received is not None and tick >= received:
                if horizon is not None and -1.5 < horizon < 2.5:
                    last_attempt = tick + horizon
                if 5. <= tick - last_attempt <= 24:
                    reply_windows += reply_time(frame, ready_depth) is not None
            if 'recovery' in frame:
                recovery = frame['recovery']
                active.append(recovery['command'])
                timed_frames += recovery['reply_frames'] is not None
                planned_frames += frame['horizon'] is not None
                optical_frames += recovery['error'] is not None
            if 'neutral_request' in frame:
                active.append(frame['neutral_request'])
        if active:
            magnitudes = np.abs(active).sum(axis=1)
            work += [np.sum(magnitudes > 1e-10), np.sum(magnitudes), np.any(magnitudes > 1e-10)]
    return dict(active_frames=int(work[0]), requested_travel=float(work[1]), points_with_recovery=int(work[2]),
                frames_per_start=float(work[0] / max(1, len(rows))),
                travel_per_start=float(work[1] / max(1, len(rows))), timing_available_frames=timed_frames,
                active_plan_frames=planned_frames, visible_blade_frames=optical_frames, observed_reply_frames=reply_windows)


def first_contacts(record):
    contacts = {}
    for row in record['points']:
        if row['server'] != -1:
            continue
        contact = next((event for event in row['events'] if event['kind'] == 'contact' and event['side'] == 1), None)
        if contact is not None:
            contacts[row['seed']] = row['actions'][:contact['step']], contact
    return contacts


def sustained_pairs(baseline, alternative):
    before = {r['seed']: r for r in baseline['points']}
    records = []
    for row in alternative['points']:
        other = before.get(row['seed'])
        if other is None or max(row['player_returns'], other['player_returns']) < 2:
            continue
        delta = next((index for index, (a, b) in enumerate(zip(other['actions'], row['actions']))
                      if a != b), None)
        replies = [[event for event in point['events'] if event['kind'] == 'legal_return' and event['side'] == -1]
                   for point in (other, row)]
        matched = [a['step'] for a, b in zip(*replies) if same_incoming_events([a], [b])]
        records.append(dict(seed=row['seed'], server=row['server'], first_changed_action=delta,
                            matched_reply_bounces=matched, player_returns=[other['player_returns'], row['player_returns']],
                            points_won=[other['points_for'], row['points_for']]))
    return records


def compare(folder, label, modes, fixtures):
    state = _read(START)
    cfg = state['rotary_return']
    ready = load(START)[0].models[cfg['maps']['recalled']].ready
    output = {}
    attempts = {mode: [] for mode in modes}
    for fixture in fixtures:
        name = fixture_name(fixture, label)
        records = {mode: _read(path(folder, f'{mode}-{name}')) for mode in modes}
        held = fixture == 0
        baseline = records['blind']
        block = dict(branches={}, pairs={})
        for mode, receipt in records.items():
            if ('replayed_points' not in receipt or receipt['schedule'] != baseline['schedule']
                    or receipt['source_sha256'] != baseline['source_sha256']):
                raise ValueError(f'{mode} did not verify or use the shared schedule and start')
            rows = receipt['points']
            sides = {'receive': [r for r in rows if r['server'] == -1], 'serve': [r for r in rows if r['server'] == 1]}
            attempts[mode].extend(dict(a, fixture=name) for row in rows for a in later_returns(row, ready))
            block['branches'][mode] = dict(score=receipt.get('score'), metrics=receipt['metrics'],
                                           second={side: second_returns(points) for side, points in sides.items()},
                                           recovery_work={side: recovery_work(points, ready[1])
                                                          for side, points in sides.items()},
                                           replayed_points=receipt['replayed_points'])
            bounded = [r['bounded_replies'] for r in rows if 'bounded_replies' in r]
            if bounded:
                block['branches'][mode]['bounded_replies'] = {
                    key: sum(r[key] for r in bounded) for key in bounded[0]
                }
            if mode == 'feedback':
                block['branches'][mode]['timed_feedback_frames'] = sum(r['timed_feedback_frames'] for r in rows)
            if mode != 'blind':
                pairing = matched_control(baseline, receipt)
                pair_a, pair_b = first_contacts(baseline), first_contacts(receipt)
                common = pair_a.keys() & pair_b.keys()
                pairing.update(common_first_contacts=len(common),
                               exact_first_strike_commands=sum(pair_a[seed][0] == pair_b[seed][0] for seed in common),
                               sustained_points=sustained_pairs(baseline, receipt))
                block['pairs'][mode] = pairing
        if held and len(baseline['points']) != len(baseline['schedule']['-1']):
            raise ValueError('the held cohort must replay every scheduled point')
        output[name] = block
    output['later_audit'] = dict(branches={mode: dict(metrics=contact_metrics(rows), attempts=rows)
                                        for mode, rows in attempts.items()},
                                 matching={mode: matched_later(attempts['blind'], rows)
                                           for mode, rows in attempts.items() if mode != 'blind'})
    _write(path(folder, f'comparison-{label}' if label else 'comparison'), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--modes', nargs='+', default=MODES)
    parser.add_argument('--fixtures', type=int, nargs='+', default=range(len(FIXTURES)))
    cli = parser.parse_args()
    compare(cli.folder, cli.label, cli.modes, cli.fixtures)

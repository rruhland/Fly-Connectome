"""Audit matched first-shot choices, actual second returns and optical outcome coverage."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt13_rotary_trial import path
from .tt14_rotary_trial import matched_control
from .tt15_rotary_report import second_returns, sustained_pairs
from .tt16_rotary_player import CONTEXT_SCALE, DEPTHS, FirstShotMemory, observed_outcome


def actual_first(row):
    return next((event for event in row['events'] if event['kind'] == 'legal_return' and event['side'] == 1), None)


def outcome_evidence(rows):
    categories = {name: dict(true=0, supported=0, both=0) for name in ('first_return', 'rival_reply', 'second_return')}
    supported = 0
    values = []
    for row in rows:
        first = row['first_shot']
        if first is None:
            continue
        supported += 1
        optical = first['outcome']
        values.append(optical['value'])
        events = dict(first_return=(row['player_returns'] >= 1,
                                    optical['first_landing'] is not None or optical['reply'] is not None),
                      rival_reply=(row['rival_returns'] >= 1, optical['reply'] is not None),
                      second_return=(row['player_returns'] >= 2, optical['second_landing'] is not None))
        for name, (true, seen) in events.items():
            categories[name]['true'] += int(true)
            categories[name]['supported'] += int(seen)
            categories[name]['both'] += int(true and seen)
    return dict(count=supported, mean_value=float(np.mean(values)) if values else None, categories=categories)


def outcome_check_older():
    details = {}
    for cohort in ('tt15/rotary-blind-held', 'tt14/rotary-recenter-held-replication'):
        source = ROOT / f'artifacts/{cohort}.json.gz'
        rows = _read(source)['points']
        examples = []
        for row in rows:
            begin = row['optical']['contact_prediction']
            if begin is not None:
                optical = observed_outcome(row['optical_trace'], begin, row['public_reward'])
                examples.append(dict(row, first_shot=dict(outcome=optical)))
        details[cohort] = dict(source_sha256=_hash(source), evidence=outcome_evidence(examples))
    return details


def summarize(receipt):
    rows = receipt['points']
    receiving = [row for row in rows if row['server'] == -1]
    serving = [row for row in rows if row['server'] == 1]
    choices = Counter(row['first_shot']['mode'] for row in receiving if row['first_shot'] is not None)
    if any(row['first_shot'] is not None for row in serving):
        raise ValueError('first return choice was applied during own service')
    lands = [event['location'][:2] for row in receiving if (event := actual_first(row)) is not None]
    sweeps = sum(row['first_shot'] is not None and row['first_shot'].get('sweep') is not None for row in receiving)
    return dict(score=receipt.get('score'), metrics=receipt['metrics'], chosen=dict(choices),
                second={side: second_returns(points) for side, points in (('receive', receiving), ('serve', serving))},
                first_landing_xy_m=lands, sweep_shots=sweeps, outcome_evidence=outcome_evidence(receiving),
                frames=sum(row['frames'] for row in rows), replayed_points=receipt['replayed_points'])


def compare_pair(base, alternative):
    result = matched_control(base, alternative)
    result['sustained_points'] = sustained_pairs(base, alternative)
    controls = {row['seed']: row for row in base['points'] if row['server'] == -1}
    groups = {}
    for row in alternative['points']:
        control = controls.get(row['seed'])
        if row['server'] != -1 or control is None:
            continue
        shot = row['first_shot']
        choice = 'base' if shot is None else shot['mode']
        if choice == 'center' and shot['sweep'] is None:
            choice = 'base'
        subset = groups.setdefault(choice, [])
        if base['practice']:
            end = len(row['actions']) if shot is None or choice == 'base' else shot['tick']
            if row['actions'][:end] != control['actions'][:end]:
                raise ValueError(f"matched tapes diverged before decision on {row['seed']}")
        subset.append((control, row))
    result['by_choice'] = {choice: dict(starts=len(rows),
                                       first=[sum(pair[i]['player_returns'] >= 1 for pair in rows) for i in (0, 1)],
                                       second=[sum(pair[i]['player_returns'] >= 2 for pair in rows) for i in (0, 1)],
                                       wins=[sum(pair[i]['points_for'] for pair in rows) for i in (0, 1)])
                           for choice, rows in groups.items()}
    both = [[actual_first(row) for row in pair] for pair in groups.get('center', ())]
    result['paired_sweep_landings'] = [[a['location'][:2], b['location'][:2]] for a, b in both if a and b]
    return result


def paired_contrast(memory):
    points = {}
    for obs in memory.observations:
        points.setdefault(tuple(obs['context']), {})[obs['mode']] = obs['outcome']['value']
    if any(len(actions) != len(DEPTHS) for actions in points.values()):
        raise ValueError('paired contrast requires visible context on the same development starts')
    cues = np.asarray(list(points))
    scores = np.asarray([[actions[mode] for mode in DEPTHS] for actions in points.values()])
    changes = scores[:, 1:] - scores[:, :1]
    distance = np.linalg.norm((cues[:, None, :] - cues[None, :, :]) / CONTEXT_SCALE, axis=-1)
    grid = []
    for width in (.35, .55, .8, 1.1):
        weight = np.exp(-.5 * (distance / width)**2)
        np.fill_diagonal(weight, 0.)
        expected = (weight @ changes) / (1. + weight.sum(axis=1, keepdims=True))
        for threshold in (.01, .035, .06):
            choice = np.argmax(np.column_stack((np.full(len(scores), threshold), expected)), axis=1)
            gain = scores[np.arange(len(scores)), choice] - scores[:, 0]
            grid.append(dict(width=width, threshold=threshold, gain=float(gain.sum()),
                             changed=int(np.count_nonzero(choice))))
    return grid


def sweep_practice(base, center):
    deltas = []
    for a, b in zip(base['points'], center['points'], strict=True):
        if a['seed'] != b['seed'] or a['phase'] != b['phase']:
            raise ValueError('sweep practice used a different matched start')
        left, right = a['first_shot'], b['first_shot']
        if right is None or right['sweep'] is None:
            continue
        if left is None or left['context'] != right['context']:
            raise ValueError('sweep context differed before the first-shot choice')
        deltas.append(right['outcome']['value'] - left['outcome']['value'])
    average = sum(deltas) / max(1, len(deltas))
    gains = sum(delta > 0. for delta in deltas)
    return dict(eligible=len(deltas), mean_gain=average, improves=gains,
                choice='center' if average > .025 and gains >= 2 else 'base')


def report(folder, fixtures, modes):
    memory = FirstShotMemory(_read(path(folder, 'first-fit'))['observations'])
    output = dict(optical_archive=outcome_check_older(), paired_contrast=paired_contrast(memory),
                  practice_values=memory.values(np.zeros(4), local=False), fixtures={})
    for fixture in fixtures:
        branches = tuple(DEPTHS) if fixture == 'practice' else modes
        if fixture == 'practice' and 'center' in modes:
            branches = (*branches, 'center')
        if fixture == 'repeat':
            branches = ('base', 'center')
        records = {mode: _read(path(folder, f'first-{mode}-{fixture}')) for mode in branches}
        base = records['base']
        for mode, receipt in records.items():
            if receipt['schedule'] != base['schedule'] or receipt['source_sha256'] != base['source_sha256']:
                raise ValueError('a comparison changed the initial player or fixture schedule')
            for row in receipt['points']:
                shot = row['first_shot']
                if shot is not None and mode in ('context', 'global'):
                    expected, scores = memory.choose(shot['context'], local=mode == 'context')
                    if (shot['mode'], shot['values']) != (expected, scores):
                        raise ValueError('the frozen choice could not be reproduced from optical context')
        output['fixtures'][fixture] = dict(branches={mode: summarize(receipt) for mode, receipt in records.items()},
                                           pairs={mode: compare_pair(base, receipt) for mode, receipt in records.items()
                                                  if mode != 'base'})
        if fixture == 'practice' and 'center' in records:
            output['sweep_practice'] = sweep_practice(base, records['center'])
    _write(path(folder, 'first-comparison'), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--fixtures', nargs='+', default=('practice', 'held', 'one', 'two', 'repeat'))
    parser.add_argument('--modes', nargs='+', default=('base', 'context', 'center'))
    cli = parser.parse_args()
    report(cli.folder, cli.fixtures, cli.modes)

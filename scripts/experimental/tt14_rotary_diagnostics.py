"""Audit optical availability and blade tracking on second and later returns."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt6_memory import load
from .tt11_stroke import _blade_motion
from .tt13_rotary_report import median, outcome
from .tt13_rotary_trial import path
from .tt14_rotary_trial import MODES, START, TRAIN


def later_returns(row, ready):
    """Use referee alignment after play to inspect the player's optical trace."""
    events = row['events']
    rival_bounces = [index for index, event in enumerate(events)
                     if event['kind'] == 'legal_return' and event['side'] == -1]
    if row['server'] == 1:
        rival_bounces = rival_bounces[1:]
    results = []
    trace = row['optical_trace']
    for bounce in rival_bounces:
        strike = next(event for event in reversed(events[:bounce])
                      if event['kind'] == 'contact' and event['side'] == -1)
        end = next((event['step'] for event in events[bounce + 1:]
                    if event['kind'] == 'contact' and event['side'] == -1), row['steps'] + 1)
        start = strike['step']
        contenders = [event for event in events[bounce + 1:] if event['step'] < end]
        charged = [frame for frame in trace if start - 6 <= frame['tick'] <= start and frame['blade'] is not None]
        home_offset = np.asarray(charged[-1]['blade']) - ready if charged else None
        observed = [frame for frame in trace if start <= frame['tick'] < end]
        plan = [frame for frame in observed if frame['horizon'] is not None]
        contact = next((event['step'] for event in contenders
                        if event['kind'] == 'contact' and event.get('side') == 1), None)
        legal = any(event['kind'] == 'legal_return' and event.get('side') == 1 for event in contenders)
        result = dict(seed=row['seed'], phase=row['phase'], server=row['server'], rival_strike=start,
                      legal_bounce=events[bounce]['step'], own_contact=contact, own_landed=legal,
                      incoming=[events[bounce]['ball'], events[bounce]['velocity']],
                      home_offset=None if home_offset is None else home_offset.tolist(),
                      planned_frames=len(plan))
        if plan:
            first = plan[0]
            nearest = min(plan, key=lambda frame: abs(frame['horizon']))
            deadline = nearest['tick'] + nearest['horizon']
            window = [frame for frame in observed if abs(frame['tick'] - deadline) <= 4]
            blade = min((frame for frame in plan if frame['blade'] is not None
                         and abs(frame['tick'] - deadline) <= 4), default=None,
                        key=lambda frame: abs(frame['tick'] - deadline))
            planned_offset = None if blade is None else (np.asarray(blade['blade']) - blade['goal']).tolist()
            result.update(first_plan=first['tick'], estimate_lead=deadline - first['tick'],
                          predicted_contact=deadline, paired_sightings=sum(frame['blade'] is not None
                                                                          and frame['ball'] is not None
                                                                          for frame in window),
                          target_offset=planned_offset, requested_lateral=float(nearest['goal'][0] - ready[0]),
                          saturated_commands=sum(bool(np.max(np.abs(row['actions'][frame['tick']])) >= .98)
                                                 for frame in window if frame['tick'] < len(row['actions'])))
        results.append(result)
    return results


def metrics(attempts):
    armed = [row for row in attempts if row['planned_frames']]
    observed = [row for row in armed if row['target_offset'] is not None]
    homes = [row['home_offset'] for row in attempts if row['home_offset'] is not None]
    offsets = [row['target_offset'] for row in observed]
    return dict(opportunities=len(attempts), contacts=sum(row['own_contact'] is not None for row in attempts),
                landed=sum(row['own_landed'] for row in attempts), planned=len(armed), measured_blade=len(observed),
                median_plan_lead_frames=median([row['estimate_lead'] for row in armed]),
                median_paired_sightings=median([row['paired_sightings'] for row in armed]),
                median_abs_home_offset_m=np.median(np.abs(homes), axis=0).tolist() if homes else None,
                median_abs_goal_offset_m=np.median(np.abs(offsets), axis=0).tolist() if offsets else None,
                median_abs_lateral_reach_m=median([abs(row['requested_lateral']) for row in armed]),
                median_saturated_commands=median([row['saturated_commands'] for row in armed]))


def first_stroke_changes(baseline, alternative):
    def measurements(receipt):
        blade, departure = {}, {}
        for row in receipt['points']:
            if row['server'] != -1 or row['optical']['contact_prediction'] is None:
                continue
            motion = _blade_motion(row['optical_trace'], row['optical']['contact_prediction'])
            if motion is not None:
                blade[row['seed']] = np.array([motion[0], *motion[1]])
            fit = row['optical']['departure']
            if fit is not None:
                departure[row['seed']] = np.asarray(fit['velocity_frame'])
        return blade, departure

    changes = {}
    for name, a, b in zip(('blade', 'departure'), measurements(baseline), measurements(alternative), strict=True):
        paired = [b[seed] - a[seed] for seed in a.keys() & b.keys()]
        changes[name] = dict(pairs=len(paired), median_delta=np.median(paired, axis=0).tolist() if paired else None)
    return changes


def matched_later(before, after):
    def key(row):
        return tuple(row[name] for name in ('fixture', 'seed', 'server', 'rival_strike', 'legal_bounce'))

    baseline = {key(row): row for row in before}
    comparisons = []
    for row in after:
        match = baseline.get(key(row))
        if match is not None and np.allclose(row['incoming'], match['incoming'], atol=1e-7, rtol=0):
            comparisons.append((match, row))
    return dict(baseline=metrics([a for a, _ in comparisons]), alternative=metrics([b for _, b in comparisons]),
                seeds=[a['seed'] for a, _ in comparisons])


def audit(folder, modes, phases, label=None):
    model = _read(START)['rotary_return']['service_model']
    ready = load(START)[0].models[model].ready
    sections = {}
    before = {phase: _read(path(folder, f'baseline-{phase}-{label}' if label else f'baseline-{phase}'))
              for phase in phases}
    for mode in modes:
        attempts = []
        neutral_commands = 0
        points_with_neutral_commands = 0
        first_strokes = {}
        for phase in phases:
            receipt = _read(path(folder, f'{mode}-{phase}-{label}' if label else f'{mode}-{phase}'))
            first_strokes[phase] = first_stroke_changes(before[phase], receipt)
            for row in receipt['points']:
                attempts.extend(dict(attempt, fixture=phase) for attempt in later_returns(row, ready))
                neutral = sum('neutral_request' in frame for frame in row['optical_trace'])
                neutral_commands += neutral
                points_with_neutral_commands += neutral > 0
        sections[mode] = dict(metrics=metrics(attempts), attempts=attempts, neutral_commands=neutral_commands,
                              points_with_neutral_commands=points_with_neutral_commands, first_strokes=first_strokes)
    fit = _read(path(folder, 'flight-fit'))
    training = {row['optical']['contact_prediction']: row for row in _read(TRAIN)['points']}
    labels = [dict(seed=training[sample['tick']]['seed'], physical_outcome=outcome(training[sample['tick']]),
                   physical_reason=training[sample['tick']]['evaluator']['reason'],
                   short_by_optics=sample['below_clearance']) for sample in fit['samples']]
    output = f'diagnostics-{label}' if label else 'diagnostics'
    baseline = sections['baseline']['attempts']
    matched = {mode: matched_later(baseline, row['attempts']) for mode, row in sections.items() if mode != 'baseline'}
    _write(path(folder, output), dict(training_cross_check=labels, later=sections, matched_later=matched))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--modes', nargs='+', default=MODES)
    parser.add_argument('--phases', nargs='+', default=('held', 'game-1', 'game-2'))
    parser.add_argument('--label')
    cli = parser.parse_args()
    audit(cli.folder, cli.modes, cli.phases, cli.label)

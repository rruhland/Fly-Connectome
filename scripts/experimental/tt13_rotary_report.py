"""Summarize matched rotary returns and compare optical evidence with the referee."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tt4_transfer_trial import ROOT, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt12_body_trial import own_service_prefix
from .tt13_rotary_trial import BRANCHES, path


def median(values):
    return float(np.median(values)) if len(values) else None


def outcome(row):
    if not row['evaluator']['legal_serve']:
        return 'no_opportunity'
    if row['player_returns']:
        return 'landed'
    if row['contacts']:
        return 'contact_only'
    return 'miss'


def receiving(rows):
    groups = {}
    for category in ('landed', 'contact_only', 'miss'):
        points = [row for row in rows if outcome(row) == category]
        observations = [row['optical']['closest'] for row in points if row['optical']['closest'] is not None]
        gaps = np.asarray([optical['offset'] for optical in observations])
        innovations = []
        for row in points:
            nearest = row['optical']['closest']
            if nearest is not None:
                model = next(frame['actor'] for frame in row['optical_trace'] if frame['tick'] == nearest['tick'])
                innovations.append(np.linalg.norm(np.asarray(model) - nearest['blade_pose']))
        departures = [row['optical']['departure']['velocity_frame']
                      for row in points if row['optical']['departure'] is not None]
        groups[category] = dict(count=len(points), wins=sum(row['points_for'] for row in points),
                                reasons=dict(Counter(row['evaluator']['reason'] for row in points)),
                                planned=sum(row['optical']['contact_prediction'] is not None for row in points),
                                paired=len(observations), median_gap_m=median([c['distance_m'] for c in observations]),
                                median_offset_m=np.median(gaps, axis=0).tolist() if len(gaps) else None,
                                median_estimated_blade_error_m=median(innovations), optical_departures=len(departures),
                                median_exit_m_per_frame=np.median(departures, axis=0).tolist() if departures else None)
    return groups


def incoming_events(row):
    events = row['events']
    legal = next((i for i, event in enumerate(events) if event['kind'] == 'legal_serve'), None)
    return events[:legal + 1] if legal is not None else []


def same_incoming_events(first, second):
    if not first or len(first) != len(second):
        return False
    continuous = {'ball', 'velocity', 'spin', 'location', 'incoming', 'time'}
    for a, b in zip(first, second, strict=True):
        if a.keys() != b.keys():
            return False
        for key in a:
            if key in continuous and not np.allclose(a[key], b[key], atol=1e-8, rtol=0):
                return False
            if key not in continuous and a[key] != b[key]:
                return False
    return True


def paired_returns(first, second):
    rows = [{r['seed']: r for r in receipt['points'] if r['server'] == -1} for receipt in (first, second)]
    shared = sorted(rows[0].keys() & rows[1].keys())
    transitions = Counter()
    same_incoming = 0
    for seed in shared:
        a, b = (row[seed] for row in rows)
        if a['phase'] != b['phase']:
            raise ValueError('matched receive schedules differ')
        transitions[f'{outcome(a)} → {outcome(b)}'] += 1
        same_incoming += same_incoming_events(incoming_events(a), incoming_events(b))
    serves = [{r['seed']: r for r in receipt['points'] if r['server'] == 1} for receipt in (first, second)]
    own = sorted(serves[0].keys() & serves[1].keys())
    same = sum(own_service_prefix(serves[0][seed]) == own_service_prefix(serves[1][seed]) for seed in own)
    return dict(starts=len(shared), matched_incoming_legal_serves=same_incoming,
                outcome_pairs=dict(transitions), shared_own_tosses=len(own),
                identical_service_prefixes=same)


def audit_ports(folder):
    """Inspect the evaluator's wiring only after the visual choices were saved."""
    target = _read(ROOT / 'artifacts/tt12/body-rotary-probes.json.gz')['wiring']
    probes = _read(path(folder, 'probes'))
    comparisons = []
    for side, aligned in zip(('a', 'b'), probes['alignments'], strict=True):
        archive = _read(ROOT / f'artifacts/tt9/rotary-lift-carried-{side}.json.gz')
        source = TT6Wiring(SimpleNamespace(action_size=6), archive['installation_seed']).state()
        order = [source['order'].index(axis) for axis in target['order']]
        signs = [source['signs'][index] * sign for index, sign in zip(order, target['signs'], strict=True)]
        comparisons.append(dict(donor=aligned['donor'], expected_order=order, expected_signs=signs,
                                order_correct=sum(a == b for a, b in zip(order, aligned['order'], strict=True)),
                                signs_correct=sum(a == b for a, b in zip(signs, aligned['signs'], strict=True))))
    return comparisons


def summarize(folder, branches):
    report = dict(ports=audit_ports(folder))
    for fixture in (0, 1, 2):
        label = 'held' if fixture == 0 else f'game-{fixture}'
        records = {branch: _read(path(folder, f'{branch}-{label}')) for branch in branches}
        section = {}
        report[label] = section
        section['branches'] = {branch: dict(score=record.get('score'), metrics=record['metrics'],
                                            optical=receiving([r for r in record['points'] if r['server'] == -1]),
                                            replayed_points=record['replayed_points'])
                               for branch, record in records.items()}
        comparisons = (('cold', 'recalled'), ('recalled', 'commissioned'), ('recalled', 'coherent'))
        section['pairs'] = {f'{first}_to_{second}': paired_returns(records[first], records[second])
                            for first, second in comparisons if first in records and second in records}
    _write(path(folder, 'comparison'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('branches', nargs='*', default=BRANCHES)
    args = parser.parse_args()
    summarize(args.folder, args.branches)

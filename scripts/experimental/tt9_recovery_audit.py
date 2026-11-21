"""Evaluate follow-up contacts from referee events after the controller stops."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .tt4_transfer_trial import _read


def return_opportunities(row):
    """Yield the first receiving, post-serve and later receiving opportunities."""
    server = row['server']
    events = row['events']
    if not row['evaluator']['legal_serve']:
        return
    openings = [i for i, event in enumerate(events)
                if ((event['kind'] == 'legal_serve' and event['side'] == -1)
                    or (event['kind'] == 'legal_return' and event['side'] == -1))]
    for index, start in enumerate(openings[:1] if server == 1 else openings[:2]):
        finish = openings[index + 1] if index + 1 < len(openings) else len(events)
        follow_up = events[start + 1:finish]
        hit = any(event['kind'] == 'contact' and event['side'] == 1 for event in follow_up)
        returned = any(event['kind'] == 'legal_return' and event['side'] == 1 for event in follow_up)
        kind = 'after_own_serve' if server == 1 else 'opening_receive' if index == 0 else 'later_receive'
        yield kind, hit, returned


def counts(points):
    stats = Counter()
    for row in points:
        result = row['evaluator']
        stats['played'] += 1
        stats['own_serve_attempts'] += row['server'] == 1
        stats['own_legal_serves'] += row['server'] == 1 and result['legal_serve']
        stats['own_service_wins'] += row['server'] == 1 and result['points_for'] == 1
        stats['all_legal_returns'] += result['legal_returns']
        stats['points_won'] += result['points_for']
        stats['points_lost'] += result['points_against']
        stats['sustained'] += result['legal_returns'] >= (1 if row['server'] == 1 else 2)
        stats['coast_frames'] += row.get('follow_up_coast_frames', 0)
        stats['shaped_frames'] += row.get('follow_up_shaped_frames', 0)
        for stage, hit, returned in return_opportunities(row):
            stats[stage + '_opportunities'] += 1
            stats[stage + '_hits'] += hit
            stats[stage + '_returns'] += returned
            stats[stage + '_wins'] += result['points_for']
            if stage == 'after_own_serve' and not returned:
                stats[stage + '_fault_' + result['reason']] += 1
    return dict(sorted(stats.items()))


def match_prefix(baseline, candidate):
    matched = Counter()
    left = {(row['seed'], row['phase'], row['server']): row for row in baseline}
    for trial in candidate:
        control = left.get((trial['seed'], trial['phase'], trial['server']))
        if control is None:
            continue
        matched['starts'] += 1
        matched['identical_point_actions'] += control['actions'] == trial['actions']
        if trial['server'] != 1:
            continue
        matched['own_serve_starts'] += 1
        cutoff = next((event['step'] for event in control['events']
                       if event['kind'] == 'contact' and event['side'] == -1), control['steps'])
        matched['identical_through_rival_contact_or_end'] += control['actions'][:cutoff] == trial['actions'][:cutoff]
    return dict(sorted(matched.items()))


def audit(data, selected=None):
    if 'points' in data:
        sections = {'match': data['points']}
    else:
        sections = {name: block['points'] if 'points' in block else block['rows']
                    for name, block in data['sections'].items()}
    if selected is not None:
        return {selected: counts(sections[selected])}
    report = {name: counts(points) for name, points in sections.items()}
    if 'fixed' in sections:
        report['matched_prefix'] = {name: match_prefix(sections['fixed'], points)
                                    for name, points in sections.items() if name != 'fixed'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', nargs='+', type=Path)
    parser.add_argument('--section')
    args = parser.parse_args()
    print(json.dumps({path.name: audit(_read(path), args.section) for path in args.receipts}, indent=2))

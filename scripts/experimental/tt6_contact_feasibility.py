"""Bound the attainable softened rotary return with truth-timed, legal-rate tape edits."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_racket import set_racket_response


DEFAULT_INPUT = ROOT / 'artifacts/tt6/contact-blocked.json.gz'
HELD_INPUT = ROOT / 'artifacts/tt6/contact-high-loft.json.gz'


def candidate_edits():
    """Prefer short, small changes in ordinary physical shoulder/elbow/wrist rates."""
    pitch = [(0., 0., value) for value in (-.65, -.35, -.18, .18, .35, .65, .95)]
    drive = [(s, e, w) for (s, e), w in product(((.18, -.34), (.36, -.58), (.60, -.88), (.42, -.29)),
                                                 (-.28, 0., .28, .55))]
    edits = [dict(lead=lead, duration=duration, delta=list(delta))
             for delta, lead, duration in product((*pitch, *drive), (1, 2, 3, 4, 5, 6), (1, 2, 3))]
    return sorted(edits, key=lambda edit: edit['duration'] * np.linalg.norm(edit['delta']))


def replay(arena, wiring, row, *, impact=None, edit=None, stop_on_return=True):
    wiring.reset(seed=row['seed'], render=False)
    actions = np.asarray(row['actions'], dtype=float)
    requested = []
    incoming_crossing = None
    if edit is not None:
        physical = np.zeros(6)
        physical[1:4] = edit['delta']
        offset = wiring.signs * physical[wiring.order]
        first, stop = impact - edit['lead'], impact - edit['lead'] + edit['duration']
    else:
        first = stop = 0
    for index in range(arena.max_steps):
        action = actions[index] if index < len(actions) else np.zeros(6)
        if first <= index < stop:
            action = np.clip(action + offset, -1., 1.)
        requested.append(action.tolist())
        was_above = arena.ball[1] >= -1.39
        _, _, done, truncated, _ = wiring.step(action, render=False)
        if incoming_crossing is None and was_above and arena.ball[1] < -1.39 and arena.velocity[1] < 0.:
            incoming_crossing = arena.steps
        if (stop_on_return and arena.legal_returns) or done or truncated:
            break
    contact = next((event for event in arena.events if event['kind'] == 'contact' and event['side'] == 1), None)
    landing = next((event for event in arena.events if event['kind'] == 'legal_return' and event['side'] == 1), None)
    outgoing = next((event for event in arena.events
                     if event['kind'] == 'separation' and event['collider'] == 'learner_racket'), None)
    return dict(contact=contact, landing=landing, outgoing=outgoing, incoming_crossing=incoming_crossing,
                evaluator=arena.point_result(), actions=requested)


def _work(job):
    geometry, station, row, nominal = job
    arena = TT6Arena(station['arena_seed'], body='rotary', view_seed=station['view_seed'], **geometry)
    wiring = TT6Wiring(arena, station['arena_seed'])
    try:
        set_racket_response(arena, 'medium')
        baseline = replay(arena, wiring, row, stop_on_return=False)
        if baseline['evaluator'] != row['evaluator']:
            raise ValueError(f"recorded point did not reproduce for seed {row['seed']}")
        contact = baseline['contact']
        impact = contact['step'] if contact is not None else baseline['incoming_crossing']
        if impact is None:
            raise ValueError(f"could not locate the observed stroke for seed {row['seed']}")
        edits = candidate_edits() if nominal is None else [nominal]
        result = dict(seed=row['seed'], baseline=baseline['evaluator'], impact_step=impact,
                      impact_source='baseline_contact' if contact is not None else 'incoming_plane_crossing',
                      anchor_event=contact, edit=None, witness=None, tested=0, source_action_sha256=row['action_sha256'])
        if baseline['landing'] is not None:
            return result
        for edit in edits:
            witness = replay(arena, wiring, row, impact=impact, edit=edit)
            result['tested'] += 1
            if witness['landing'] is not None:
                result.update(edit=edit, witness=witness)
                return result
        return result
    finally:
        arena.close()


def _jobs(path, condition, edit=None):
    doc = _read(path)
    section = doc['runs'][condition]
    rows = section['points'] if isinstance(section, dict) else section
    return [(doc['geometry'], doc['commissioning']['installation'], row, edit)
            for row in rows if row['phase'] == 'soft']


def run(output, training, held, workers):
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='tt6-soft-rotary-contact-feasibility-v1', evaluator_only=True,
                  training_sha256=_hash(training), held_sha256=_hash(held), candidate_edits=len(candidate_edits()),
                  search=[], locked=[], held_search=[], implementation_sha256=_hash(Path(__file__)))
    training_jobs = _jobs(training, 'fixed')
    held_jobs = _jobs(held, 'baseline')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(_work, training_jobs):
            report['search'].append(row)
            _write(output, report)
            print('search', row['seed'], row['tested'], row['edit'], flush=True)
        options = [row['edit'] for row in report['search'] if row['edit'] is not None]
        if options:
            # The modal successful training edit is frozen before either held-out run.
            counts = Counter((e['lead'], e['duration'], tuple(e['delta'])) for e in options)
            lead, duration, delta = counts.most_common(1)[0][0]
            edit = dict(lead=lead, duration=duration, delta=list(delta))
            report['locked_edit'] = edit
            for row in executor.map(_work, _jobs(held, 'baseline', edit)):
                report['locked'].append(row)
                _write(output, report)
                print('locked', row['seed'], row['tested'], row['edit'], flush=True)
        for row in executor.map(_work, held_jobs):
            report['held_search'].append(row)
            _write(output, report)
            print('held search', row['seed'], row['tested'], row['edit'], flush=True)
    report['summary'] = {group: dict(points=len(report[group]), successful_edits=sum(
        row['edit'] is not None for row in report[group]), baseline_returns=sum(
            row['baseline']['legal_returns'] > 0 for row in report[group])) for group in ('search', 'locked', 'held_search')}
    _write(output, report)
    print(report['summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--training', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--held', type=Path, default=HELD_INPUT)
    parser.add_argument('--workers', type=int, default=2)
    arguments = parser.parse_args()
    run(arguments.output, arguments.training, arguments.held, arguments.workers)

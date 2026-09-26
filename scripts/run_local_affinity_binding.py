"""Compare fixed and locally learned sensory binding in entity files."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from persistent_entity_files import PersistentEntityFiles
from run_decisive_representation_audit import ACTIVE, make_cases
from run_local_motion_assemblies import case_events
from run_persistent_entity_files import (different_shape_crossing,
                                         empty_state, noise_case, snapshot)
from run_sparse_recurrent_field import empty_forecast, rank_hits


OUT = Path('docs/experiments/2026-09-26-local-affinity-binding-results.json')
PRIOR = Path('docs/experiments/2026-09-26-persistent-entity-files-results.json')
ARMS = ('four', 'eight', 'aligned', 'shuffled')
EIGHT = {(-1., 1): True, (-1., -1): True,
         (1., 1): True, (1., -1): True}


def affinity_report(links):
    report = {name: {f'{sign:+.0f},{direction:+d}': bool(
        links[name].get((sign, direction), False))
        for sign in (-1., 1.) for direction in (1, -1)}
        for name in ('aligned', 'shuffled')}
    report.update({name: links[name] for name in ('counts', 'weights')
                   if name in links})
    return report


def make_models(links):
    return {name: PersistentEntityFiles(diagonal_links=edges)
            for name, edges in (('four', {}), ('eight', EIGHT),
                                ('aligned', links['aligned']),
                                ('shuffled', links['shuffled']))}


@torch.no_grad()
def file_counts(events, links):
    models = make_models(links)
    counts = {name: [] for name in ARMS}
    for event in events:
        for name, model in models.items():
            model.step(event)
            counts[name].append(len(model.live_slots))
    return counts


@torch.no_grad()
def evaluate_forecasts(episodes, links):
    rows = {name: dict(empty_forecast(), cases=[])
            for name in ARMS}
    for events in episodes:
        models = make_models(links)
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        for event in events:
            if bool(event.any()) and pending is not None:
                for name, forecast in pending.items():
                    for row in (rows[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((event > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, event, 8)
                        row['top_32_hits'] += rank_hits(forecast, event, 32)
            for model in models.values():
                model.step(event)
            if bool(event.any()):
                pending = {name: model.forecast()
                           for name, model in models.items()}
        for name in ARMS:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    return rows


@torch.no_grad()
def evaluate_state(cases, event_cache, links):
    result = {}
    for case, events in zip(cases, event_cache):
        split = case['split']
        rows = result.setdefault(split, {name: empty_state()
                                         for name in ARMS})
        models = make_models(links)
        identity = {name: {} for name in ARMS}
        for frame, event in enumerate(events):
            for model in models.values():
                model.step(event)
            if frame in ACTIVE:
                for name, model in models.items():
                    snapshot(rows[name], model.live_slots, case,
                             frame, identity[name])
    for group in result.values():
        for row in group.values():
            row['exact_count_fraction'] = row['exact_count']/max(
                row['frames'], 1)
            row['localized_fraction'] = row['localized']/max(
                row['objects'], 1)
            row['mean_center_error'] = row['center_error']/max(
                row['matched'], 1)
    return result


@torch.no_grad()
def audit_generic(episodes, expected_counts, links):
    result = {}
    for events, expected in zip(episodes, expected_counts):
        rows = result.setdefault(expected, {
            name: dict(active_frames=0, wrong_count=0, total_files=0,
                       zero_files=0) for name in ARMS})
        models = make_models(links)
        for event in events:
            for model in models.values():
                model.step(event)
            if not bool(event.any()):
                continue
            for name, model in models.items():
                count = len(model.live_slots)
                row = rows[name]
                row['active_frames'] += 1
                row['wrong_count'] += int(count != expected)
                row['total_files'] += count
                row['zero_files'] += int(count == 0)
    for group in result.values():
        for row in group.values():
            row['wrong_count_fraction'] = row['wrong_count']/max(
                row['active_frames'], 1)
            row['mean_files'] = row['total_files']/max(
                row['active_frames'], 1)
    return result


@torch.no_grad()
def audit_stress(cases, links):
    result = {}
    for name, (events, expected, scored_frames) in cases.items():
        counts = {arm: [] for arm in ARMS}
        models = make_models(links)
        for frame, event in enumerate(events):
            for arm, model in models.items():
                model.step(event)
                if frame in scored_frames:
                    counts[arm].append(len(model.live_slots))
        result[name] = {
            arm: dict(frames=len(values),
                      exact_count=sum(value == expected for value in values),
                      final_count=values[-1])
            for arm, values in counts.items()}
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    links = fit_diagonal_affinity(training)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    generic = [scene_events(1000+seed, heldout=True)
               for seed in range(16)]
    groups = dict(generic_heldout=generic)
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {name: evaluate_forecasts(episodes, links)
                 for name, episodes in groups.items()}
    state = evaluate_state(cases, event_cache, links)
    counts = audit_generic(generic,
                           [1+seed % 3+int(seed % 3 == 0)
                            for seed in range(1000, 1016)], links)
    stress = audit_stress({
        'static_noisy': (noise_case(), 1, range(2, 15)),
        'different_shape_crossing': (
            different_shape_crossing(), 2, range(2, 15))}, links)
    prior = json.loads(PRIOR.read_text())
    for split, arms in forecasts.items():
        assert arms['four']['top_32_hits'] == prior[
            'forecasts'][split]['aligned']['top_32_hits']
    result = dict(training_episodes=len(training),
                  links=affinity_report(links),
                  forecasts=forecasts, state=state, generic_state=counts,
                  stress=stress, prior_fast_only=prior['forecasts'],
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        links=affinity_report(links),
        forecast={split: {arm: round(row['top_32_recall'], 3)
                          for arm, row in group.items()}
                  for split, group in forecasts.items()},
        generic_state=counts,
        crossing_switches={arm: row['identity_switch_frames']
                           for arm, row in state['crossing'].items()},
        stress=stress,
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

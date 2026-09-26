"""Registered online entity-state and next-event forecast comparison."""

import json
import time
from pathlib import Path

import torch

from generic_entity_baseline import (EventOccupancy, VisualContinuity,
                                     components)
from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from persistent_entity_files import PersistentEntityFiles
from run_decisive_representation_audit import ACTIVE, make_cases
from run_local_motion_assemblies import (case_events, direction_from_history,
                                         target_center)
from run_sparse_recurrent_field import (empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)


OUT = Path('docs/experiments/2026-09-26-persistent-entity-files-results.json')
ENTITY_ARMS = ('aligned', 'shuffled', 'frozen')
FORECAST_ARMS = ('fast_only', *ENTITY_ARMS)


def make_models():
    return {name: PersistentEntityFiles(credit=name, seed=19)
            for name in ENTITY_ARMS}


@torch.no_grad()
def evaluate_forecasts(fast, episodes):
    rows = {name: empty_forecast() for name in FORECAST_ARMS}
    for row in rows.values():
        row['cases'] = []
    models = make_models()
    for events in episodes:
        fast.reset_state()
        for model in models.values():
            model.reset_state()
        local = {name: empty_forecast() for name in FORECAST_ARMS}
        pending = None
        for event in events:
            if bool(event.any()) and pending is not None:
                for name, forecast in pending.items():
                    for row in (rows[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((event > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, event, 8)
                        row['top_32_hits'] += rank_hits(forecast, event, 32)
            fast.step(event)
            for model in models.values():
                model.step(event)
            if bool(event.any()):
                pending = {'fast_only': fast.forecast()}
                pending.update({name: model.forecast()
                                for name, model in models.items()})
        for name in FORECAST_ARMS:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    return rows


def empty_state():
    return dict(frames=0, exact_count=0, excess=0, missing=0,
                identity_switch_frames=0, localized=0, matched=0,
                objects=0,
                center_error=0., direction_correct=0,
                direction_total=0)


def snapshot(row, slots, case, frame, identity):
    expected = [target_center(item, frame) for item in case['objects']]
    row['frames'] += 1
    row['objects'] += len(expected)
    row['exact_count'] += int(len(slots) == len(expected))
    row['excess'] += max(len(slots)-len(expected), 0)
    row['missing'] += max(len(expected)-len(slots), 0)
    if len(slots) >= len(expected) and not identity:
        available = set(range(len(slots)))
        for index, center in enumerate(expected):
            nearest = min(available, key=lambda j: sum(
                (slots[j]['center'][axis]-center[axis])**2
                for axis in (0, 1)))
            identity[index] = slots[nearest]['id']
            available.remove(nearest)
    for index, center in enumerate(expected):
        slot = next((slot for slot in slots
                     if slot['id'] == identity.get(index)), None)
        if slot is None:
            continue
        row['matched'] += 1
        own = sum((slot['center'][axis]-center[axis])**2
                  for axis in (0, 1))
        row['center_error'] += own**.5
        row['localized'] += int(own <= 9)
        if len(expected) == 2:
            other = expected[1-index]
            alternate = sum((slot['center'][axis]-other[axis])**2
                            for axis in (0, 1))
            if alternate < own:
                row['identity_switch_frames'] += 1


@torch.no_grad()
def evaluate_state(cases, event_cache):
    result = {split: {name: empty_state() for name in
                      (*ENTITY_ARMS, 'fixed_continuity')}
              for split in ('calibration', 'position', 'speed', 'shape',
                            'separated', 'crossing')}
    for case, events in zip(cases, event_cache):
        models = make_models()
        baseline_surface = EventOccupancy()
        baseline = VisualContinuity()
        identity = {name: {} for name in (*ENTITY_ARMS,
                                          'fixed_continuity')}
        for frame, event in enumerate(events):
            for model in models.values():
                model.step(event)
            fixed = baseline.step(components(baseline_surface.step(event)))
            if frame not in ACTIVE:
                continue
            states = {name: model.live_slots for name, model in
                      models.items()}
            states['fixed_continuity'] = [dict(id=index,
                                               center=track['center'],
                                               history=[])
                                           for index, track in enumerate(fixed)]
            for name, slots in states.items():
                snapshot(result[case['split']][name], slots, case,
                         frame, identity[name])
        for name, model in models.items():
            for index, item in enumerate(case['objects']):
                slot = next((slot for slot in model.slots
                             if slot['id'] == identity[name].get(index)), None)
                if slot is None:
                    continue
                predicted = direction_from_history(slot['history'])
                row = result[case['split']][name]
                row['direction_total'] += 1
                row['direction_correct'] += int(
                    predicted == item['direction'])
    for group in result.values():
        for row in group.values():
            row['exact_count_fraction'] = row['exact_count']/max(
                row['frames'], 1)
            row['mean_excess'] = row['excess']/max(row['frames'], 1)
            row['localized_fraction'] = row['localized']/max(
                row['objects'], 1)
            row['mean_center_error'] = row['center_error']/max(
                row['matched'], 1)
    return result


def noise_case():
    path = [None]*18
    for frame in range(2, 15):
        path[frame] = (16, 32)
    events = path_events([('square', path)])
    rng = torch.Generator().manual_seed(0)
    noisy = []
    for event in events:
        retained = event*(torch.rand(event.shape,
                                     generator=rng) >= .1)
        false = (torch.rand(event.shape,
                            generator=rng) < .001).float()
        noisy.append(torch.maximum(retained, false))
    return noisy


def different_shape_crossing():
    left = [None]*18
    right = [None]*18
    for frame in range(2, 15):
        left[frame] = (16, 26+2*(frame-8))
        right[frame] = (16, 38-2*(frame-8))
    return path_events([('square', left), ('plus', right)])


@torch.no_grad()
def stress_counts(episodes):
    result = {}
    for name, events in episodes.items():
        rows = {}
        for mode in ENTITY_ARMS:
            model = PersistentEntityFiles(credit=mode, seed=19)
            counts = []
            for frame, event in enumerate(events):
                model.step(event)
                if 2 <= frame < 15:
                    counts.append(len(model.live_slots))
            expected = 1 if name == 'static_noisy' else 2
            rows[mode] = dict(exact_count=sum(count == expected
                                              for count in counts),
                              frames=len(counts), final_count=counts[-1])
        result[name] = rows
    return result


@torch.no_grad()
def generic_count_audit():
    rows = {}
    for offset in range(16):
        seed = 1000+offset
        model = PersistentEntityFiles()
        expected = 1+seed % 3+int(seed % 3 == 0)
        row = rows.setdefault(expected, dict(active_frames=0,
                                              wrong_count=0,
                                              total_files=0,
                                              zero_files=0))
        for event in scene_events(seed, heldout=True):
            model.step(event)
            if not bool(event.any()):
                continue
            count = len(model.live_slots)
            row['active_frames'] += 1
            row['wrong_count'] += int(count != expected)
            row['total_files'] += count
            row['zero_files'] += int(count == 0)
    for row in rows.values():
        row['wrong_count_fraction'] = row['wrong_count']/row['active_frames']
        row['mean_files'] = row['total_files']/row['active_frames']
    return rows


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_dictionary(training)
    fast = fit_transitions(fast, training)['aligned']
    fast_seconds = time.perf_counter()-started
    print(f'fast predictor trained in {fast_seconds:.1f}s', flush=True)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {split: evaluate_forecasts(fast, episodes)
                 for split, episodes in groups.items()}
    state = evaluate_state(cases, event_cache)
    stress = stress_counts(dict(static_noisy=noise_case(),
                                different_shape_crossing=
                                different_shape_crossing()))
    generic_state = generic_count_audit()
    result = dict(training_episodes=len(training),
                  fast_seconds=fast_seconds, forecasts=forecasts,
                  state=state, stress=stress,
                  generic_state=generic_state,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        forecast={split: {name: round(row['top_32_recall'], 3)
                          for name, row in arms.items()}
                  for split, arms in forecasts.items()},
        state={split: {name: dict(
            count=round(row['exact_count_fraction'], 3),
            localized=round(row['localized_fraction'], 3),
            switches=row['identity_switch_frames'],
            direction=f"{row['direction_correct']}/{row['direction_total']}")
            for name, row in arms.items()}
            for split, arms in state.items()},
        stress=stress,
        generic_state=generic_state,
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

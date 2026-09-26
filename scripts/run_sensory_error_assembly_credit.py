"""Matched local future sensory-error credit experiment."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generic_native_cadence import scene_events
from run_decisive_representation_audit import (ACTIVE, fit_centroids,
                                               make_cases, object_mask,
                                               score as probe_score)
from run_local_motion_assemblies import case_events, target_center
from run_sparse_recurrent_field import (empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)
from run_spatial_state_audit import spatial_feature
from run_two_timescale_assemblies import (component_count,
                                          fit_slow_dictionary, step_fast)


OUT = Path('docs/experiments/2026-09-26-sensory-error-assembly-credit-results.json')
SLOW_ARMS = ('sensory_error', 'winner_error', 'frozen',
             'shuffled_sensory')
ARMS = ('fast_only', *SLOW_ARMS)


@torch.no_grad()
def fit_models(fast, episodes):
    base = fit_slow_dictionary(fast, episodes)
    models = {name: copy.deepcopy(base) for name in SLOW_ARMS}
    for model in models.values():
        for unit in range(model.units):
            model.transitions[unit, unit, 1, 1] = .5
    for index, events in enumerate(episodes):
        fast.reset_state()
        base.reset_state()
        active = []
        for event in events:
            winner = base.step(step_fast(fast, event))
            if bool(event.any()):
                active.append((fast.state.clone(), winner))
        if len(active) < 2:
            continue
        shuffled = [current_fast for current_fast, _ in active[1:]]
        random.Random(6000+index).shuffle(shuffled)
        for (previous_fast, previous_winner), (
                current_fast, current_winner), control_fast in zip(
                    active[:-1], active[1:], shuffled):
            baseline = fast.predict_field(previous_fast)
            models['sensory_error'].credit_sensory_residual(
                previous_winner, current_fast, baseline)
            models['winner_error'].credit_transition(
                previous_winner, current_winner)
            models['shuffled_sensory'].credit_sensory_residual(
                previous_winner, control_fast, baseline)
    return models


@torch.no_grad()
def evaluate_forecasts(fast, models, episodes):
    rows = {name: empty_forecast() for name in ARMS}
    for row in rows.values():
        row['cases'] = []
    for events in episodes:
        fast.reset_state()
        for model in models.values():
            model.reset_state()
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        for event in events:
            active = bool(event.any())
            if active and pending is not None:
                for name, forecast in pending.items():
                    for row in (rows[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((event > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, event, 8)
                        row['top_32_hits'] += rank_hits(forecast, event, 32)
            incoming = step_fast(fast, event)
            for model in models.values():
                model.step(incoming)
            if active:
                baseline = fast.predict_field()
                pending = {'fast_only': fast.forecast()}
                pending.update({name: model.residual_forecast(
                    fast.dictionary, baseline)
                    for name, model in models.items()})
        for name in ARMS:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    return rows


@torch.no_grad()
def evaluate_state(fast, models, cases, event_cache):
    rows = {name: {} for name in ARMS}
    totals = {name: {} for name in SLOW_ARMS}
    for case, events in zip(cases, event_cache):
        fast.reset_state()
        for model in models.values():
            model.reset_state()
        features = {name: [torch.zeros((fast.units if name ==
                                        'fast_only' else models[name].units)*9)
                           for _ in case['objects']]
                    for name in ARMS}
        split = case['split']
        for name in SLOW_ARMS:
            totals[name].setdefault(split, [0.]*4)
        for frame, event in enumerate(events):
            incoming = step_fast(fast, event)
            for model in models.values():
                model.step(incoming)
            if frame not in ACTIVE:
                continue
            union = torch.zeros((fast.height//4, fast.width//4),
                                dtype=torch.bool)
            for index, item in enumerate(case['objects']):
                center = target_center(item, frame)
                features['fast_only'][index] += spatial_feature(
                    fast.state, center, is_fast=True)
                for name, model in models.items():
                    features[name][index] += spatial_feature(
                        model.state, center)
                region = F.max_pool2d(
                    object_mask(item, frame).float()[None, None],
                    7, stride=4, padding=3)[0, 0].bool()
                union |= region
            for name, model in models.items():
                significant = model.state > .5
                count = component_count(significant.any(0))
                summary = totals[name][split]
                summary[0] += float(significant.sum())
                summary[1] += float(significant[:, ~union].sum())
                summary[2] += count
                summary[3] += max(count-len(case['objects']), 0)
        for name in ARMS:
            rows[name].setdefault(split, [])
            for item, feature in zip(case['objects'], features[name]):
                rows[name][split].append(dict(direction=item['direction'],
                                              polarity=case['polarity'],
                                              feature=feature))
    result = {}
    for name, groups in rows.items():
        centroids = fit_centroids(groups['calibration'], 'feature')
        result[name] = {}
        for split, samples in groups.items():
            scored = probe_score(samples, centroids, 'feature')
            if name != 'fast_only':
                activity, outside, components, excess = totals[name][split]
                count = sum(case['split'] == split for case in cases)
                scored['outside_activity_fraction'] = outside/max(
                    activity, 1.)
                scored['components_per_frame'] = components/max(
                    count*len(ACTIVE), 1)
                scored['excess_components_per_frame'] = excess/max(
                    count*len(ACTIVE), 1)
            result[name][split] = scored
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_dictionary(training)
    fast = fit_transitions(fast, training)['aligned']
    fast_seconds = time.perf_counter()-started
    print(f'fast predictor trained in {fast_seconds:.1f}s', flush=True)
    models = fit_models(fast, training)
    training_seconds = time.perf_counter()-started
    print(f'slow credit arms trained in {training_seconds:.1f}s total',
          flush=True)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {split: evaluate_forecasts(fast, models, episodes)
                 for split, episodes in groups.items()}
    probe = evaluate_state(fast, models, cases, event_cache)
    result = dict(training_episodes=len(training),
                  fast_seconds=fast_seconds,
                  training_seconds=training_seconds,
                  transition_mass={name: float(model.transitions.sum())
                                   for name, model in models.items()},
                  forecasts=forecasts, probe=probe,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        forecast={split: {name: round(row['top_32_recall'], 3)
                          for name, row in arms.items()}
                  for split, arms in forecasts.items()},
        probe={name: {split: f"{row['correct']}/{row['total']}"
                      for split, row in groups.items()}
               for name, groups in probe.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

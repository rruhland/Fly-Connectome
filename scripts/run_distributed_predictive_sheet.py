"""Registered distributed recurrent predictive-sheet comparison."""

import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from distributed_predictive_sheet import DistributedPredictiveSheet
from generic_native_cadence import scene_events
from run_decisive_representation_audit import (ACTIVE, fit_centroids,
                                               make_cases, object_mask,
                                               score as probe_score)
from run_local_motion_assemblies import case_events, target_center
from run_predictive_pair_assemblies import evaluate_forecasts
from run_sparse_recurrent_field import fit_dictionary, fit_transitions
from run_spatial_state_audit import spatial_feature
from run_two_timescale_assemblies import component_count, step_fast


OUT = Path('docs/experiments/2026-09-26-distributed-predictive-sheet-results.json')
SHEET_ARMS = ('aligned_sheet', 'shuffled_sheet', 'frozen_sheet',
              'zero_recurrence')


def occupied_sites(state):
    return state.sum(0) > .5


@torch.no_grad()
def fit_sheet(fast, episodes):
    sheet = DistributedPredictiveSheet(fast_units=fast.units)
    for events in episodes:
        fast.reset_state()
        active = [step_fast(fast, event).clone() for event in events
                  if bool(event.any())]
        for index in range(1, len(active)-1):
            observed = torch.cat((active[index], active[index-1]))
            future_error = (active[index+1]-
                            fast.predict_field(active[index]))
            activity = sheet.evidence(observed, future=future_error)
            sheet.credit_pair(observed, future_error, activity)
    return sheet


@torch.no_grad()
def fit_recurrence(fast, base, episodes):
    models = {name: copy.deepcopy(base) for name in SHEET_ARMS}
    models['zero_recurrence'].transitions.zero_()
    for index, events in enumerate(episodes):
        fast.reset_state()
        active = [step_fast(fast, event).clone() for event in events
                  if bool(event.any())]
        states = [base.evidence(torch.cat((active[i], active[i-1])))
                  for i in range(1, len(active))]
        targets = states[1:].copy()
        random.Random(13000+index).shuffle(targets)
        for previous, current, shuffled in zip(states[:-1],
                                                states[1:], targets):
            models['aligned_sheet'].credit_transition(previous, current)
            models['shuffled_sheet'].credit_transition(previous, shuffled)
    return models


@torch.no_grad()
def evaluate_state(fast, models, cases, event_cache):
    arms = ('fast_only', *models)
    rows = {name: {} for name in arms}
    totals = {name: {} for name in models}
    for case, events in zip(cases, event_cache):
        fast.reset_state()
        for model in models.values():
            model.reset_state()
        features = {name: [torch.zeros((fast.units if name ==
                                        'fast_only' else models[name].units)*9)
                           for _ in case['objects']]
                    for name in arms}
        split = case['split']
        for name in models:
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
                occupied = occupied_sites(model.state)
                count = component_count(occupied)
                summary = totals[name][split]
                summary[0] += float(model.state.sum())
                summary[1] += float(model.state[:, ~union].sum())
                summary[2] += count
                summary[3] += max(count-len(case['objects']), 0)
        for name in arms:
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
                scored['activity_per_frame'] = activity/max(
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
    base = fit_sheet(fast, training)
    sensory_seconds = time.perf_counter()-started
    print(f'distributed templates trained in {sensory_seconds:.1f}s total',
          flush=True)
    models = fit_recurrence(fast, base, training)
    training_seconds = time.perf_counter()-started
    print(f'recurrent arms trained in {training_seconds:.1f}s total',
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
                  sensory_seconds=sensory_seconds,
                  training_seconds=training_seconds,
                  input_mass=float(base.input_templates.sum()),
                  output_abs_mass=float(base.output_templates.abs().sum()),
                  recurrence_mass={name: float(model.transitions.sum())
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
        coherence={name: {split: dict(
            outside=round(row['outside_activity_fraction'], 3),
            components=round(row['components_per_frame'], 3))
                          for split, row in groups.items()}
                   for name, groups in probe.items()
                   if name != 'fast_only'},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

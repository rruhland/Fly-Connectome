"""Registered comparison of slow assemblies over local event predictors."""

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
from run_local_motion_assemblies import case_events
from run_sparse_recurrent_field import (case_features as fast_case_features,
                                        empty_forecast, fit_dictionary,
                                        fit_transitions, rank_hits)
from slow_predictive_assemblies import SlowPredictiveAssemblies


OUT = Path('docs/experiments/2026-09-26-two-timescale-assembly-results.json')
SLOW_ARMS = ('aligned', 'frozen', 'shuffled')
ARMS = ('fast_only', *SLOW_ARMS)


def component_count(active):
    remaining = set(map(tuple, active.nonzero(as_tuple=False).tolist()))
    groups = 0
    while remaining:
        groups += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y-1, x), (y+1, x), (y, x-1),
                             (y, x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return groups


@torch.no_grad()
def step_fast(fast, event):
    fast.step(event)
    return fast.state if bool(event.any()) else torch.zeros_like(fast.state)


@torch.no_grad()
def fit_slow_dictionary(fast, episodes):
    slow = SlowPredictiveAssemblies(fast_units=fast.units)
    for events in episodes:
        fast.reset_state()
        slow.reset_state()
        for event in events:
            slow.step(step_fast(fast, event), learn_sensory=True)
    return slow


@torch.no_grad()
def fit_slow_transitions(fast, base, episodes):
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
                active.append(winner)
        if len(active) < 2:
            continue
        shuffled = active[1:].copy()
        random.Random(5000+index).shuffle(shuffled)
        for previous, current, control in zip(active[:-1],
                                              active[1:], shuffled):
            models['aligned'].credit_transition(previous, current)
            models['shuffled'].credit_transition(previous, control)
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
            fast_input = step_fast(fast, event)
            for model in models.values():
                model.step(fast_input)
            if active:
                pending = {'fast_only': fast.forecast()}
                pending.update({name: model.forecast(fast.dictionary)
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
def slow_case_features(fast, slow, case, events):
    fast.reset_state()
    slow.reset_state()
    features = [torch.zeros(slow.units) for _ in case['objects']]
    activity = outside = components = excess = 0.
    for frame, event in enumerate(events):
        slow.step(step_fast(fast, event))
        if frame not in ACTIVE:
            continue
        union = torch.zeros((slow.height//4, slow.width//4),
                            dtype=torch.bool)
        for index, item in enumerate(case['objects']):
            mask = object_mask(item, frame)
            neighborhood = F.max_pool2d(mask.float()[None, None], 7,
                                         stride=4, padding=3)[0, 0].bool()
            features[index] += slow.state[:, neighborhood].sum(1)
            union |= neighborhood
        significant = slow.state > .5
        activity += float(significant.sum())
        outside += float(significant[:, ~union].sum())
        count = component_count(significant.any(0))
        components += count
        excess += max(count-len(case['objects']), 0)
    return features, activity, outside, components, excess


@torch.no_grad()
def evaluate_probe(fast, models, cases, event_cache):
    rows = {name: {} for name in ARMS}
    totals = {name: {} for name in ARMS}
    for case, events in zip(cases, event_cache):
        split = case['split']
        for name in ARMS:
            if name == 'fast_only':
                features, activity, outside = fast_case_features(
                    fast, case, events)
                components = excess = 0.
            else:
                features, activity, outside, components, excess = (
                    slow_case_features(fast, models[name], case, events))
            rows[name].setdefault(split, [])
            totals[name].setdefault(split, [0.]*4)
            for index, value in enumerate((activity, outside,
                                           components, excess)):
                totals[name][split][index] += value
            for item, vector in zip(case['objects'], features):
                rows[name][split].append(dict(direction=item['direction'],
                                              polarity=case['polarity'],
                                              feature=vector))
    output = {}
    for name, groups in rows.items():
        centroids = fit_centroids(groups['calibration'], 'feature')
        output[name] = {}
        for split, samples in groups.items():
            scored = probe_score(samples, centroids, 'feature')
            activity, outside, components, excess = totals[name][split]
            cases_in_split = sum(case['split'] == split for case in cases)
            scored['outside_activity_fraction'] = outside/max(activity, 1.)
            scored['activity_per_case'] = activity/max(cases_in_split, 1)
            if name != 'fast_only':
                scored['components_per_frame'] = components/max(
                    cases_in_split*len(ACTIVE), 1)
                scored['excess_components_per_frame'] = excess/max(
                    cases_in_split*len(ACTIVE), 1)
            output[name][split] = scored
    return output


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_dictionary(training)
    fast = fit_transitions(fast, training)['aligned']
    fast_seconds = time.perf_counter()-started
    print(f'fast predictor trained in {fast_seconds:.1f}s', flush=True)
    slow = fit_slow_dictionary(fast, training)
    sensory_seconds = time.perf_counter()-started
    print(f'slow motifs trained in {sensory_seconds:.1f}s total',
          flush=True)
    models = fit_slow_transitions(fast, slow, training)
    training_seconds = time.perf_counter()-started
    print(f'slow transitions trained in {training_seconds:.1f}s total',
          flush=True)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {split: evaluate_forecasts(fast, models, episodes)
                 for split, episodes in groups.items()}
    probe = evaluate_probe(fast, models, cases, event_cache)
    result = dict(training_episodes=len(training),
                  fast_seconds=fast_seconds,
                  sensory_seconds=sensory_seconds,
                  training_seconds=training_seconds,
                  slow_dictionary_mass=float(slow.dictionary.sum()),
                  aligned_transition_mass=float(
                      models['aligned'].transitions.sum()),
                  shuffled_transition_mass=float(
                      models['shuffled'].transitions.sum()),
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

"""Registered one-shot comparison of a locally plastic visual field."""

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
from sparse_recurrent_field import SparseRecurrentField


OUT = Path('docs/experiments/2026-09-26-sparse-recurrent-field-results.json')
ARMS = ('aligned', 'frozen', 'shuffled', 'repeat_event')


def rank_hits(forecast, target, count):
    values = forecast.flatten()
    active = int((values > 0).sum())
    if not active:
        return 0
    selected = values.topk(min(count, active)).indices
    return int((target.flatten()[selected] > 0).sum())


@torch.no_grad()
def fit_dictionary(episodes):
    field = SparseRecurrentField()
    for events in episodes:
        field.reset_state()
        for event in events:
            field.step(event, learn_sensory=True)
    return field


@torch.no_grad()
def fit_transitions(base, episodes):
    models = {name: copy.deepcopy(base) for name in ARMS[:-1]}
    for model in models.values():
        for unit in range(model.units):
            model.transitions[unit, unit, 3, 3] = .5
    for index, events in enumerate(episodes):
        base.reset_state()
        active = [base.step(event) for event in events if bool(event.any())]
        if len(active) < 2:
            continue
        targets = active[1:].copy()
        random.Random(3000+index).shuffle(targets)
        for previous, current, shuffled in zip(active[:-1], active[1:],
                                               targets):
            models['aligned'].credit_transition(previous, current)
            models['shuffled'].credit_transition(previous, shuffled)
    return models


def empty_forecast():
    return dict(events=0, targets=0, top_8_hits=0, top_32_hits=0)


@torch.no_grad()
def evaluate_forecasts(models, episodes):
    rows = {name: empty_forecast() for name in ARMS}
    for row in rows.values():
        row['cases'] = []
    for events in episodes:
        local = {name: empty_forecast() for name in ARMS}
        for model in models.values():
            model.reset_state()
        previous = None
        pending = {}
        for event in events:
            if not bool(event.any()):
                continue
            if previous is not None:
                for name, forecast in pending.items():
                    for row in (rows[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((event > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, event, 8)
                        row['top_32_hits'] += rank_hits(forecast, event, 32)
            pending = {}
            for name, model in models.items():
                model.step(event)
                pending[name] = model.forecast()
            pending['repeat_event'] = event
            previous = event
        for name in ARMS:
            rows[name]['cases'].append(local[name])
    for row in rows.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(32*row['events'], 1)
    return rows


@torch.no_grad()
def case_features(model, case, events):
    model.reset_state()
    features = [torch.zeros(model.units) for _ in case['objects']]
    activity = 0.
    outside = 0.
    for frame, event in enumerate(events):
        state = model.step(event)
        if frame not in ACTIVE:
            continue
        union = torch.zeros((model.height, model.width), dtype=torch.bool)
        for index, item in enumerate(case['objects']):
            mask = object_mask(item, frame)
            neighborhood = F.max_pool2d(mask.float()[None, None], 5,
                                         stride=1, padding=2)[0, 0].bool()
            features[index] += state[:, neighborhood].sum(1)
            union |= neighborhood
        activity += float(state.sum())
        outside += float(state[:, ~union].sum())
    return features, activity, outside


@torch.no_grad()
def evaluate_probe(models, cases, event_cache):
    rows = {name: {} for name in models}
    totals = {name: {} for name in models}
    for case, events in zip(cases, event_cache):
        for name, model in models.items():
            features, activity, outside = case_features(model, case, events)
            split = case['split']
            rows[name].setdefault(split, [])
            totals[name].setdefault(split, [0., 0.])
            totals[name][split][0] += activity
            totals[name][split][1] += outside
            for item, vector in zip(case['objects'], features):
                rows[name][split].append(dict(direction=item['direction'],
                                              polarity=case['polarity'],
                                              feature=vector))
    result = {}
    for name, groups in rows.items():
        centroids = fit_centroids(groups['calibration'], 'feature')
        result[name] = {}
        for split, samples in groups.items():
            scored = probe_score(samples, centroids, 'feature')
            activity, outside = totals[name][split]
            scored['outside_activity_fraction'] = outside/max(activity, 1.)
            scored['activity_per_case'] = activity/max(
                sum(case['split'] == split for case in cases), 1)
            result[name][split] = scored
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    base = fit_dictionary(training)
    dictionary_seconds = time.perf_counter()-started
    print(f'sensory motifs trained in {dictionary_seconds:.1f}s',
          flush=True)
    models = fit_transitions(base, training)
    training_seconds = time.perf_counter()-started
    print(f'local recurrent transitions trained in '
          f'{training_seconds:.1f}s total', flush=True)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {split: evaluate_forecasts(models, episodes)
                 for split, episodes in groups.items()}
    probe = evaluate_probe(models, cases, event_cache)
    result = dict(training_episodes=len(training),
                  dictionary_seconds=dictionary_seconds,
                  training_seconds=training_seconds,
                  learned_dictionary_mass=float(base.dictionary.sum()),
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

"""One-shot local sensory-agreement test ahead of the distributed predictor."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from local_observation_model import LocalObservationModel
from run_absolute_refresh import contrast_frames
from run_distributed_predictive_field import fit_fields
from run_sparse_recurrent_field import empty_forecast, rank_hits
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-learned-observation-model-results.json')
ARMS = ('learned_aligned', 'learned_event_only',
        'learned_shuffled_intensity', 'shuffled_credit', 'raw_input')


def visual_changes(appearance):
    previous = torch.zeros_like(appearance[0])
    changes = []
    for current in appearance:
        delta = current-previous
        changes.append(torch.stack(((-delta).clamp(0., 1.),
                                    delta.clamp(0., 1.))))
        previous = current
    return changes


@torch.no_grad()
def fit_observers(episodes):
    models = {name: LocalObservationModel()
              for name in ('aligned', 'shuffled_credit')}
    for index, (clean, appearance) in enumerate(episodes):
        noisy = corrupt_events(clean, seed=10000+index)
        targets = visual_changes(appearance)
        shuffled = targets.copy()
        random.Random(5000+index).shuffle(shuffled)
        for frame, event in enumerate(noisy):
            for name, model in models.items():
                _, features = model.step(
                    event, absolute=appearance[frame]
                    if frame % 8 == 0 else None)
                model.credit(features, event,
                             targets[frame] if name == 'aligned'
                             else shuffled[frame])
        for model in models.values():
            model.reset_state()
    return models


@torch.no_grad()
def evaluate_streams(streams, trained_observers, trained_field):
    scores = {name: dict(empty_forecast(), cases=[],
                         target_mass=0., other_mass=0.,
                         positive_sites=0) for name in ARMS}
    state = {name: dict(active_frames=0, false_contrast_mass=0.,
                        true_raw_count=0, true_filtered_mass=0.,
                        false_raw_count=0, false_filtered_mass=0.)
             for name in ARMS}
    for clean, observed, unrelated, absolute, shuffled in streams:
        assert len(clean) == len(observed) == len(unrelated)
        assert len(clean) == len(absolute) == len(shuffled)
        fields = {name: copy.deepcopy(trained_field) for name in ARMS}
        observers = {name: copy.deepcopy(trained_observers[
            'shuffled_credit' if name == 'shuffled_credit' else 'aligned'])
            for name in ARMS if name != 'raw_input'}
        for model in fields.values():
            model.reset_state()
        for model in observers.values():
            model.reset_state()
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        for frame, (target, event) in enumerate(zip(clean, observed)):
            if bool(target.any()) and pending is not None:
                mask = target > 0
                for name, forecast in pending.items():
                    for row in (scores[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int(mask.sum())
                        row['top_8_hits'] += rank_hits(forecast, target, 8)
                        row['top_32_hits'] += rank_hits(forecast, target, 32)
                    scores[name]['target_mass'] += float(forecast[mask].sum())
                    scores[name]['other_mass'] += float(forecast[~mask].sum())
                    scores[name]['positive_sites'] += int((forecast > 0).sum())
            for name, field in fields.items():
                intensity = (None if name == 'learned_event_only' or frame % 8
                             else shuffled[frame]
                             if name == 'learned_shuffled_intensity'
                             else absolute[frame])
                filtered = (event if name == 'raw_input' else
                            observers[name].step(event, absolute=intensity)[0])
                field.step(filtered, absolute=intensity)
                if bool(event.any()):
                    row = state[name]
                    row['active_frames'] += 1
                    outside = absolute[frame].abs() <= .1
                    row['false_contrast_mass'] += float(
                        field.contrast[outside].abs().sum())
                    true_raw = (event > 0) & (target > 0)
                    false_raw = (event > 0) & (target == 0)
                    row['true_raw_count'] += int(true_raw.sum())
                    row['false_raw_count'] += int(false_raw.sum())
                    row['true_filtered_mass'] += float(filtered[true_raw].sum())
                    row['false_filtered_mass'] += float(filtered[false_raw].sum())
            if bool(event.any()):
                pending = {name: field.forecast()
                           for name, field in fields.items()}
        for name in ARMS:
            scores[name]['cases'].append(local[name])
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['target_mass_per_event'] = row['target_mass']/max(row['events'], 1)
        row['other_mass_per_event'] = row['other_mass']/max(row['events'], 1)
        row['positive_sites_per_event'] = row['positive_sites']/max(
            row['events'], 1)
    for row in state.values():
        row['mean_false_contrast_mass'] = row['false_contrast_mass']/max(
            row['active_frames'], 1)
        row['true_event_credibility'] = row['true_filtered_mass']/max(
            row['true_raw_count'], 1)
        row['false_event_credibility'] = row['false_filtered_mass']/max(
            row['false_raw_count'], 1)
    return scores, state


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    field = fit_fields([events for events, _ in training])['aligned']
    observers = fit_observers(training)
    training_seconds = time.perf_counter()-started
    seeds = (2000, 2001, 2002, 2003)
    clean_pairs = [scene_events(seed, heldout=True,
                                return_contrast=True) for seed in seeds]
    clean = [events for events, _ in clean_pairs]
    absolute = [images for _, images in clean_pairs]
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip(seeds, clean)]
    unrelated = clean[1:]+clean[:1]
    shuffled = absolute[1:]+absolute[:1]
    path = [None]*18
    for frame in range(2, 15):
        path[frame] = (16, 32)
    stationary = path_events([('square', path)])
    groups = dict(
        clean_generic=list(zip(clean, clean, unrelated,
                               absolute, shuffled)),
        corrupted_generic=list(zip(clean, noisy, unrelated,
                                   absolute, shuffled)),
        stationary_noisy=[(
            stationary, corrupt_events(stationary, seed=0),
            clean[0][:len(stationary)], contrast_frames(stationary),
            absolute[0][:len(stationary)])])
    scores = {}
    state = {}
    for name, streams in groups.items():
        scores[name], state[name] = evaluate_streams(
            streams, observers, field)
    result = dict(training_episodes=len(training), seeds=seeds,
                  training_seconds=training_seconds,
                  scores=scores, state=state,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={split: {name: round(row['top_32_recall'], 3)
                        for name, row in arms.items()}
                for split, arms in scores.items()},
        top_8={split: {name: round(row['top_8_recall'], 3)
                       for name, row in arms.items()}
               for split, arms in scores.items()},
        state={split: {name: dict(
            outside=round(row['mean_false_contrast_mass'], 2),
            true_cred=round(row['true_event_credibility'], 3),
            false_cred=round(row['false_event_credibility'], 3))
            for name, row in arms.items()} for split, arms in state.items()},
        training_seconds=round(training_seconds, 1),
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

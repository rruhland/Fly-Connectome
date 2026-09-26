"""One-shot opt-in comparison of a learned continuous predictive visual field."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from distributed_predictive_field import DistributedPredictiveField
from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from run_absolute_refresh import contrast_frames
from run_sparse_recurrent_field import empty_forecast, rank_hits
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-distributed-predictive-field-results.json')
ARMS = ('aligned', 'event_only', 'shuffled_intensity',
        'shuffled_credit')


@torch.no_grad()
def fit_fields(episodes):
    models = {name: DistributedPredictiveField()
              for name in ('aligned', 'shuffled_credit')}
    for index, events in enumerate(episodes):
        absolute = contrast_frames(events)
        targets = [event for event in events if bool(event.any())][1:]
        shuffled = targets.copy()
        random.Random(3000+index).shuffle(shuffled)
        previous = {name: None for name in models}
        target_index = 0
        for frame, event in enumerate(events):
            if bool(event.any()) and target_index < len(targets):
                for name, model in models.items():
                    if previous[name] is not None:
                        credited = (event if name == 'aligned'
                                    else shuffled[target_index])
                        model.credit(previous[name], credited)
                target_index += int(any(row is not None
                                        for row in previous.values()))
            for name, model in models.items():
                state = model.step(
                    event, absolute=absolute[frame]
                    if frame % 8 == 0 else None)
                if bool(event.any()):
                    previous[name] = state
        for model in models.values():
            model.reset_state()
    return models


@torch.no_grad()
def evaluate_streams(streams, trained):
    scores = {name: dict(empty_forecast(), cases=[],
                         target_mass=0., other_mass=0.) for name in ARMS}
    state = {name: dict(active_frames=0, state_occupancy=0,
                        false_contrast_mass=0.) for name in ARMS}
    for clean, observed, unrelated in streams:
        assert len(clean) == len(observed) == len(unrelated)
        absolute = contrast_frames(clean)
        shuffled = contrast_frames(unrelated)
        models = {name: copy.deepcopy(trained[
            'shuffled_credit' if name == 'shuffled_credit' else 'aligned'])
            for name in ARMS}
        for model in models.values():
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
            for name, model in models.items():
                intensity = (None if name == 'event_only' or frame % 8
                             else shuffled[frame] if name == 'shuffled_intensity'
                             else absolute[frame])
                model.step(event, absolute=intensity)
                if bool(event.any()):
                    state[name]['active_frames'] += 1
                    state[name]['state_occupancy'] += int(
                        (model.state.abs() > .1).sum())
                    outside = absolute[frame].abs() <= .1
                    state[name]['false_contrast_mass'] += float(
                        model.contrast[outside].abs().sum())
            if bool(event.any()):
                pending = {name: model.forecast()
                           for name, model in models.items()}
        for name in ARMS:
            scores[name]['cases'].append(local[name])
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['target_mass_per_event'] = row['target_mass']/max(row['events'], 1)
        row['other_mass_per_event'] = row['other_mass']/max(row['events'], 1)
    for row in state.values():
        row['mean_state_occupancy'] = row['state_occupancy']/max(
            row['active_frames'], 1)
        row['mean_false_contrast_mass'] = row['false_contrast_mass']/max(
            row['active_frames'], 1)
    return scores, state


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    trained = fit_fields(training)
    training_seconds = time.perf_counter()-started
    seeds = (2000, 2001, 2002, 2003)
    clean = [scene_events(seed, heldout=True) for seed in seeds]
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip(seeds, clean)]
    unrelated = clean[1:]+clean[:1]
    path = [None]*18
    for frame in range(2, 15):
        path[frame] = (16, 32)
    stationary = path_events([('square', path)])
    groups = dict(
        clean_generic=list(zip(clean, clean, unrelated)),
        corrupted_generic=list(zip(clean, noisy, unrelated)),
        stationary_noisy=[(
            stationary, corrupt_events(stationary, seed=0),
            clean[0][:len(stationary)])])
    scores = {}
    state = {}
    for name, streams in groups.items():
        scores[name], state[name] = evaluate_streams(streams, trained)
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
            occupancy=round(row['mean_state_occupancy'], 1))
            for name, row in arms.items()} for split, arms in state.items()},
        training_seconds=round(training_seconds, 1),
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

"""Registered opt-in test of soft visual state against file/refresh controls."""

import json
import time
from pathlib import Path

import torch

from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from graded_visual_state import GradedVisualState
from local_affinity_binding import fit_diagonal_affinity
from run_absolute_refresh import contrast_frames
from run_sparse_recurrent_field import empty_forecast, rank_hits
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-graded-visual-state-results.json')
PRIOR = Path('docs/experiments/2026-09-26-absolute-refresh-results.json')
ARMS = ('graded_aligned', 'graded_event_only', 'graded_shuffled')


@torch.no_grad()
def evaluate_streams(streams, *, links, expected):
    scores = {name: dict(empty_forecast(), cases=[]) for name in ARMS}
    activity = {name: dict(active_frames=0, total_live=0,
                           max_live_files=0, wrong_count=0,
                           total_births=0, total_confirmed_births=0)
                for name in ARMS}
    for (clean, observed, unrelated), true_count in zip(streams, expected):
        absolute = contrast_frames(clean)
        shuffled = contrast_frames(unrelated)
        assert len(clean) == len(observed) == len(unrelated)
        models = {name: GradedVisualState(
            height=clean[0].shape[-2], width=clean[0].shape[-1],
            diagonal_links=links) for name in ARMS}
        confirmed = {name: set() for name in ARMS}
        local = {name: empty_forecast() for name in ARMS}
        pending = None
        for frame, (target, event) in enumerate(zip(clean, observed)):
            if bool(target.any()) and pending is not None:
                for name, forecast in pending.items():
                    for row in (scores[name], local[name]):
                        row['events'] += 1
                        row['targets'] += int((target > 0).sum())
                        row['top_8_hits'] += rank_hits(forecast, target, 8)
                        row['top_32_hits'] += rank_hits(forecast, target, 32)
            models['graded_aligned'].step(
                event, absolute=absolute[frame] if frame % 8 == 0 else None)
            models['graded_event_only'].step(event)
            models['graded_shuffled'].step(
                event, absolute=shuffled[frame] if frame % 8 == 0 else None)
            if bool(event.any()):
                for name, model in models.items():
                    row = activity[name]
                    live = model.live_slots
                    count = len(live)
                    confirmed[name].update(item['id'] for item in live)
                    row['active_frames'] += 1
                    row['total_live'] += count
                    row['max_live_files'] = max(row['max_live_files'], count)
                    row['wrong_count'] += int(count != true_count)
                pending = {name: model.forecast()
                           for name, model in models.items()}
        for name in ARMS:
            scores[name]['cases'].append(local[name])
            activity[name]['total_births'] += models[name].next_id
            activity[name]['total_confirmed_births'] += len(confirmed[name])
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(32*row['events'], 1)
    for row in activity.values():
        row['mean_live_files'] = row['total_live']/max(row['active_frames'], 1)
        row['wrong_count_fraction'] = row['wrong_count']/max(
            row['active_frames'], 1)
        row['mean_total_births'] = row['total_births']/max(len(streams), 1)
        row['mean_confirmed_births'] = row['total_confirmed_births']/max(
            len(streams), 1)
    return scores, activity


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    links = fit_diagonal_affinity(training)['aligned']
    seeds = (2000, 2001, 2002, 2003)
    clean = [scene_events(seed, heldout=True) for seed in seeds]
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip(seeds, clean)]
    unrelated = clean[1:]+clean[:1]
    expected = [1+seed % 3+int(seed % 3 == 0) for seed in seeds]
    path = [None]*18
    for frame in range(2, 15):
        path[frame] = (16, 32)
    stationary = path_events([('square', path)])
    groups = dict(
        clean_generic=(list(zip(clean, clean, unrelated)), expected),
        corrupted_generic=(list(zip(clean, noisy, unrelated)), expected),
        stationary_noisy=(
            [(stationary, corrupt_events(stationary, seed=0),
              clean[0][:len(stationary)])], [1]))
    results = {}
    states = {}
    for name, (streams, counts) in groups.items():
        results[name], states[name] = evaluate_streams(
            streams, links=links, expected=counts)
    assert PRIOR.is_file()
    result = dict(training_episodes=len(training), seeds=seeds,
                  results=results, state=states,
                  control_path=str(PRIOR),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        forecast={split: {name: round(row['top_32_recall'], 3)
                          for name, row in arms.items()}
                  for split, arms in results.items()},
        state={split: {name: dict(
            live=round(row['mean_live_files'], 2),
            wrong=round(row['wrong_count_fraction'], 3),
            max=row['max_live_files'],
            births=round(row['mean_total_births'], 2),
            confirmed=round(row['mean_confirmed_births'], 2))
            for name, row in arms.items()}
            for split, arms in states.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

"""Compare event-only and periodic absolute visual evidence."""

import json
import time
from pathlib import Path

import torch

from absolute_refresh_files import AbsoluteRefreshFiles
from generic_motion_stress import path_events
from generic_native_cadence import scene_events
from local_affinity_binding import fit_diagonal_affinity
from persistent_entity_files import PersistentEntityFiles
from run_sparse_recurrent_field import empty_forecast, rank_hits
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-absolute-refresh-results.json')
PRIOR = Path('docs/experiments/2026-09-26-zero-shot-visual-transfer-results.json')
ARMS = ('event_only', 'refresh_8', 'refresh_every')


@torch.no_grad()
def contrast_frames(events):
    _, height, width = events[0].shape
    contrast = torch.zeros((height, width))
    result = []
    for event in events:
        contrast.add_(event[1]-event[0]).clamp_(-1., 1.)
        result.append(contrast.clone())
    return result


@torch.no_grad()
def evaluate_streams(streams, *, links, expected):
    scores = {name: dict(empty_forecast(), cases=[]) for name in ARMS}
    activity = {name: dict(active_frames=0, total_live=0,
                           max_live_files=0, wrong_count=0,
                           total_births=0) for name in ARMS}
    for (clean, observed), true_count in zip(streams, expected):
        absolute = contrast_frames(clean)
        models = dict(event_only=PersistentEntityFiles(
            height=clean[0].shape[-2], width=clean[0].shape[-1],
            diagonal_links=links),
            refresh_8=AbsoluteRefreshFiles(
                height=clean[0].shape[-2], width=clean[0].shape[-1],
                diagonal_links=links),
            refresh_every=AbsoluteRefreshFiles(
                height=clean[0].shape[-2], width=clean[0].shape[-1],
                diagonal_links=links))
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
            models['event_only'].step(event)
            models['refresh_8'].step(
                event, absolute=absolute[frame] if frame % 8 == 0 else None)
            models['refresh_every'].step(event, absolute=absolute[frame])
            if bool(event.any()):
                for name, model in models.items():
                    row = activity[name]
                    count = len(model.live_slots)
                    row['active_frames'] += 1
                    row['total_live'] += count
                    row['max_live_files'] = max(row['max_live_files'], count)
                    row['wrong_count'] += int(count != true_count)
                pending = {name: model.forecast()
                           for name, model in models.items()}
        for name in ARMS:
            scores[name]['cases'].append(local[name])
            activity[name]['total_births'] += models[name].next_id
    for row in scores.values():
        row['top_8_recall'] = row['top_8_hits']/max(row['targets'], 1)
        row['top_32_recall'] = row['top_32_hits']/max(row['targets'], 1)
        row['top_32_precision'] = row['top_32_hits']/max(
            32*row['events'], 1)
    for row in activity.values():
        row['mean_live_files'] = row['total_live']/max(row['active_frames'], 1)
        row['wrong_count_fraction'] = row['wrong_count']/max(
            row['active_frames'], 1)
        row['mean_total_births'] = row['total_births']/max(len(streams), 1)
    return scores, activity


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    links = fit_diagonal_affinity(training)['aligned']
    clean = [scene_events(seed, heldout=True)
             for seed in (2000, 2001, 2002, 2003)]
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip((2000, 2001, 2002, 2003), clean)]
    expected = [1+seed % 3+int(seed % 3 == 0)
                for seed in (2000, 2001, 2002, 2003)]
    path = [None]*18
    for frame in range(2, 15):
        path[frame] = (16, 32)
    stationary = path_events([('square', path)])
    groups = dict(clean_generic=(list(zip(clean, clean)), expected),
                  corrupted_generic=(list(zip(clean, noisy)), expected),
                  stationary_noisy=(
                      [(stationary, corrupt_events(stationary, seed=0))],
                      [1]))
    results = {}
    states = {}
    for name, (streams, counts) in groups.items():
        results[name], states[name] = evaluate_streams(
            streams, links=links, expected=counts)
    prior = json.loads(PRIOR.read_text())
    assert results['corrupted_generic']['event_only'][
        'top_32_hits'] == prior['results']['generic_corrupted'][
            'entity_files']['top_32_hits']
    result = dict(training_episodes=len(training),
                  results=results, state=states,
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
            births=round(row['mean_total_births'], 2))
            for name, row in arms.items()}
            for split, arms in states.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

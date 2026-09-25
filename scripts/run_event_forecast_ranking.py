"""Frozen rank audit: calibration error or wrong future-event locations?"""

import json
import time
from pathlib import Path

import torch

from gap_timing_transfer import MODEL_OUT
from history_gated_code import DIRECTIONS
from run_observation_horizon_forecast import (sensory_sequence, train_heads,
                                              train_recurrent)
from run_pong_camera_transfer import SEEDS, pong_events
from run_sparse_event_transfer import dots
from run_variable_cadence_transition import training_cases


OUT = Path('docs/experiments/2026-09-25-event-forecast-ranking-results.json')


@torch.no_grad()
def rank_episode(recurrent, head, events, row):
    recurrent.reset_state()
    head.reset_state()
    previous = None
    for t, (event, code, surface) in enumerate(sensory_sequence(events)):
        if event.any() and previous is not None:
            origin, forecast = previous
            if origin >= 4:
                values = forecast.flatten()
                target = (event >= .5).flatten()
                row['events'] += 1
                row['targets'] += int(target.sum())
                for k in (2, 8, 32):
                    selected = values.topk(k).indices
                    row[f'top_{k}_hits'] += int(target[selected].sum())
                    spatial = forecast.max(0).values.flatten()
                    locations = event.any(0).flatten()
                    row[f'top_{k}_spatial_hits'] += int(locations[
                        spatial.topk(k).indices].sum())
                row['spatial_targets'] += int(event.any(0).sum())
                row['positive_score_sum'] += float(values[target].sum())
                row['negative_score_sum'] += float(values[~target].sum())
                row['negative_count'] += int((~target).sum())
                row['above_half'] += int((values >= .5).sum())
        recurrent.step((code, surface))
        issued = head.step(recurrent.state, event)
        if event.any():
            previous = (t, issued)


def empty():
    return dict(events=0, targets=0, top_2_hits=0, top_8_hits=0,
                top_32_hits=0, spatial_targets=0,
                top_2_spatial_hits=0, top_8_spatial_hits=0,
                top_32_spatial_hits=0, positive_score_sum=0.,
                negative_score_sum=0., negative_count=0, above_half=0)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    recurrent = train_recurrent(saved['models']['learned_split'].code.motion,
                                training_cases())
    training = dots(((10, 20), (14, 28), (16, 32), (18, 36),
                     (22, 44)), DIRECTIONS.values())
    head = train_heads(recurrent, training)['next_event']['learned']
    groups = dict(generic_dot=dots(((12, 24), (20, 40)),
                                   DIRECTIONS.values()),
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    result = {}
    for name, cases in groups.items():
        row = empty()
        for events in cases:
            rank_episode(recurrent, head, events, row)
        result[name] = {**row,
                        'top_2_recall': row['top_2_hits']/max(row['targets'], 1),
                        'top_8_recall': row['top_8_hits']/max(row['targets'], 1),
                        'top_32_recall': row['top_32_hits']/max(row['targets'], 1),
                        'top_8_spatial_recall': row['top_8_spatial_hits']/max(
                            row['spatial_targets'], 1),
                        'top_32_spatial_recall': row['top_32_spatial_hits']/max(
                            row['spatial_targets'], 1),
                        'mean_positive_score': row['positive_score_sum']/max(
                            row['targets'], 1),
                        'mean_negative_score': row['negative_score_sum']/max(
                            row['negative_count'], 1),
                        'above_half_per_event': row['above_half']/max(
                            row['events'], 1)}
    result['elapsed_seconds'] = time.perf_counter()-start
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {key: round(value, 4)
                             for key, value in row.items()
                             if key in ('top_2_recall', 'top_8_recall',
                                        'top_32_recall', 'top_8_spatial_recall',
                                        'top_32_spatial_recall',
                                        'mean_positive_score',
                                        'mean_negative_score',
                                        'above_half_per_event')}
                      for name, row in result.items()
                      if isinstance(row, dict)}), flush=True)


if __name__ == '__main__':
    main()

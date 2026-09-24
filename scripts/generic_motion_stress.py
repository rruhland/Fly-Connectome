"""Stress the generic local event forecast without retraining or object labels."""

import json
import time
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, event_sequence, finish, item)
from generic_motion_probe import events_map, object_mask


OUT = Path('docs/experiments/2026-09-23-generic-motion-stress-results.json')
WEIGHTS = Path('docs/experiments/2026-09-23-generic-local-competition-results.json')


def path_events(paths, *, background=True):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    sequence = []
    for frame in range(18):
        occupied = torch.zeros((32, 64), dtype=torch.bool)
        if 2 <= frame < 15:
            for shape, centers in paths:
                occupied |= object_mask(shape, centers[frame])
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, occupied] = not background
        sequence.append(events_map(camera.observe(image)))
    return sequence


def score_case(clean, predictors, *, false_event_rate=0., dropout=0., seed=0):
    rng = torch.Generator().manual_seed(seed)
    scores = {name: empty_score() for name in predictors}
    for predictor in predictors.values():
        predictor.reset_state()
    for t, current in enumerate(clean):
        observed = current.clone()
        if dropout:
            observed *= (torch.rand(observed.shape, generator=rng) >= dropout)
        if false_event_rate:
            observed = torch.maximum(observed,
                (torch.rand(observed.shape, generator=rng) < false_event_rate).float())
        for name, predictor in predictors.items():
            prediction = predictor.step(observed, learn=False)
            if 3 <= t < 14:
                accumulate(scores[name], prediction, clean[t+1])
    return {name: finish(value) for name, value in scores.items()}


def main():
    started = time.perf_counter()
    learned = LocalTripletLearner(eta=.3, local_competition=True)
    learned.weights.copy_(torch.tensor(json.loads(WEIGHTS.read_text())['weights']))
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    raw = LocalTripletLearner(eta=.3)
    raw.weights.fill_(1)
    predictors = dict(learned_competitive=learned,
                      fixed_competitive=fixed, fixed_raw=raw)

    turn = [None]*18
    speed_change = [None]*18
    crossing_a = [None]*18
    crossing_b = [None]*18
    occluder = [None]*18
    crossing_mover = [None]*18
    for t in range(2, 15):
        turn[t] = (16, 20+t-2) if t <= 8 else (16-(t-8), 26)
        speed_change[t] = (16, 20+t-2) if t <= 8 else (16, 26+2*(t-8))
        crossing_a[t] = (16, 26+2*(t-8))
        crossing_b[t] = (16, 38-2*(t-8))
        occluder[t] = (16, 32)
        crossing_mover[t] = (16, 20+2*(t-2))
    cases = dict(
        clean_translation=(event_sequence(
            [item('square', (16, 32), 'right', 1)], background=True), 0., 0.),
        direction_change=(path_events([('square', turn)]), 0., 0.),
        speed_change=(path_events([('square', speed_change)]), 0., 0.),
        crossing=(path_events([('square', crossing_a), ('square', crossing_b)]), 0., 0.),
        occupied_region=(path_events([('square', crossing_mover),
                                     ('bar', occluder)]), 0., 0.),
        event_noise_low=(event_sequence(
            [item('square', (16, 32), 'right', 1)], background=True), .001, .1),
        event_noise_high=(event_sequence(
            [item('square', (16, 32), 'right', 1)], background=True), .005, .1))
    result = {}
    noise_ranges = {}
    for name, (sequence, rate, dropout) in cases.items():
        seeds = range(5) if rate else (9,)
        runs = [score_case(sequence, predictors, false_event_rate=rate,
                           dropout=dropout, seed=seed) for seed in seeds]
        combined = {method: empty_score() for method in predictors}
        for run in runs:
            for method, row in run.items():
                for key in combined[method]:
                    combined[method][key] += row[key]
        result[name] = {method: finish(row) for method, row in combined.items()}
        if rate:
            noise_ranges[name] = {method: [min(run[method]['f1'] for run in runs),
                                           max(run[method]['f1'] for run in runs)]
                                  for method in predictors}
    OUT.write_text(json.dumps(dict(cases=result, noise_seed_f1_ranges=noise_ranges,
                                   elapsed_seconds=time.perf_counter()-started),
                              indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {method: round(row['f1'], 3)
                             for method, row in methods.items()}
                      for name, methods in result.items()}), flush=True)


if __name__ == '__main__':
    main()

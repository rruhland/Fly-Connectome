"""Frozen cross-scene and cross-noise audit of local event credibility."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from run_absolute_refresh import contrast_frames
from run_decisive_representation_audit import make_cases
from run_distributed_predictive_field import fit_fields
from run_local_motion_assemblies import case_events
from run_local_observation_model import evaluate_streams, fit_observers
from run_pong_camera_transfer import pong_events
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-observation-model-transfer-results.json')


def make_streams(episodes, observed=None):
    observed = episodes if observed is None else observed
    unrelated = episodes[1:]+episodes[:1]
    absolute = [contrast_frames(events) for events in episodes]
    shuffled = absolute[1:]+absolute[:1]
    return list(zip(episodes, observed, unrelated, absolute, shuffled))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    field = fit_fields([events for events, _ in training])['aligned']
    observers = fit_observers(training)
    training_seconds = time.perf_counter()-started

    cases = make_cases()
    groups = {}
    for split in ('position', 'speed', 'shape', 'separated', 'crossing'):
        groups[split] = make_streams([
            case_events(case) for case in cases if case['split'] == split])
    for stride in (1, 4):
        groups[f'pong_stride_{stride}'] = make_streams([
            pong_events(seed, stride=stride, frames=120)
            for seed in (1101, 1102, 1103, 1104)])
    seeds = (2000, 2001, 2002, 2003)
    clean = [scene_events(seed, heldout=True) for seed in seeds]
    for name, dropout, false_rate in (
            ('known_noise', .1, .001),
            ('lighter_noise', .05, .0005),
            ('heavier_noise', .2, .002)):
        observed = [corrupt_events(events, seed=seed,
                                   dropout=dropout,
                                   false_rate=false_rate)
                    for seed, events in zip(seeds, clean)]
        groups[name] = make_streams(clean, observed)
    scores = {}
    state = {}
    for name, streams in groups.items():
        scores[name], state[name] = evaluate_streams(
            streams, observers, field)
        print(name, dict(learned=round(
            scores[name]['learned_aligned']['top_32_recall'], 3),
            raw=round(scores[name]['raw_input']['top_32_recall'], 3),
            shuffled=round(scores[name]['shuffled_credit'][
                'top_32_recall'], 3)), flush=True)
    result = dict(training_episodes=len(training),
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
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

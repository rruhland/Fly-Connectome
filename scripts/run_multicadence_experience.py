"""Equal-budget test of repeated versus varied unlabeled motion tempo."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from run_decisive_representation_audit import make_cases
from run_distributed_predictive_field import fit_fields
from run_local_motion_assemblies import case_events
from run_local_observation_model import evaluate_streams, fit_observers
from run_observation_model_transfer import make_streams
from run_pong_camera_transfer import pong_events
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-multicadence-experience-results.json')
PRIOR = Path('docs/experiments/2026-09-26-observation-model-transfer-results.json')


def training_sequences(*, seeds=range(64), strides=(1, 2, 4, 8),
                       frames=80):
    base = [scene_events(seed, frames=frames) for seed in seeds]
    repeated = [events for events in base for _ in strides]
    diverse = [scene_events(seed, frames=frames,
                            motion_stride=stride)
               for seed in seeds for stride in strides]
    return repeated, diverse


@torch.no_grad()
def transfer_groups():
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
    groups['clean_generic'] = make_streams(clean)
    noisy = [corrupt_events(events, seed=seed)
             for seed, events in zip(seeds, clean)]
    groups['known_noise'] = make_streams(clean, noisy)
    return groups


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observers = fit_observers(original)
    repeated, diverse = training_sequences()
    training = {}
    fields = {}
    for name, episodes in (('repeated', repeated), ('diverse', diverse)):
        began = time.perf_counter()
        fields[name] = fit_fields(episodes)['aligned']
        training[name] = time.perf_counter()-began
        print(name, 'trained in', round(training[name], 1), 'seconds',
              flush=True)
    groups = transfer_groups()
    scores = {}
    state = {}
    for name, field in fields.items():
        scores[name] = {}
        state[name] = {}
        for split, streams in groups.items():
            scores[name][split], state[name][split] = evaluate_streams(
                streams, observers, field)
            print(name, split, round(scores[name][split][
                'learned_aligned']['top_32_recall'], 3), flush=True)
    prior = json.loads(PRIOR.read_text())
    result = dict(training_episodes=len(repeated),
                  training_seconds=training,
                  prior_path=str(PRIOR),
                  prior_top_32={split: row['learned_aligned'][
                      'top_32_recall'] for split, row in prior['scores'].items()},
                  scores=scores, state=state,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        top_32={name: {split: round(arms['learned_aligned'][
            'top_32_recall'], 3) for split, arms in groups.items()}
            for name, groups in scores.items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

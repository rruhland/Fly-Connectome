"""Train sparse spatial winners within a generic local sequence dictionary."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from event_lag_stack import lagged_event_sequence
from generic_native_cadence import scene_events
from learned_transition_units import TransitionPopulation
from run_learned_sequence_units import (probe, train_heads, train_state)
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-learned-spatial-competition-results.json')
BASELINE = Path('docs/experiments/2026-09-25-learned-sequence-units-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    encoder = TransitionPopulation(channels=16, seed=0, spatial_radius=1)
    for events in training:
        encoder.reset_state()
        for sensory in lagged_event_sequence(events):
            encoder.step(sensory, learn_dictionary=True)
    dictionary_seconds = time.perf_counter()-started
    print(f'spatial dictionary trained in {dictionary_seconds:.1f}s',
          flush=True)
    model = train_state(encoder, training)
    heads = train_heads(model, training)
    training_seconds = time.perf_counter()-started
    print(f'sparse state and local heads trained in '
          f'{training_seconds:.1f}s total', flush=True)
    calibration = probe(model, heads, training)['arms']
    gains = {name: row['targets']/max(row['mass'], 1e-6)
             for name, row in calibration.items()}
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)],
                  native=[pong_events(seed, stride=1, frames=120)
                          for seed in SEEDS])
    for family, _, events in make_robust_cases():
        if family in ('independent', 'crossing'):
            groups.setdefault(family, []).append(events)
    scores = {name: probe(model, heads, cases, gains=gains)
              for name, cases in groups.items()}
    baseline = json.loads(BASELINE.read_text())
    result = dict(training_episodes=len(training),
                  dictionary_updates=encoder.dictionary_updates,
                  dictionary_assignments=encoder.total_assignments,
                  dictionary_seconds=dictionary_seconds,
                  training_seconds=training_seconds,
                  gains=gains, scores=scores,
                  baseline_scores=baseline['scores'],
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({group: {name: dict(
        sparse_f1=round(row['budget']['f1'], 3),
        baseline_f1=round(baseline['scores'][group]['arms'][name]
                          ['budget']['f1'], 3),
        sparse_top_8=round(row['top_8_recall'], 3),
        baseline_top_8=round(baseline['scores'][group]['arms'][name]
                             ['top_8_recall'], 3),
        sparse_sites=round(row['mean_source_sites'], 2))
        for name, row in scores[group]['arms'].items()}
        for group in groups}), flush=True)


if __name__ == '__main__':
    main()

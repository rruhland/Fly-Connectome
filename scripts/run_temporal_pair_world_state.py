"""Matched opt-in temporal world-state forecast experiment."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_predictive_pair_assemblies import (evaluate_forecasts,
                                            evaluate_state, fit_models)
from run_sparse_recurrent_field import fit_dictionary, fit_transitions
from run_two_timescale_assemblies import step_fast
from temporal_pair_world_state import TemporalPairWorldState


OUT = Path('docs/experiments/2026-09-26-temporal-pair-world-state-results.json')


@torch.no_grad()
def fit_temporal_models(fast, paired, episodes):
    base = TemporalPairWorldState(paired)
    sequences = []
    for events in episodes:
        fast.reset_state()
        base.reset_state()
        active = []
        for event in events:
            winner = base.step(step_fast(fast, event))
            if bool(event.any()):
                active.append((fast.state.clone(), winner))
        for (previous_fast, _), (current_fast, winner) in zip(
                active[:-1], active[1:]):
            residual = current_fast-fast.predict_field(previous_fast)
            base.credit_decoder(winner, residual)
        sequences.append([winner for _, winner in active])
    models = {name: copy.deepcopy(base) for name in
              ('aligned_transition', 'shuffled_transition',
               'frozen_transition')}
    for index, sequence in enumerate(sequences):
        targets = sequence[1:].copy()
        random.Random(9000+index).shuffle(targets)
        for previous, current, shuffled in zip(sequence[:-1],
                                                sequence[1:], targets):
            models['aligned_transition'].credit_transition(previous,
                                                           current)
            models['shuffled_transition'].credit_transition(previous,
                                                            shuffled)
    return models


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = [scene_events(seed) for seed in range(64)]
    fast = fit_dictionary(training)
    fast = fit_transitions(fast, training)['aligned']
    fast_seconds = time.perf_counter()-started
    print(f'fast predictor trained in {fast_seconds:.1f}s', flush=True)
    paired = fit_models(fast, training)['aligned_pair']
    pair_seconds = time.perf_counter()-started
    print(f'paired identities trained in {pair_seconds:.1f}s total',
          flush=True)
    models = fit_temporal_models(fast, paired, training)
    models['static_pair'] = paired
    training_seconds = time.perf_counter()-started
    print(f'temporal arms trained in {training_seconds:.1f}s total',
          flush=True)
    cases = make_cases()
    event_cache = [case_events(case) for case in cases]
    groups = dict(generic_heldout=[scene_events(1000+seed, heldout=True)
                                   for seed in range(16)])
    for case, events in zip(cases, event_cache):
        groups.setdefault(case['split'], []).append(events)
    forecasts = {split: evaluate_forecasts(fast, models, episodes)
                 for split, episodes in groups.items()}
    probe = evaluate_state(fast, models, cases, event_cache)
    result = dict(training_episodes=len(training),
                  fast_seconds=fast_seconds, pair_seconds=pair_seconds,
                  training_seconds=training_seconds,
                  decoder_mass=float(models['aligned_transition'].decoder
                                     .abs().sum()),
                  transition_mass={name: float(model.transitions.sum())
                                   for name, model in models.items()
                                   if name != 'static_pair'},
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

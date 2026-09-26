"""Registered state-level continuity and binding comparison."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from continuity_binding_field import ContinuityBindingField
from generic_native_cadence import scene_events
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_predictive_pair_assemblies import (evaluate_forecasts,
                                            evaluate_state, fit_models)
from run_sparse_recurrent_field import fit_dictionary, fit_transitions
from run_two_timescale_assemblies import step_fast


OUT = Path('docs/experiments/2026-09-26-continuity-binding-field-results.json')


@torch.no_grad()
def fit_binding_models(fast, paired, episodes):
    base = ContinuityBindingField(paired)
    models = {name: copy.deepcopy(base) for name in
              ('aligned_binding', 'shuffled_binding', 'frozen_feedback')}
    for index, events in enumerate(episodes):
        fast.reset_state()
        base.reset_state()
        active = []
        for event in events:
            winner = base.step(step_fast(fast, event))
            if bool(event.any()):
                active.append(winner)
        targets = active[1:].copy()
        random.Random(11000+index).shuffle(targets)
        for previous, current, shuffled in zip(active[:-1],
                                                active[1:], targets):
            models['aligned_binding'].credit(previous, current)
            models['shuffled_binding'].credit(previous, shuffled)
    for model in models.values():
        model.normalize()
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
    paired_seconds = time.perf_counter()-started
    print(f'paired sensory code trained in {paired_seconds:.1f}s total',
          flush=True)
    models = fit_binding_models(fast, paired, training)
    models['sparse_pair'] = paired
    training_seconds = time.perf_counter()-started
    print(f'continuity fields trained in {training_seconds:.1f}s total',
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
                  fast_seconds=fast_seconds,
                  paired_seconds=paired_seconds,
                  training_seconds=training_seconds,
                  kernels={name: dict(
                      temporal_mass=float(model.temporal_counts.sum()),
                      lateral_mass=float(model.lateral_counts.sum()),
                      learned_temporal_nonzero=int(
                          (model.temporal_counts > 0).sum()),
                      learned_lateral_nonzero=int(
                          (model.lateral_counts > 0).sum()))
                      for name, model in models.items()
                      if name != 'sparse_pair'},
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
        coherence={name: {split: dict(
            outside=round(row['outside_activity_fraction'], 3),
            components=round(row['components_per_frame'], 3))
                          for split, row in groups.items()}
                   for name, groups in probe.items()
                   if name != 'fast_only'},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

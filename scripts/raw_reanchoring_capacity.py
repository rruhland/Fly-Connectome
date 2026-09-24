"""Offline capacity diagnostic for frozen raw-event + coincidence latents."""

import json
import time
from pathlib import Path

import torch

from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import unseen_shape_sequences
from frozen_code_capacity import evaluate as evaluate_capacity, fit_decoder
from raw_correlation_latent import (decoder_matrix, raw_correlation_sequence,
                                    train_raw_models)
from run_learned_transition_units import evaluate as evaluate_local


OUT = Path('docs/experiments/2026-09-24-raw-reanchoring-capacity-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    models, training = train_raw_models()
    frozen = {name: models[name] for name in ('learned', 'random_dictionary')}
    weights = {}
    fit = {}
    for name, model in frozen.items():
        weights[name], fit[name] = fit_decoder(model, training)
    capacity_models = {name: (frozen[name], weights[name]) for name in frozen}
    training_cases = [('training', direction, 1, True, events, codes)
                      for direction, events, codes in training]
    single_cases = [(shape, direction, speed, background, events,
                     raw_correlation_sequence(events))
                    for shape, direction, speed, background, events
                    in unseen_shape_sequences()]
    robust_cases = [(family, 'mixed', meta['speed'], meta['background'], events,
                     raw_correlation_sequence(events))
                    for family, meta, events in make_robust_cases()]
    local_training = evaluate_local(
        frozen, training_cases, decoder_matrix(18))
    training_capacity = evaluate_capacity(capacity_models, training_cases)
    single_capacity = evaluate_capacity(capacity_models, single_cases)
    robust_capacity = evaluate_capacity(capacity_models, robust_cases)
    result = dict(training_episodes=len(training), fit=fit,
                  local_training=local_training['overall'],
                  training=training_capacity,
                  single=single_capacity, robust=robust_capacity,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        local_training={name: round(value['event']['f1'], 3)
                        for name, value in local_training['overall'].items()},
        capacity_training={name: round(value['f1'], 3)
                           for name, value in training_capacity['overall'].items()},
        capacity_single={name: round(value['f1'], 3)
                         for name, value in single_capacity['overall'].items()},
        capacity_robust={name: round(value['f1'], 3)
                         for name, value in robust_capacity['overall'].items()},
        coverage_single={name: round(value['reachable_fraction'], 3)
                         for name, value in single_capacity['coverage'].items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

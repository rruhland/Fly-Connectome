"""Opt-in learned visual state over fixed local event-correlation input."""

import json
import random
import time
from pathlib import Path

import torch

from diverse_visual_experience import (diverse_training_sequences,
                                       unseen_shape_sequences)
from frozen_code_capacity import evaluate as evaluate_capacity, fit_decoder
from generic_local_transition import OFFSETS, shift
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import evaluate as evaluate_local, probe


OUT = Path('docs/experiments/2026-09-24-correlation-input-latent-results.json')


def correlation_sequence(events):
    previous = torch.zeros_like(events[0])
    codes = []
    for current in events:
        channels = torch.stack([shift(previous, dy, dx)*current
                                for dy, dx in OFFSETS])
        codes.append(channels.permute(1, 0, 2, 3).reshape(16, 32, 64))
        previous = current
    return codes


def primitive_to_events(prediction):
    return prediction.reshape(2, len(OFFSETS), 32, 64).sum(1).clamp(0, 1)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    raw_passes = diverse_training_sequences()
    training_passes = [[(shape, direction, speed, background, events,
                         correlation_sequence(events))
                        for shape, direction, speed, background, events in episodes]
                       for episodes in raw_passes]
    training = [case for episodes in training_passes for case in episodes]
    heldout = [(shape, direction, speed, background, events,
                correlation_sequence(events))
               for shape, direction, speed, background, events
               in unseen_shape_sequences()]
    learned = TransitionPopulation(channels=16, seed=0)
    random_code = TransitionPopulation(channels=16, seed=0)
    rng = random.Random(0)
    for episodes in training_passes:
        order = episodes.copy()
        rng.shuffle(order)
        for *_, codes in order:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)
    shuffled = TransitionPopulation(channels=16, seed=0)
    shuffled.dictionary.copy_(learned.dictionary)
    shuffled.usage.copy_(learned.usage)
    shuffled.total_assignments = learned.total_assignments
    rng = random.Random(1)
    for episodes in training_passes:
        order = episodes.copy()
        rng.shuffle(order)
        for *_, codes in order:
            future = codes[1:].copy()
            rng.shuffle(future)
            for model in (learned, random_code, shuffled):
                model.reset_state()
            for t, code in enumerate(codes):
                learned.step(code, learn_prediction=True)
                random_code.step(code, learn_prediction=True)
                shuffled.step(code, learn_prediction=True,
                              credit_target=future[t-1] if t else None)
    fixed_decoder = torch.zeros((16, 2))
    fixed_decoder[:8, 0] = 1
    fixed_decoder[8:, 1] = 1
    models = {'learned': learned, 'random_dictionary': random_code,
              'shuffled_future': shuffled}
    local = evaluate_local(models, heldout, fixed_decoder)
    local_train = evaluate_local(models, training, fixed_decoder)
    decoder_training = [(direction, events, codes)
                        for _, direction, _, _, events, codes in training]
    weights = {}
    fit = {}
    for name in ('learned', 'random_dictionary'):
        weights[name], fit[name] = fit_decoder(models[name], decoder_training)
    capacity_models = {name: (models[name], weights[name]) for name in weights}
    capacity = evaluate_capacity(capacity_models, heldout)
    capacity_train = evaluate_capacity(capacity_models, training)
    result = dict(training_episodes=len(training),
                  code_probe={name: probe(model, decoder_training, heldout)
                              for name, model in models.items()},
                  local=local, local_training=local_train['overall'],
                  capacity=capacity,
                  capacity_training=capacity_train['overall'], fit=fit,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        probe={name: value['accuracy'] for name, value in result['code_probe'].items()},
        local={name: value['event']['f1']
               for name, value in local['overall'].items()},
        local_train={name: value['event']['f1']
                     for name, value in local_train['overall'].items()},
        capacity={name: value['f1'] for name, value in capacity['overall'].items()},
        capacity_train={name: value['f1']
                        for name, value in capacity_train['overall'].items()},
        fixed=local['fixed_correlation_event']['f1'],
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

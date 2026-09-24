"""Opt-in empirical co-occurrence readout on frozen local transition codes."""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from learned_latent_probe import (LocalVisualLatent, heldout_sequences,
                                  training_sequences)
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import encode, evaluate, probe


OUT = Path('docs/experiments/2026-09-24-empirical-transition-readout-results.json')


@torch.no_grad()
def conditional_weights(model, sequences, *, shuffle_targets=False, seed=1):
    counts = torch.zeros_like(model.predictive)
    source_counts = torch.zeros(model.units)
    rng = random.Random(seed)
    for sequence in sequences:
        model.reset_state()
        targets = sequence[1:].copy()
        if shuffle_targets:
            rng.shuffle(targets)
        for current, target in zip(sequence[:-1], targets):
            model.step(current)
            padded = F.pad(target[None], (2, 2, 2, 2))[0]
            for y, x, unit in model.previous_sources:
                counts[:, unit] += padded[:, y:y+5, x:x+5]
                source_counts[unit] += 1
    return counts/source_counts.clamp(min=1)[None, :, None, None]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    raw_train = training_sequences()
    raw_heldout = heldout_sequences()
    sensory = LocalVisualLatent(seed=0, homeostasis=True)
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(raw_train)
        for _, events in raw_train:
            sensory.reset_state()
            for event in events:
                sensory.step(event)
    training = [(direction, events, encode(sensory, events))
                for direction, events in raw_train]
    heldout = [(shape, direction, speed, background, events,
                encode(sensory, events))
               for shape, direction, speed, background, events in raw_heldout]

    dictionary = TransitionPopulation(seed=0)
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(training)
        for _, _, codes in training:
            dictionary.reset_state()
            for code in codes:
                dictionary.step(code, learn_dictionary=True)
    sequences = [codes for _, _, codes in training]
    empirical = TransitionPopulation(seed=0)
    shuffled = TransitionPopulation(seed=0)
    for model in (empirical, shuffled):
        model.dictionary.copy_(dictionary.dictionary)
        model.usage.copy_(dictionary.usage)
        model.total_assignments = dictionary.total_assignments
    empirical.predictive.copy_(conditional_weights(dictionary, sequences))
    shuffled.predictive.copy_(conditional_weights(
        dictionary, sequences, shuffle_targets=True))

    decoder = sensory.sensory[:, [12, 37]]
    heldout_scores = evaluate(dict(empirical=empirical,
                                   shuffled_target=shuffled), heldout, decoder)
    square_cases = [('square', direction, 1, True, events, codes)
                    for direction, events, codes in training]
    training_scores = evaluate(dict(empirical=empirical),
                               square_cases, decoder)['overall']['empirical']
    result = dict(dictionary_probe=probe(dictionary, training, heldout),
                  heldout=heldout_scores, training_fit=training_scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(dictionary_probe=result['dictionary_probe']['accuracy'],
                          heldout={name: {metric: round(row['f1'], 3)
                                          for metric, row in pair.items()}
                                   for name, pair in heldout_scores['overall'].items()},
                          by_speed={speed: {name: {metric: round(row['f1'], 3)
                                                   for metric, row in pair.items()}
                                            for name, pair in methods.items()}
                                    for speed, methods in heldout_scores['by_speed'].items()},
                          training_fit={metric: round(row['f1'], 3)
                                        for metric, row in training_scores.items()},
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

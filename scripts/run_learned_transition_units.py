"""Compare learned local transition units with matched unlabeled controls."""

import json
import random
import time
from pathlib import Path

import torch

from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from learned_latent_probe import (DIRECTIONS, LocalVisualLatent,
                                  add_threshold_counts, empty_threshold_counts,
                                  finish_threshold_counts, heldout_sequences,
                                  training_sequences)
from learned_transition_units import TransitionPopulation


OUT = Path('docs/experiments/2026-09-24-learned-transition-units-results.json')


@torch.no_grad()
def encode(sensory, events):
    sensory.reset_state()
    sequence = []
    for event in events:
        sensory.step(event, learn=False)
        sequence.append(sensory.latent.clone())
    return sequence


@torch.no_grad()
def code_feature(model, codes):
    model.reset_state()
    counts = torch.zeros(model.units)
    for t, code in enumerate(codes):
        model.step(code)
        if 3 <= t < 14:
            counts += model.latent.sum((1, 2))
    return counts


def probe(model, training, heldout):
    by_direction = {direction: [] for direction in DIRECTIONS}
    for direction, _, codes in training:
        counts = code_feature(model, codes)
        by_direction[direction].append(counts/counts.sum().clamp(min=1))
    centers = torch.stack([torch.stack(by_direction[direction]).mean(0)
                           for direction in DIRECTIONS])
    correct = 0
    use = torch.zeros(model.units)
    groups = {}
    for shape, direction, speed, background, _, codes in heldout:
        counts = code_feature(model, codes)
        use += counts
        feature = counts/counts.sum().clamp(min=1)
        predicted = DIRECTIONS[torch.cdist(feature[None], centers).argmin()]
        match = predicted == direction
        correct += match
        key = f'{shape}:s{speed}:{"dark" if background else "bright"}'
        group = groups.setdefault(key, dict(correct=0, total=0))
        group['correct'] += match
        group['total'] += 1
    frequency = use/use.sum().clamp(min=1)
    active = frequency > 0
    effective = float((-(frequency[active]*frequency[active].log()).sum()).exp())
    return dict(correct=correct, total=len(heldout), accuracy=correct/len(heldout),
                effective_units=effective, usage=frequency.tolist(), groups=groups)


@torch.no_grad()
def evaluate(models, cases, center_decoder):
    scores = {name: dict(event=empty_score(), latent=empty_score())
              for name in models}
    by_speed = {speed: {name: dict(event=empty_score(), latent=empty_score())
                        for name in models} for speed in (1, 2)}
    by_shape_polarity = {}
    thresholds = {name: dict(event=empty_threshold_counts(),
                             latent=empty_threshold_counts()) for name in models}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    fixed_score = empty_score()
    for shape, _, speed, background, events, codes in cases:
        label = f'{shape}:s{speed}:{"dark" if background else "bright"}'
        group = by_shape_polarity.setdefault(label, {
            name: dict(event=empty_score(), latent=empty_score())
            for name in models})
        for model in models.values():
            model.reset_state()
        fixed.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            forecasts = {name: model.step(code) for name, model in models.items()}
            fixed_prediction = fixed.step(event, learn=False)
            if 3 <= t < 14:
                accumulate(fixed_score, fixed_prediction, events[t+1])
                for name, predicted_code in forecasts.items():
                    predicted_event = torch.einsum('kc,khw->chw', center_decoder,
                                                    predicted_code).clamp_(0, 1)
                    for metric, prediction, target in (
                            ('event', predicted_event, events[t+1]),
                            ('latent', predicted_code, codes[t+1])):
                        accumulate(scores[name][metric], prediction, target)
                        accumulate(by_speed[speed][name][metric], prediction, target)
                        accumulate(group[name][metric], prediction, target)
                        add_threshold_counts(thresholds[name][metric],
                                             prediction, target)
    return dict(overall={name: {metric: finish(score) for metric, score in pair.items()}
                         for name, pair in scores.items()},
                by_speed={speed: {name: {metric: finish(score)
                                         for metric, score in pair.items()}
                                  for name, pair in methods.items()}
                          for speed, methods in by_speed.items()},
                by_shape_polarity={label: {name: {metric: finish(score)
                                                   for metric, score in pair.items()}
                                           for name, pair in methods.items()}
                                   for label, methods in by_shape_polarity.items()},
                threshold_f1={name: {metric: finish_threshold_counts(value)
                                     for metric, value in pair.items()}
                              for name, pair in thresholds.items()},
                fixed_correlation_event=finish(fixed_score))


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

    learned = TransitionPopulation(seed=0)
    random_dictionary = TransitionPopulation(seed=0)
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(training)
        for _, _, codes in training:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)

    shuffled = TransitionPopulation(seed=0)
    shuffled.dictionary.copy_(learned.dictionary)
    shuffled.usage.copy_(learned.usage)
    shuffled.total_assignments = learned.total_assignments
    rng = random.Random(1)
    for _ in range(3):
        rng.shuffle(training)
        for _, _, codes in training:
            learned.reset_state()
            shuffled.reset_state()
            targets = codes[1:].copy()
            rng.shuffle(targets)
            for t, code in enumerate(codes):
                learned.step(code, learn_prediction=True)
                shuffled.step(code, learn_prediction=True,
                              credit_target=targets[t-1] if t else None)

    decoder = sensory.sensory[:, [12, 37]]
    probes = dict(learned=probe(learned, training, heldout),
                  random_dictionary=probe(random_dictionary, training, heldout))
    forecasts = evaluate(dict(learned=learned, shuffled_target=shuffled),
                         heldout, decoder)
    square_cases = [('square', direction, 1, True, events, codes)
                    for direction, events, codes in training]
    train_fit = evaluate(dict(learned=learned), square_cases, decoder)['overall']
    result = dict(training_episodes=3*len(training),
                  dictionary_updates=learned.dictionary_updates,
                  predictive_updates=learned.predictive_updates,
                  probes=probes, forecasts=forecasts,
                  training_fit=train_fit,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(probes={name: dict(accuracy=row['accuracy'],
                                              effective_units=row['effective_units'])
                                   for name, row in probes.items()},
                          heldout={name: {metric: round(row['f1'], 3)
                                          for metric, row in pair.items()}
                                   for name, pair in forecasts['overall'].items()},
                          by_speed={speed: {name: {metric: round(row['f1'], 3)
                                                   for metric, row in pair.items()}
                                            for name, pair in methods.items()}
                                    for speed, methods in forecasts['by_speed'].items()},
                          fixed_f1=forecasts['fixed_correlation_event']['f1'],
                          training_fit={metric: round(row['f1'], 3)
                                        for metric, row in train_fit['learned'].items()},
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

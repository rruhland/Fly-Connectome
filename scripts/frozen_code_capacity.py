"""Evaluation-only sparse linear readout of frozen learned visual codes."""

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
import torch

from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from learned_latent_probe import (LocalVisualLatent, add_threshold_counts,
                                  empty_threshold_counts,
                                  finish_threshold_counts,
                                  heldout_sequences, training_sequences)
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import encode, probe


OUT = Path('docs/experiments/2026-09-24-frozen-code-capacity-results.json')


def scatter_local_decoder(weights, sources, *, units):
    canvas = torch.zeros((2, 36, 68))
    reachable = torch.zeros((36, 68), dtype=torch.bool)
    for y, x, unit in sources:
        canvas[:, y:y+5, x:x+5] += weights[:, unit*25:(unit+1)*25].reshape(2, 5, 5)
        reachable[y:y+5, x:x+5] = True
    return canvas[:, 2:-2, 2:-2].clamp_(0, 1), reachable[2:-2, 2:-2]


def local_features(sources, *, units):
    pixels = {}
    for y, x, unit in sources:
        for dy in range(-2, 3):
            ty = y+dy
            if not 0 <= ty < 32:
                continue
            for dx in range(-2, 3):
                tx = x+dx
                if not 0 <= tx < 64:
                    continue
                pixel = ty*64+tx
                feature = unit*25+(dy+2)*5+dx+2
                row = pixels.setdefault(pixel, {})
                row[feature] = row.get(feature, 0)+1
    return pixels


@torch.no_grad()
def fit_decoder(model, training):
    row_indices = []
    col_indices = []
    values = []
    targets = []
    positives = covered = 0
    for _, events, codes in training:
        model.reset_state()
        for t, code in enumerate(codes):
            model.step(code)
            if not 3 <= t < 14:
                continue
            future = events[t+1].reshape(2, -1)
            features = local_features(model.previous_sources, units=model.units)
            positives += int(future.sum())
            for pixel, entries in features.items():
                row = len(targets)
                for feature, value in entries.items():
                    row_indices.append(row)
                    col_indices.append(feature)
                    values.append(value)
                target = future[:, pixel].tolist()
                targets.append(target)
                covered += int(sum(target))
    matrix = sparse.csr_matrix((values, (row_indices, col_indices)),
                               shape=(len(targets), model.units*25),
                               dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    weights = np.stack([lsqr(matrix, labels[:, polarity], damp=1.,
                              atol=1e-5, btol=1e-5, iter_lim=200)[0]
                        for polarity in range(2)])
    return torch.from_numpy(weights).float(), dict(
        candidate_rows=len(targets), nonzero_features=matrix.nnz,
        target_events=positives, reachable_events=covered,
        reachable_fraction=covered/positives if positives else None)


@torch.no_grad()
def evaluate(models, cases):
    scores = {name: empty_score() for name in models}
    thresholds = {name: empty_threshold_counts() for name in models}
    by_speed = {speed: {name: empty_score() for name in models}
                for speed in (1, 2)}
    groups = {}
    coverage = {name: dict(target_events=0, reachable_events=0)
                for name in models}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    fixed_score = empty_score()
    for shape, _, speed, background, events, codes in cases:
        label = f'{shape}:s{speed}:{"dark" if background else "bright"}'
        group = groups.setdefault(label, {name: empty_score() for name in models})
        for model, _ in models.values():
            model.reset_state()
        fixed.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            predictions = {}
            for name, (model, weights) in models.items():
                model.step(code)
                predictions[name] = scatter_local_decoder(
                    weights, model.previous_sources, units=model.units)
            fixed_prediction = fixed.step(event, learn=False)
            if 3 <= t < 14:
                target = events[t+1]
                accumulate(fixed_score, fixed_prediction, target)
                for name, (prediction, reachable) in predictions.items():
                    accumulate(scores[name], prediction, target)
                    accumulate(by_speed[speed][name], prediction, target)
                    accumulate(group[name], prediction, target)
                    add_threshold_counts(thresholds[name], prediction, target)
                    coverage[name]['target_events'] += int(target.sum())
                    coverage[name]['reachable_events'] += int(
                        target[:, reachable].sum())
    return dict(overall={name: finish(value) for name, value in scores.items()},
                by_speed={speed: {name: finish(value) for name, value in methods.items()}
                          for speed, methods in by_speed.items()},
                by_shape_polarity={label: {name: finish(value)
                                           for name, value in methods.items()}
                                   for label, methods in groups.items()},
                threshold_f1={name: finish_threshold_counts(value)
                              for name, value in thresholds.items()},
                coverage={name: dict(**value, reachable_fraction=(
                    value['reachable_events']/value['target_events']
                    if value['target_events'] else None))
                          for name, value in coverage.items()},
                fixed_correlation=finish(fixed_score))


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
    random_code = TransitionPopulation(seed=0)
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(training)
        for _, _, codes in training:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)
    learned_weights, learned_fit = fit_decoder(learned, training)
    random_weights, random_fit = fit_decoder(random_code, training)
    models = dict(learned=(learned, learned_weights),
                  random_dictionary=(random_code, random_weights))
    heldout_scores = evaluate(models, heldout)
    square_cases = [('square', direction, 1, True, events, codes)
                    for direction, events, codes in training]
    training_scores = evaluate(models, square_cases)
    result = dict(dictionary_probe=dict(
                    learned=probe(learned, training, heldout),
                    random=probe(random_code, training, heldout)),
                  fit=dict(learned=learned_fit, random=random_fit),
                  training=training_scores, heldout=heldout_scores,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(probe={name: row['accuracy'] for name, row in
                                  result['dictionary_probe'].items()},
                          heldout={name: round(row['f1'], 3)
                                   for name, row in heldout_scores['overall'].items()},
                          training={name: round(row['f1'], 3)
                                    for name, row in training_scores['overall'].items()},
                          coverage={name: round(row['reachable_fraction'], 3)
                                    for name, row in heldout_scores['coverage'].items()},
                          elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

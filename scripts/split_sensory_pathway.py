"""Frozen-code test of separate learned coincidence and raw-event pathways."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases, train_models
from diverse_visual_experience import (diverse_training_sequences,
                                       unseen_shape_sequences)
from frozen_code_capacity import (evaluate, fit_decoder,
                                  scatter_local_decoder)
from generic_local_transition import accumulate, empty_score, finish
from learned_transition_units import TransitionPopulation
from raw_correlation_latent import raw_correlation_sequence


OUT = Path('docs/experiments/2026-09-24-split-sensory-pathway-results.json')


class SplitPopulation:
    """Independent local populations; their source lists share a readout only."""

    def __init__(self, correlation, raw):
        self.correlation = correlation
        self.raw = raw
        self.units = correlation.units+raw.units
        self.reset_state()

    def reset_state(self):
        self.correlation.reset_state()
        self.raw.reset_state()
        self.previous_sources = []
        self.latent = torch.zeros((self.units, 32, 64))

    @torch.no_grad()
    def step(self, code):
        self.correlation.step(code[2:])
        self.raw.step(code[:2])
        self.latent = torch.cat((self.correlation.latent, self.raw.latent))
        self.previous_sources = self.correlation.previous_sources + [
            (y, x, unit+self.correlation.units)
            for y, x, unit in self.raw.previous_sources]


@torch.no_grad()
def train_raw_branch():
    raw = TransitionPopulation(channels=2, units=8, seed=0)
    random_raw = TransitionPopulation(channels=2, units=8, seed=0)
    passes = [[(direction, events) for _, direction, _, _, events in episodes]
              for episodes in diverse_training_sequences()]
    rng = random.Random(0)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, events in order:
            raw.reset_state()
            for event in events:
                raw.step(event, learn_dictionary=True)
    training = [(direction, events, raw_correlation_sequence(events))
                for episodes in passes for direction, events in episodes]
    return raw, random_raw, training


@torch.no_grad()
def reappearance(models, weights, cases):
    scores = {name: empty_score() for name in models}
    sources = {name: 0 for name in models}
    for family, _, events in cases:
        if family != 'occlusion':
            continue
        combined = raw_correlation_sequence(events)
        coincidence = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        for t in range(11):
            for name, model in models.items():
                model.step(coincidence[t] if name == 'correlation_only'
                           else combined[t])
        for name, model in models.items():
            prediction, _ = scatter_local_decoder(
                weights[name], model.previous_sources, units=model.units)
            accumulate(scores[name], prediction, events[11])
            sources[name] += len(model.previous_sources)
    return dict(scores={name: finish(value) for name, value in scores.items()},
                source_count=sources)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    correlation = train_models()['learned']
    raw, random_raw, training = train_raw_branch()
    models = dict(
        correlation_only=copy.deepcopy(correlation),
        split_learned=SplitPopulation(copy.deepcopy(correlation), raw),
        split_random_raw=SplitPopulation(copy.deepcopy(correlation), random_raw))
    baseline_training = [(direction, events, correlation_sequence(events))
                         for direction, events, _ in training]
    weights = {}
    fit = {}
    for name, model in models.items():
        weights[name], fit[name] = fit_decoder(
            model, baseline_training if name == 'correlation_only' else training)
    single_raw = unseen_shape_sequences()
    robust_raw = make_robust_cases()
    results = {}
    for label, cases in (
        ('training', [('training', direction, 1, True, events)
                      for direction, events, _ in training]),
        ('single', single_raw),
        ('robust', [(family, 'mixed', meta['speed'], meta['background'], events)
                    for family, meta, events in robust_raw])):
        scores = {}
        for name, model in models.items():
            encoded = [(shape, direction, speed, background, events,
                        correlation_sequence(events) if name == 'correlation_only'
                        else raw_correlation_sequence(events))
                       for shape, direction, speed, background, events in cases]
            scores[name] = evaluate({name: (model, weights[name])}, encoded)
        results[label] = scores
    selected = reappearance(models, weights, robust_raw)
    result = dict(training_episodes=len(training), fit=fit, **results,
                  reappearance=selected,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training={name: round(data['overall'][name]['f1'], 3)
                  for name, data in results['training'].items()},
        single={name: round(data['overall'][name]['f1'], 3)
                for name, data in results['single'].items()},
        robust={name: round(data['overall'][name]['f1'], 3)
                for name, data in results['robust'].items()},
        reappearance={name: round(score['f1'], 3)
                      for name, score in selected['scores'].items()},
        source_count=selected['source_count'],
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

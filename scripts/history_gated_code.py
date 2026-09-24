"""Unlabeled history-selective units gated by generic raw event onset."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import scene_sequence, train_models
from diverse_visual_experience import TRAIN_SHAPES, TEST_SHAPES
from learned_transition_units import TransitionPopulation
from local_context_trace import context_code
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-24-history-gated-code-results.json')
DIRECTIONS = {'up': (-1, 0), 'down': (1, 0),
              'left': (0, -1), 'right': (0, 1)}


class HistoryGatedCode:
    def __init__(self, motion):
        self.motion = motion
        self.history = TransitionPopulation(channels=24, units=8, seed=0)
        self.reset_state()

    def reset_state(self):
        self.motion.reset_state()
        self.history.reset_state()
        self.trace = torch.zeros((self.motion.units, 32, 64))

    @torch.no_grad()
    def step(self, events, coincidence, *, learn_dictionary=False):
        self.motion.step(coincidence)
        code = context_code(events, coincidence, self.trace,
                            use_trace=True)[2:]
        self.history.step(code, learn_dictionary=learn_dictionary)
        self.trace.mul_(.8).add_(self.motion.latent).clamp_(0, 1)
        return self.history.previous_sources


@torch.no_grad()
def train_models_for_history(cases):
    motion = train_models()['learned']
    learned = HistoryGatedCode(copy.deepcopy(motion))
    random_code = HistoryGatedCode(copy.deepcopy(motion))
    encoded = [(events, correlation_sequence(events))
               for _, _, events in cases]
    rng = random.Random(3)
    for _ in range(2):
        rng.shuffle(encoded)
        for events, codes in encoded:
            learned.reset_state()
            for event, code in zip(events, codes):
                learned.step(event, code, learn_dictionary=True)
    return dict(learned=learned, random=random_code)


def interrupted_cases(shapes):
    cases = []
    for shape in shapes:
        for direction, (dy, dx) in DIRECTIONS.items():
            for speed in (1, 2):
                for background in (True, False):
                    obj = dict(shape=shape, center=(16, 32),
                               before=(dy*speed, dx*speed),
                               after=(dy*speed, dx*speed),
                               hidden=(7, 8, 9))
                    cases.append((shape, direction, speed, background,
                                  scene_sequence([obj],
                                                 background=background)))
    return cases


@torch.no_grad()
def sources_at_reappearance(model, events, *, reset_trace=False):
    model.reset_state()
    codes = correlation_sequence(events)
    for t in range(11):
        if t == 10 and reset_trace:
            model.trace.zero_()
        model.step(events[t], codes[t])
    return model.history.previous_sources.copy()


def unit_histogram(sources, units):
    counts = torch.bincount(torch.tensor([unit for _, _, unit in sources],
                                         dtype=torch.long),
                            minlength=units).float()
    return counts/counts.sum().clamp(min=1)


@torch.no_grad()
def direction_probe(model, training, heldout):
    prototypes = {direction: torch.zeros(model.history.units)
                  for direction in DIRECTIONS}
    for _, direction, _, _, events in training:
        sources = sources_at_reappearance(model, events)
        prototypes[direction] += unit_histogram(
            sources, model.history.units)
    for direction in prototypes:
        prototypes[direction] /= sum(case[1] == direction for case in training)
        prototypes[direction] /= prototypes[direction].norm().clamp(min=1e-6)
    correct = active_sites = 0
    by_shape = {}
    for shape, direction, _, _, events in heldout:
        sources = sources_at_reappearance(model, events)
        vector = unit_histogram(sources, model.history.units)
        predicted = max(DIRECTIONS, key=lambda key: float(
            prototypes[key] @ vector))
        correct += predicted == direction
        active_sites += len(sources)
        row = by_shape.setdefault(shape, dict(correct=0, total=0))
        row['correct'] += predicted == direction
        row['total'] += 1
    return dict(correct=correct, total=len(heldout),
                accuracy=correct/len(heldout),
                active_sites=active_sites,
                by_shape=by_shape)


@torch.no_grad()
def matched_pair(model):
    sources = []
    raw = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        raw.append(events[10])
        sources.append(sources_at_reappearance(model, events))
    left = {(y, x): unit for y, x, unit in sources[0]}
    right = {(y, x): unit for y, x, unit in sources[1]}
    common = left.keys() & right.keys()
    different = sum(left[site] != right[site] for site in common)
    return dict(same_raw_input=torch.equal(*raw),
                left_sources=len(left), right_sources=len(right),
                common_sites=len(common), different_units=different,
                different_fraction=different/len(common) if common else 0)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    models = train_models_for_history(make_training_cases())
    training = interrupted_cases(TRAIN_SHAPES)
    heldout = interrupted_cases(TEST_SHAPES)
    result = dict(training_cases=len(training), heldout_cases=len(heldout),
                  pair={name: matched_pair(model)
                        for name, model in models.items()},
                  probe={name: direction_probe(model, training, heldout)
                         for name, model in models.items()},
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        pair=result['pair'],
        probe={name: dict(accuracy=round(row['accuracy'], 3),
                          correct=row['correct'],
                          active_sites=row['active_sites'])
               for name, row in result['probe'].items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

"""One shared, bounded representation-first comparison of seven families."""

import copy
import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from representation_families import FAMILIES
from representation_family_audit import (SensoryControl, heldout_inputs,
                                         observed_inputs, score_cases,
                                         score_perturbations)
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events
from run_local_observation_model import fit_observers
from run_multicadence_experience import training_sequences


OUT = Path('docs/experiments/2026-09-26-representation-family-results.json')
HARD = ('separated_two_components', 'crossing_identity', 'noise_margin',
        'missing_gap_similarity', 'missing_recovery_similarity')
EASY = ('position', 'speed', 'shape')


@torch.no_grad()
def train(model, episodes):
    for episode in episodes:
        model.reset_state()
        for value in episode:
            model.step(value, learn=True)


def _value(scores, name):
    if name in ('separated_two_components', 'crossing_identity'):
        return scores['cases'][name]
    return scores['perturbations'][name]


def promising(trained, sensory, frozen):
    hard_gains = {name: _value(trained, name)-max(
        _value(sensory, name), _value(frozen, name)) for name in HARD}
    if max(hard_gains.values()) < .05:
        return False, hard_gains
    if not .001 <= trained['cases']['active_fraction'] <= .75:
        return False, hard_gains
    for split in EASY:
        for measure in ('occupancy_f1', 'direction_accuracy'):
            if (trained['cases']['splits'][split][measure] <
                    sensory['cases']['splits'][split][measure]-.10):
                return False, hard_gains
    return True, hard_gains


@torch.no_grad()
def evaluate(model, cases, case_inputs, clean, noisy, missing):
    return dict(cases=score_cases(model, cases, case_inputs),
                perturbations=score_perturbations(
                    model, clean, noisy, missing))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observer = fit_observers(original)['aligned']
    _, training = training_sequences(seeds=range(16))
    training_inputs = [observed_inputs(observer, episode)
                       for episode in training]
    cases = make_cases()
    case_inputs = [observed_inputs(observer, case_events(case))
                   for case in cases]
    heldout = [scene_events(seed, heldout=True)
               for seed in (2000, 2001, 2002, 2003)]
    clean, noisy, missing = heldout_inputs(observer, heldout)
    prepared_seconds = time.perf_counter()-started
    print('prepared shared inputs in', round(prepared_seconds, 1), 'seconds',
          flush=True)
    sensory = evaluate(SensoryControl(), cases, case_inputs,
                       clean, noisy, missing)
    result = dict(protocol='2026-09-26-representation-first-family-protocol',
                  training_episodes=len(training),
                  training_frames=sum(map(len, training)),
                  prepared_seconds=prepared_seconds, sensory=sensory,
                  families={})
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, family in FAMILIES.items():
        began = time.perf_counter()
        model = family()
        frozen = copy.deepcopy(model)
        train(model, training_inputs)
        training_seconds = time.perf_counter()-began
        trained_scores = evaluate(model, cases, case_inputs,
                                  clean, noisy, missing)
        frozen_scores = evaluate(frozen, cases, case_inputs,
                                 clean, noisy, missing)
        keep, gains = promising(trained_scores, sensory, frozen_scores)
        result['families'][name] = dict(
            trained=trained_scores, frozen=frozen_scores,
            promising=keep, hard_gains=gains,
            training_seconds=training_seconds,
            elapsed_seconds=time.perf_counter()-began)
        result['elapsed_seconds'] = time.perf_counter()-started
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(name, 'promising', keep,
              'hard', {key: round(value, 3) for key, value in gains.items()},
              'seconds', round(result['families'][name]['elapsed_seconds'], 1),
              flush=True)
    print('completed in', round(result['elapsed_seconds'], 1), 'seconds',
          flush=True)


if __name__ == '__main__':
    main()

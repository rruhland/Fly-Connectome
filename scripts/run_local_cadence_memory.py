"""One frozen no-backprop motion-memory candidate on the M1A.5 gate."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_cadence_memory import LocalCadenceMemory
from run_local_observation_model import fit_observers
from run_visual_history_controls import (cases, changing_motion_cases,
                                         two_mover_cases)
from visual_history_benchmark import periodic_steps, render_case
from visual_history_controls import sensory_sequence
from visual_history_scoring import hidden_rank


OUT = Path('docs/experiments/2026-09-26-local-cadence-memory-results.json')


def training_cases():
    return [render_case(direction=direction, speed=1, y=y, shape=shape,
                        background=background,
                        steps=periodic_steps(64, cycle=(1, 2, 3),
                                             phase=phase))
            for y, shape in ((8, 'dot'), (12, 'square'),
                             (16, 'dot'), (20, 'square'))
            for phase in (0, 1, 2)
            for direction in (-1, 1)
            for background in (False, True)]


@torch.no_grad()
def field_at_decision(model, observer, case, *, hybrid):
    model.reset_state()
    sensory = sensory_sequence(observer, case, hybrid=hybrid)
    for value in sensory[:case['decision']+1]:
        field = model.step(value)
    return field


@torch.no_grad()
def evaluate(model, observer, *, hybrid):
    result = {}
    for name, episodes, multi in (
            ('constant_motion', cases(heldout=True), False),
            ('two_movers', two_mover_cases(), True),
            ('changing_motion', changing_motion_cases(), False)):
        scores = {8: [], 32: [], 64: []}
        for case in episodes:
            field = field_at_decision(model, observer, case, hybrid=hybrid)
            masks = (case['hidden_by_entity'] if multi else [case['hidden']])
            for mask in masks:
                target = mask[case['decision']]
                for k in scores:
                    scores[k].append(hidden_rank(field, target, k=k))
        result[name] = dict(cases=len(episodes),
                            targets=len(scores[32]),
                            top_k={str(k): sum(values)/len(values)
                                   for k, values in scores.items()})
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    observer = fit_observers([
        scene_events(seed, return_contrast=True)
        for seed in range(64)])['aligned']
    learned = LocalCadenceMemory()
    episodes = training_cases()
    for case in episodes:
        learned.reset_state()
        for sensory in sensory_sequence(observer, case, hybrid=False):
            learned.step(sensory, learn=True)
    frozen = LocalCadenceMemory()
    result = dict(training_cases=len(episodes),
                  learned_transition_count=int(learned.transitions.sum()),
                  learned_nonzero_transitions=int(
                      (learned.transitions > 0).sum()), arms={})
    for hybrid in (False, True):
        result['arms']['hybrid' if hybrid else 'event_only'] = {
            'learned': evaluate(learned, observer, hybrid=hybrid),
            'frozen': evaluate(frozen, observer, hybrid=hybrid)}
    result['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

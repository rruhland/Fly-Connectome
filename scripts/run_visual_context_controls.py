"""Audit whether visible context predicts hidden motion beyond kinematics."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from run_local_observation_model import fit_observers
from run_visual_history_controls import capture, cases
from visual_history_benchmark import render_context_case
from visual_history_controls import (FixedLeakyMemory, GenericMultiTracker,
                                     GenericTracker, sensory_sequence)
from visual_history_scoring import fit_visible_readout, hidden_rank


OUT = Path('docs/experiments/2026-09-26-visual-context-control-results.json')


def context_cases():
    return [render_context_case(direction=direction, speed=speed, y=y,
                                shape=shape, cue_sign=cue_sign,
                                background=bool(y % 2))
            for y, speed, shape in ((11, 1, 'plus'), (14, 2, 'square'),
                                    (17, 1, 'square'), (20, 2, 'plus'))
            for direction in (-1, 1)
            for cue_sign in (-1, 1)]


@torch.no_grad()
def evaluate(observer, examples, *, hybrid):
    readouts = {name: fit_visible_readout(examples[name])
                for name in ('sensory', 'leaky')}
    scores = {name: {8: [], 32: []}
              for name in ('sensory', 'leaky', 'tracker', 'acceleration',
                           'multi_tracker')}
    for case in context_cases():
        states = sensory_sequence(observer, case, hybrid=hybrid)
        memory = FixedLeakyMemory()
        trackers = dict(tracker=GenericTracker(),
                        acceleration=GenericTracker(acceleration=True),
                        multi_tracker=GenericMultiTracker())
        for frame, state in enumerate(states[:case['decision']+1]):
            fields = {name: tracker.step(state)
                      for name, tracker in trackers.items()}
            saved = dict(sensory=state, leaky=memory.step(state), **fields)
        target = case['hidden'][frame]
        for name in scores:
            field = (readouts[name](saved[name]) if name in readouts
                     else saved[name])
            for k in (8, 32):
                scores[name][k].append(hidden_rank(field, target, k=k))
    return dict(cases=len(context_cases()), top_k={
        str(k): {name: sum(values[k])/len(values[k])
                 for name, values in scores.items()} for k in (8, 32)},
        oracle_top_k=1.)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    observer = fit_observers([
        scene_events(seed, return_contrast=True)
        for seed in range(64)])['aligned']
    result = {}
    for hybrid in (False, True):
        examples = {'sensory': [], 'leaky': []}
        for case in cases(heldout=False):
            _, local = capture(observer, case, hybrid=hybrid,
                               calibration=True)
            for name in examples:
                examples[name].extend(local[name])
        result['hybrid' if hybrid else 'event_only'] = evaluate(
            observer, examples, hybrid=hybrid)
    result['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

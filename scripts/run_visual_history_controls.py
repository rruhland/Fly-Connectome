"""Freeze and audit causal baselines before testing a learned M1A state."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from run_local_observation_model import fit_observers
from visual_history_benchmark import render_case
from visual_history_controls import (FixedLeakyMemory, GenericTracker,
                                     sensory_sequence)
from visual_history_scoring import (fit_visible_readout, hidden_rank,
                                    pair_separation)


OUT = Path('docs/experiments/2026-09-26-visual-history-control-results.json')


def cases(*, heldout):
    if heldout:
        settings = ((7, 1, 'plus'), (13, 3, 'plus'),
                    (19, 1, 'square'), (25, 3, 'square'))
    else:
        settings = ((8, 1, 'dot'), (12, 2, 'dot'),
                    (16, 1, 'square'), (20, 2, 'square'))
    return [render_case(direction=direction, speed=speed, y=y,
                        shape=shape, background=background)
            for y, speed, shape in settings
            for direction in (-1, 1)
            for background in (False, True)]


@torch.no_grad()
def capture(observer, case, *, hybrid, calibration):
    sensory = sensory_sequence(observer, case, hybrid=hybrid)
    leaky = FixedLeakyMemory()
    tracker = GenericTracker()
    examples = {'sensory': [], 'leaky': []}
    decision = {}
    for frame, state in enumerate(sensory):
        memory = leaky.step(state)
        tracked = tracker.step(state)
        if calibration and frame < case['decision'] and frame % 3 == 0:
            visible = case['visible_objects'][frame]
            examples['sensory'].append((state, visible))
            examples['leaky'].append((memory, visible))
        if frame == case['decision']:
            decision = dict(sensory=state, leaky=memory, tracker=tracked)
    return decision, examples


def direction_probe(calibration, heldout, name):
    centers = {}
    for direction in (-1, 1):
        centers[direction] = torch.stack([
            state[name].flatten() for case, state in calibration
            if case['direction'] == direction]).mean(0)
    correct = 0
    for case, state in heldout:
        vector = state[name].flatten()
        estimate = min(centers, key=lambda direction:
                       float((vector-centers[direction]).square().sum()))
        correct += estimate == case['direction']
    return correct / len(heldout)


def summarize(calibration, heldout, examples):
    readouts = {name: fit_visible_readout(examples[name])
                for name in ('sensory', 'leaky')}
    scores = {name: [] for name in ('sensory', 'leaky', 'tracker')}
    for case, state in heldout:
        target = case['hidden'][case['decision']]
        for name in scores:
            field = (state[name] if name == 'tracker' else
                     readouts[name](state[name]))
            scores[name].append(hidden_rank(field, target))
    pairs = []
    for index in range(0, len(heldout), 4):
        left, right = heldout[index], heldout[index+2]
        assert left[0]['direction'] == -1 and right[0]['direction'] == 1
        assert left[0]['decision'] == right[0]['decision']
        pairs.append(pair_separation(left[1]['sensory'],
                                     right[1]['sensory']))
    return dict(cases=len(heldout), top_32={
        name: sum(values)/len(values) for name, values in scores.items()},
        oracle_top_32=1.,
        full_sensory_pair_separation=pairs,
        direction_probe={name: direction_probe(calibration, heldout, name)
                         for name in ('sensory', 'leaky')})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    training = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observer = fit_observers(training)['aligned']
    results = {}
    for hybrid in (False, True):
        examples = {'sensory': [], 'leaky': []}
        calibration = []
        heldout = []
        for case in cases(heldout=False):
            state, local = capture(observer, case, hybrid=hybrid,
                                   calibration=True)
            calibration.append((case, state))
            for name in examples:
                examples[name].extend(local[name])
        for case in cases(heldout=True):
            state, _ = capture(observer, case, hybrid=hybrid,
                               calibration=False)
            heldout.append((case, state))
        results['hybrid' if hybrid else 'event_only'] = summarize(
            calibration, heldout, examples)
    results['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

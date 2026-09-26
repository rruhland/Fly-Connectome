"""One bounded locally learned visual-context state experiment."""

import json
import time
from pathlib import Path

import torch

from generic_native_cadence import scene_events
from local_context_memory import LocalContextMemory
from run_local_observation_model import fit_observers
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from visual_history_benchmark import render_context_case
from visual_history_controls import sensory_sequence
from visual_history_scoring import hidden_rank
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-local-context-memory-results.json')


def training_cases(*, shuffled):
    return [render_context_case(
        direction=direction, speed=speed, y=y, shape=shape,
        cue_sign=cue_sign, background=background,
        turn_sign=(cue_sign if not shuffled or y in (10, 16)
                   else -cue_sign))
        for y, speed, shape in ((10, 1, 'dot'), (12, 2, 'square'),
                                (16, 1, 'square'), (20, 2, 'dot'))
        for direction in (-1, 1)
        for cue_sign in (-1, 1)
        for background in (False, True)]


@torch.no_grad()
def train(model, observer, *, shuffled):
    for case in training_cases(shuffled=shuffled):
        model.reset_state()
        for state in sensory_sequence(observer, case, hybrid=False):
            model.step(state, learn=True)


@torch.no_grad()
def score(model, observer, episodes, *, hybrid, multi=False, noise=None):
    hits = {8: [], 32: [], 64: []}
    for index, original in enumerate(episodes):
        case = (original if noise is None else
                {**original, 'events': corrupt_events(
                    original['events'], seed=5000+index,
                    dropout=noise[0], false_rate=noise[1])})
        model.reset_state()
        states = sensory_sequence(observer, case, hybrid=hybrid)
        for state in states[:case['decision']+1]:
            field = model.step(state)
        masks = case['hidden_by_entity'] if multi else [case['hidden']]
        for mask in masks:
            for k in hits:
                hits[k].append(hidden_rank(field,
                                           mask[case['decision']], k=k))
    return dict(cases=len(episodes), targets=len(hits[32]),
                top_k={str(k): sum(values)/len(values)
                       for k, values in hits.items()})


@torch.no_grad()
def score_disappearance(model, observer, episodes, *, hybrid):
    predicted, continued, vanished = [], [], []
    for case in episodes:
        gone = render_context_case(direction=case['direction'],
                                   speed=case['speed'], y=case['y'],
                                   shape=case['shape'],
                                   cue_sign=case['cue_sign'],
                                   background=case['background'],
                                   disappear=True)
        fields = []
        for episode in (case, gone):
            model.reset_state()
            snapshots = {}
            for frame, state in enumerate(sensory_sequence(
                    observer, episode, hybrid=hybrid)):
                field = model.step(state)
                if frame in (case['decision'], case['reveal']+2):
                    snapshots[frame] = field
            fields.append(snapshots)
        decision, reveal = case['decision'], case['reveal']+2
        predicted.append(float((fields[0][decision]-fields[1][decision])
                               .abs().max()))
        visible = case['visible_objects'][reveal]
        continued.append(float(fields[0][reveal][visible].max()))
        vanished.append(float(fields[1][reveal][visible].max()))
    return dict(cases=len(episodes),
                maximum_pre_reveal_pair_difference=max(predicted),
                continued_target_mean=sum(continued)/len(continued),
                vanished_false_target_mean=sum(vanished)/len(vanished),
                continued_above_half=sum(x >= .5 for x in continued),
                vanished_above_half=sum(x >= .5 for x in vanished))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    observer = fit_observers([
        scene_events(seed, return_contrast=True)
        for seed in range(64)])['aligned']
    models = dict(aligned=LocalContextMemory(),
                  shuffled=LocalContextMemory(),
                  frozen=LocalContextMemory())
    train(models['aligned'], observer, shuffled=False)
    train(models['shuffled'], observer, shuffled=True)
    results = dict(training_cases=len(training_cases(shuffled=False)),
                   learned={name: dict(effects=model.effects.tolist(),
                                       counts=model.counts.tolist())
                            for name, model in models.items()}, arms={})
    groups = dict(context=(context_cases(), False),
                  constant_motion=(cases(heldout=True), False),
                  two_movers=(two_mover_cases(), True))
    for hybrid in (False, True):
        arm = {}
        for name, model in models.items():
            arm[name] = {group: score(model, observer, episodes,
                                      hybrid=hybrid, multi=multi)
                         for group, (episodes, multi) in groups.items()}
        results['arms']['hybrid' if hybrid else 'event_only'] = arm
        arm['aligned']['disappearance'] = score_disappearance(
            models['aligned'], observer, groups['context'][0], hybrid=hybrid)
    results['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

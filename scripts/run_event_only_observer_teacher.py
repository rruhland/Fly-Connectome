"""Compare event-local observer teaching with privileged intensity changes."""

import json
import time
from pathlib import Path

import torch

from event_only_observer_teacher import fit_event_only_observer
from generic_native_cadence import scene_events
from local_context_memory import LocalContextMemory
from local_observation_model import LocalObservationModel
from run_local_context_memory import (score, score_disappearance, train)
from run_local_observation_model import fit_observers
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-event-only-observer-teacher-results.json')


@torch.no_grad()
def credibility(observer, episodes):
    counts = dict(true=0, false=0, onset_true=0)
    mass = dict(true=0., false=0., onset_true=0.)
    for clean, noisy in episodes:
        observer.reset_state()
        for frame, (truth, raw) in enumerate(zip(clean, noisy)):
            filtered, _ = observer.step(raw)
            true = (raw > 0) & (truth > 0)
            false = (raw > 0) & (truth == 0)
            for name, mask in (('true', true), ('false', false)):
                counts[name] += int(mask.sum())
                mass[name] += float(filtered[mask].sum())
            if frame == 0:
                counts['onset_true'] += int(true.sum())
                mass['onset_true'] += float(filtered[true].sum())
    return dict(counts=counts, mean_strength={
        key: mass[key]/max(counts[key], 1) for key in counts})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    observed = [corrupt_events(clean, seed=10000+index)
                for index, (clean, _) in enumerate(original)]
    observers = dict(privileged=fit_observers(original)['aligned'],
                     event_only=fit_event_only_observer(observed),
                     untrained=LocalObservationModel())
    heldout = [scene_events(seed, heldout=True)
               for seed in (2000, 2001, 2002, 2003)]
    groups = {name: [(clean, corrupt_events(
        clean, seed=seed, dropout=dropout, false_rate=false_rate))
                     for seed, clean in zip((2000, 2001, 2002, 2003),
                                            heldout)]
              for name, dropout, false_rate in (
                  ('known_noise', .1, .001),
                  ('heavier_noise', .2, .002))}
    results = dict(training_episodes=len(original), observers={})
    for name, observer in observers.items():
        row = dict(credibility={group: credibility(observer, episodes)
                                for group, episodes in groups.items()})
        if name != 'untrained':
            row['context_state'] = {}
            for threshold_name, threshold in (('original', .05),
                                              ('strict', .4)):
                model = LocalContextMemory(
                    minimum_event_strength=threshold)
                train(model, observer, shuffled=False)
                row['context_state'][threshold_name] = dict(
                    learned_effects=model.effects.tolist(),
                    context=score(model, observer, context_cases(),
                                  hybrid=False),
                    noisy_context={level: {
                        'event_only': score(model, observer,
                                            context_cases(), hybrid=False,
                                            noise=noise),
                        'hybrid_sparse_frames': score(
                            model, observer, context_cases(), hybrid=True,
                            noise=noise)}
                        for level, noise in (('known_noise', (.1, .001)),
                                             ('heavier_noise', (.2, .002)))},
                    constant_motion=score(model, observer,
                                          cases(heldout=True),
                                          hybrid=False),
                    two_movers=score(model, observer,
                                     two_mover_cases(), hybrid=False,
                                     multi=True),
                    disappearance=score_disappearance(
                        model, observer, context_cases(), hybrid=False))
        results['observers'][name] = row
    results['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

"""One broad causal-coherence state experiment at the frozen M1A.5 gate."""

import json
import time
from pathlib import Path

import torch

from corroborated_context_memory import CorroboratedContextMemory
from event_only_observer_teacher import fit_event_only_observer
from generic_native_cadence import scene_events
from run_local_context_memory import (score, score_disappearance, train)
from run_local_observation_model import fit_observers
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-corroborated-context-memory-results.json')


@torch.no_grad()
def evaluate(model, observer):
    context = context_cases()
    return dict(effects=model.effects.tolist(), counts=model.counts.tolist(),
                clean=score(model, observer, context, hybrid=False),
                known_noise=score(model, observer, context,
                                  hybrid=False, noise=(.1, .001)),
                heavy_noise=score(model, observer, context,
                                  hybrid=False, noise=(.2, .002)),
                known_noise_hybrid=score(model, observer, context,
                                         hybrid=True, noise=(.1, .001)),
                constant_motion=score(model, observer,
                                      cases(heldout=True), hybrid=False),
                two_movers=score(model, observer,
                                 two_mover_cases(), hybrid=False,
                                 multi=True),
                disappearance=score_disappearance(model, observer,
                                                  context, hybrid=False))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    original = [scene_events(seed, return_contrast=True)
                for seed in range(64)]
    noisy = [corrupt_events(clean, seed=10000+index)
             for index, (clean, _) in enumerate(original)]
    observers = dict(event_only=fit_event_only_observer(noisy),
                     privileged=fit_observers(original)['aligned'])
    results = dict(training_episodes=len(original), arms={})
    for name, observer in observers.items():
        aligned = CorroboratedContextMemory()
        train(aligned, observer, shuffled=False)
        results['arms'][name] = dict(aligned=evaluate(aligned, observer))
        if name == 'event_only':
            shuffled = CorroboratedContextMemory()
            train(shuffled, observer, shuffled=True)
            results['arms'][name]['shuffled'] = evaluate(shuffled, observer)
            results['arms'][name]['frozen'] = evaluate(
                CorroboratedContextMemory(), observer)
    results['seconds'] = round(time.perf_counter()-start, 2)
    OUT.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

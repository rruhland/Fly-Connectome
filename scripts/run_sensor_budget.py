"""Hold learned state fixed while comparing available appearance evidence."""

import json
import time
from functools import partial
from pathlib import Path

import torch

from event_only_observer_teacher import fit_event_only_observer
from evidence_state import EvidenceContextMemory
from generic_native_cadence import scene_events
from run_local_context_memory import score, score_disappearance, train
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence


OUT = Path('docs/experiments/2026-09-26-sensor-budget-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    episodes = [corrupt_events(scene_events(seed, return_contrast=True)[0],
                               seed=10000+seed) for seed in range(64)]
    observer = fit_event_only_observer(episodes)
    results = dict(frame_noise='none (synthetic sensor ceiling)', arms={})
    for period in (8, 1):
        sequence = partial(anchored_sequence, period=period)
        arm = {}
        for credit in ('aligned', 'shuffled', 'frozen'):
            model = EvidenceContextMemory(representation='surface')
            arm_start = time.perf_counter()
            if credit != 'frozen':
                train(model, observer, shuffled=credit == 'shuffled',
                      sequence=sequence)
            kwargs = dict(hybrid=False, sequence=sequence)
            context = context_cases()
            arm[credit] = dict(
                effects=model.effects.tolist(), counts=model.counts.tolist(),
                clean=score(model, observer, context, **kwargs),
                known_noise=score(model, observer, context,
                                  noise=(.1, .001), **kwargs),
                heavy_noise=score(model, observer, context,
                                  noise=(.2, .002), **kwargs),
                constant_motion=score(model, observer, cases(heldout=True),
                                      **kwargs),
                two_movers=score(model, observer, two_mover_cases(), multi=True,
                                 **kwargs),
                disappearance=score_disappearance(model, observer, context,
                                                  **kwargs),
                seconds=round(time.perf_counter()-arm_start, 2))
            results['arms'][str(period)] = dict(
                frame_pixel_samples_per_camera_step=2048/period, **arm)
            results['seconds'] = round(time.perf_counter()-start, 2)
            OUT.write_text(json.dumps(results, indent=2)+'\n')
            print(period, credit, json.dumps(arm[credit]), flush=True)


if __name__ == '__main__':
    main()

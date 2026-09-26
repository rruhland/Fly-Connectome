"""Frozen broad screen of two evidence/association representations."""

import json
import time
from pathlib import Path

import torch

from event_only_observer_teacher import fit_event_only_observer
from evidence_state import EvidenceContextMemory
from generic_native_cadence import scene_events
from run_corroborated_context_memory import evaluate
from run_local_context_memory import train
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-evidence-state-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    episodes = [corrupt_events(scene_events(seed, return_contrast=True)[0],
                               seed=10000+seed) for seed in range(64)]
    observer = fit_event_only_observer(episodes)
    results = dict(observer_teacher='causal event-only', arms={})
    for representation in ('events', 'surface'):
        arm = {}
        for credit in ('aligned', 'shuffled', 'frozen'):
            model = EvidenceContextMemory(representation=representation)
            arm_start = time.perf_counter()
            if credit != 'frozen':
                train(model, observer, shuffled=credit == 'shuffled')
            arm[credit] = evaluate(model, observer)
            arm[credit]['seconds'] = round(time.perf_counter()-arm_start, 2)
            results['arms'][representation] = arm
            results['seconds'] = round(time.perf_counter()-start, 2)
            OUT.write_text(json.dumps(results, indent=2)+'\n')
            print(representation, credit, json.dumps(arm[credit]), flush=True)


if __name__ == '__main__':
    main()

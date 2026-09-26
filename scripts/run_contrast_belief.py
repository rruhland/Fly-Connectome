"""One contrast-transition uncertainty experiment with a frozen teacher."""

import json
import time
from pathlib import Path

import torch

from contrast_belief import BeliefObserver
from event_only_observer_teacher import fit_event_only_observer
from evidence_state import EvidenceContextMemory
from generic_native_cadence import scene_events
from run_corroborated_context_memory import evaluate
from run_local_context_memory import train
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-contrast-belief-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    episodes = [corrupt_events(scene_events(seed, return_contrast=True)[0],
                               seed=10000+seed) for seed in range(64)]
    observer = BeliefObserver(fit_event_only_observer(episodes))
    results = dict(teacher='unchanged causal event agreement', arms={})
    for credit in ('aligned', 'shuffled', 'frozen'):
        model = EvidenceContextMemory(representation='surface')
        if credit != 'frozen':
            train(model, observer, shuffled=credit == 'shuffled')
        results['arms'][credit] = evaluate(model, observer)
        results['seconds'] = round(time.perf_counter()-start, 2)
        OUT.write_text(json.dumps(results, indent=2)+'\n')
        print(credit, json.dumps(results['arms'][credit]), flush=True)


if __name__ == '__main__':
    main()

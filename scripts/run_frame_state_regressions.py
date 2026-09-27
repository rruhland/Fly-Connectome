"""Recheck contextual learning after the full-frame observation revision."""

import json
from functools import partial
from pathlib import Path

import torch

from appearance_relation import appearance_case
from associative_patch_state import PatchAssociation
from event_only_observer_teacher import fit_event_only_observer
from frame_observation_state import FrameObservationState
from generic_native_cadence import scene_events
from run_associative_patch_state import train_memory, rotate_case, observer_for, disappearance
from run_local_context_memory import score
from run_visual_context_controls import context_cases
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    sequence = partial(anchored_sequence, period=1)
    results = {}
    out = Path('docs/experiments/2026-09-26-frame-state-regressions-results.json')
    for name, transform in (('spatial', None), ('appearance', appearance_case)):
        results[name] = {}
        for credit in ('aligned', 'shuffled', 'frozen'):
            memory = (PatchAssociation() if credit == 'frozen' else train_memory(base,
                sequence, shuffled=credit == 'shuffled', transform=transform,
                state_type=FrameObservationState))
            results[name][credit] = {}
            for turns in range(4):
                observer = observer_for(base, turns)
                state = FrameObservationState(height=observer.height, width=observer.width,
                                               memory=memory)
                episodes = [rotate_case(transform(c) if transform else c, turns)
                            for c in context_cases()]
                row = score(state, observer, episodes, hybrid=False, sequence=sequence)
                if name == 'spatial' and credit == 'aligned':
                    row['disappearance'] = disappearance(state, observer, sequence, turns)
                results[name][credit][str(turns*90)] = row
                out.write_text(json.dumps(results, indent=2)+'\n')
                print(name, credit, turns*90, row, flush=True)


if __name__ == '__main__':
    main()

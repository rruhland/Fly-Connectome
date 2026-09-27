"""Bounded appearance-coded dynamics test of the generic association state."""

import json
import time
from functools import partial
from pathlib import Path

import torch

from appearance_relation import appearance_case
from associative_patch_state import AssociativePatchState, PatchAssociation
from event_only_observer_teacher import fit_event_only_observer
from generic_native_cadence import scene_events
from run_associative_patch_state import observer_for, rotate_case, train_memory
from run_local_context_memory import score
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence


OUT = Path('docs/experiments/2026-09-26-appearance-relation-results.json')


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    sequence = partial(anchored_sequence, period=1)
    results = dict(sensor='visible frames plus events', arms={})
    for credit in ('aligned', 'shuffled', 'frozen'):
        memory = (PatchAssociation() if credit == 'frozen' else train_memory(
            base, sequence, shuffled=credit == 'shuffled', transform=appearance_case))
        row = dict(examples=len(memory.keys), orientations={})
        for turns in range(4):
            observer = observer_for(base, turns)
            state = AssociativePatchState(height=observer.height, width=observer.width,
                                           memory=memory)
            kwargs = dict(hybrid=False, sequence=sequence)
            row['orientations'][str(turns*90)] = dict(
                context=score(state, observer, [rotate_case(appearance_case(case), turns)
                                              for case in context_cases()], **kwargs),
                constant_motion=score(state, observer,
                    [rotate_case(case, turns) for case in cases(heldout=True)], **kwargs),
                two_movers=score(state, observer,
                    [rotate_case(case, turns) for case in two_mover_cases()], multi=True, **kwargs))
            results['arms'][credit] = row
            results['seconds'] = round(time.perf_counter()-started, 2)
            OUT.write_text(json.dumps(results, indent=2)+'\n')
            print(credit, turns*90, json.dumps(row['orientations'][str(turns*90)]), flush=True)


if __name__ == '__main__':
    main()

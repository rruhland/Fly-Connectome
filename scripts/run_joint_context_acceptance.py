"""Joint experience must retain both learned contextual relations."""

import json
from functools import partial
from pathlib import Path

import torch

from appearance_relation import appearance_case
from associative_patch_state import PatchAssociation
from event_only_observer_teacher import fit_event_only_observer
from frame_observation_state import FrameObservationState
from generic_native_cadence import scene_events
from run_associative_patch_state import train_memory, rotate_case, observer_for
from run_local_context_memory import score
from run_visual_context_controls import context_cases
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence


@torch.no_grad()
def fit_joint_memory(base, *, memory_type=PatchAssociation):
    sequence = partial(anchored_sequence, period=1)
    memories = [train_memory(base, sequence, transform=transform,
                             state_type=FrameObservationState)
                for transform in (None, appearance_case)]
    memory = memory_type()
    # Interleave observed experience; no benchmark labels enter the associative rule.
    for index in range(max(len(m.keys) for m in memories)):
        for source in memories:
            if index < len(source.keys):
                memory.observe(source.keys[index], source.values[index])
    return memory


@torch.no_grad()
def main(*, memory_type=PatchAssociation,
         out=Path('docs/experiments/2026-09-26-joint-context-results.json')):
    torch.set_num_threads(1)
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    sequence = partial(anchored_sequence, period=1)
    memory = fit_joint_memory(base, memory_type=memory_type)
    results = dict(examples=len(memory.keys), relations={})
    for name, transform in (('spatial', None), ('appearance', appearance_case)):
        results['relations'][name] = {}
        for turns in range(4):
            observer = observer_for(base, turns)
            state = FrameObservationState(height=observer.height, width=observer.width,
                                           memory=memory)
            cases = [rotate_case(transform(c) if transform else c, turns) for c in context_cases()]
            results['relations'][name][str(turns*90)] = score(state, observer, cases,
                hybrid=False, sequence=sequence)
    out.write_text(
        json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()

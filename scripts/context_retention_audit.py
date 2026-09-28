"""One deterministic production context-retention screen, separate from dynamics."""

import copy
import json
from pathlib import Path

import torch

from fly_connectome.vision import ProbabilisticVisualState, load_legacy_default as load_default
from fly_connectome.vision.observation import LocalObservationModel
from appearance_relation import appearance_case
from run_associative_patch_state import rotate_case
from run_local_context_memory import training_cases
from run_visual_context_controls import context_cases
from visual_history_scoring import hidden_rank


def model_at(base, memory, turns):
    height, width = (64, 32) if turns % 2 else (32, 64)
    observer = LocalObservationModel(height=height, width=width)
    observer.weights.copy_(base.observer.weights)
    observer.bias.copy_(base.observer.bias)
    return ProbabilisticVisualState(height=height, width=width, observer=observer,
                                   memory=memory, dynamics={})


def score(base, memory):
    result = {}
    for name, transform in (('spatial', lambda c: c), ('appearance', appearance_case)):
        hits = []
        for turns in range(4):
            model = model_at(base, memory, turns)
            for original in context_cases():
                case = rotate_case(transform(original), turns)
                model.reset_state()
                for t in range(case['decision']+1):
                    state = model.step(case['events'][t], case['visible'][t])
                hits.append(hidden_rank(state['support_field'], case['hidden'][t], k=32))
        result[name] = dict(hits=sum(hits), total=len(hits))
    return result


@torch.no_grad()
def main(*, factory=load_default, out=Path('docs/experiments/2026-09-27-context-retention-results.json')):
    torch.set_num_threads(1)
    base = factory()
    memory = copy.deepcopy(base.state.memory)
    result = dict(before=score(base, memory), stages=[])
    # Whole scene training, through actual reappearance; no hidden target enters step.
    for name, transform in (('appearance', appearance_case), ('spatial', lambda c: c)):
        original_keys = memory.keys.clone()
        samples = 0
        for turns in range(4):
            model = model_at(base, memory, turns)
            for original in training_cases(shuffled=False):
                case = rotate_case(transform(original), turns)
                model.reset_state()
                for event, image in zip(case['events'], case['visible']):
                    model.step(event, image, learn=True)
                    samples += 1
        result['stages'].append(dict(domain=name, samples=samples, retained=score(base, memory),
            keys_changed=not torch.equal(original_keys, memory.keys), examples=len(memory.keys)))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

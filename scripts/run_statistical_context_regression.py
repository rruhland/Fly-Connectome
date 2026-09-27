"""Joint relation retention after replacing the observation association stage."""

import copy
import json
from pathlib import Path

import torch

from appearance_relation import appearance_case
from local_metric_association import LocalMetricAssociation
from run_associative_patch_state import rotate_case
from run_integrated_visual_candidate import observer_at
from run_visual_context_controls import context_cases
from statistical_visual_state import StatisticalVisualCandidate
from streaming_visual_state import StreamingVisualState
from visual_history_scoring import hidden_rank


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    result = {}
    for arm in ('learned', 'shuffled', 'frozen'):
        memory = copy.deepcopy(source.state.memory)
        if arm == 'shuffled':
            permutation = torch.randperm(len(memory.values), generator=torch.Generator().manual_seed(199001))
            memory.values = memory.values[permutation]
        elif arm == 'frozen':
            memory = LocalMetricAssociation()
        result[arm] = {}
        for name, transform in (('spatial', lambda c: c), ('appearance', appearance_case)):
            result[arm][name] = {}
            for turns in range(4):
                height, width = (64, 32) if turns % 2 else (32, 64)
                model = StatisticalVisualCandidate(height=height, width=width,
                    observer=observer_at(source.observer, height, width), memory=memory)
                hits = []
                for original in context_cases():
                    case = rotate_case(transform(original), turns)
                    model.reset_state()
                    for t in range(case['decision']+1):
                        state = model.step(case['events'][t], case['visible'][t])
                    hits.append(hidden_rank(state['support_field'], case['hidden'][t], k=32))
                result[arm][name][str(turns*90)] = dict(hits=sum(hits), total=len(hits))
    Path('docs/experiments/2026-09-26-statistical-context-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

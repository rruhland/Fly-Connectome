"""Matched measured T4/T5 inputs versus zeros in local contextual learning."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from appearance_relation import appearance_case
from frame_observation_state import FrameObservationState
from local_metric_association import LocalMetricAssociation
from local_observation_model import LocalObservationModel
from measured_motion_adapter import MeasuredMotionAdapter
from run_associative_patch_state import rotate_case
from run_local_context_memory import training_cases
from run_visual_context_controls import context_cases
from sensor_budget import anchored_sequence
from visual_history_scoring import hidden_rank


class MeasuredAssociationState(FrameObservationState):
    def key(self, sensory, slot):
        key = super().key(sensory, slot)
        y, x = (round(float(v)) for v in slot['position'])
        stage = sensory[12:20, max(0, y-10):y+11, max(0, x-10):x+11].mean((1, 2))
        return torch.cat((key, stage))


def resized(case, turns):
    result = rotate_case(case, turns)
    for name in ('events', 'visible', 'hidden'):
        result[name] = [F.interpolate(value.float().reshape(1, -1, *value.shape[-2:]),
                       size=(32, 64), mode='nearest')[0].reshape(-1, 32, 64)
                       for value in result[name]]
        if name != 'events':
            result[name] = [value[0].bool() for value in result[name]]
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    start = time.perf_counter()
    adapter = MeasuredMotionAdapter()
    states = {name: MeasuredAssociationState(memory=LocalMetricAssociation(dimensions=59))
              for name in ('ablated', 'measured')}
    result = dict(source_sha=adapter.graph['source_sha'], graph_sha=adapter.graph['graph_sha'],
                  scores={}, active_samples=0, samples=0)
    out = Path('docs/experiments/2026-09-26-measured-state-ablation-results.json')
    for training in (True, False):
        for relation, transform in (('spatial', lambda case: case), ('appearance', appearance_case)):
            for turns in (0, 1):
                cases = ([c for c in training_cases(shuffled=False) if c['speed'] == 2]
                         if training else context_cases())
                hits = {name: [] for name in states}
                for original in cases:
                    case = resized(transform(original), turns)
                    adapter.reset_state()
                    for state in states.values():
                        state.reset_state()
                    sensory = anchored_sequence(LocalObservationModel(), case, period=1)
                    stop = len(sensory) if training else case['decision']+1
                    for t in range(stop):
                        stage = adapter.step(case['events'][t])
                        result['samples'] += 1
                        result['active_samples'] += int(stage.any())
                        for name, state in states.items():
                            field = state.step(torch.cat((sensory[t], stage if name == 'measured'
                                                           else torch.zeros_like(stage))), learn=training)
                            if not training and t == case['decision']:
                                hits[name].append(hidden_rank(field, case['hidden'][t], k=32))
                if not training:
                    result['scores'][f'{relation}_{turns*90}'] = {name: dict(hits=sum(v), total=len(v))
                                                               for name, v in hits.items()}
                result['seconds'] = time.perf_counter()-start
                out.write_text(json.dumps(result, indent=2)+'\n')
                print(training, relation, turns, result['seconds'], result['scores'], flush=True)
    adapter.verify_frozen()
    result['frozen_graph_verified'] = True
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

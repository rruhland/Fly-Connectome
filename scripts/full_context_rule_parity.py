"""Full-graph equality of neural and causal eligibility state before learning."""
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from frame_prediction import FramePrediction
from full_context_efficacy import MultiContextEfficacyNetwork, MultiContextTimedPrediction
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig
from fly_connectome.pong import Physics, Pong
from fly_connectome.sensor import EventCamera, Retina


def main():
    torch.set_num_threads(4)
    source = Path('checkpoints/event-v1-combined-rate-initial.pt')
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    kwargs = dict(batch=1, config=NeuronConfig(**metadata['neurons']),
                  cell_types=metadata['retina']['cell_types'])
    reference = AreaMatchedKineticsNetwork(graph, metadata['delays'], metadata['pathways'], **kwargs)
    candidate = MultiContextEfficacyNetwork(graph, metadata['delays'], metadata['pathways'],
        target_mask=retina.injected, **kwargs)
    source_weights = payload['state']['network']['magnitudes']
    maximum = metadata['learning']['maximum_weight']
    out_of_bounds_source_edges = int(((source_weights < 0) | (source_weights > maximum)).sum())
    weights = source_weights.clone().clamp_(0, maximum)
    reference.magnitudes.copy_(weights)
    candidate.set_weights(weights)
    config = replace(LearningConfig(**metadata['learning']), eta_prediction=0.,
                     eta_reward=0., homeostasis_rate=0.)
    rule_kwargs = dict(sensory_mask=retina.injected,
                       sensory_gain=metadata['config']['sensory_gain'])
    old_rule = FramePrediction(reference, config, **rule_kwargs)
    new_rule = MultiContextTimedPrediction(candidate, config, **rule_kwargs)
    camera = EventCamera(1, retina.spec['height'], retina.spec['width'])
    pong = Pong([1101], Physics(**metadata['physics']))
    frames = 50
    max_keys = 0
    forecast_edges = 0
    open_gates = 0
    started = time.perf_counter()
    for frame in range(frames):
        image = pong.render(retina.spec['height'], retina.spec['width'])
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            sensory = injection if tick == 0 else torch.zeros_like(injection)
            old = reference.step(sensory, capture_increments=True)
            new = candidate.step(sensory, capture_increments=True)
            if not torch.equal(old.spikes, new.spikes) or not torch.equal(old.predicted, new.predicted):
                raise AssertionError(f'frame {frame} tick {tick}: neural parity')
            old_rule.observe(old, torch.zeros(1))
            new_rule.observe(new, torch.zeros(1))
            for field in ('keys', 'values', 'arrival_trace', 'post_trace', 'expected'):
                if not torch.equal(getattr(old_rule, field), getattr(new_rule, field)):
                    raise AssertionError(f'frame {frame} tick {tick}: {field}')
            max_keys = max(max_keys, len(new_rule.keys))
            if tick == 0:
                keys, eligibility, prediction = old_rule.forecast
                edges = keys.remainder(reference.e)
                positions = candidate.target_lookup[candidate.post[edges]]
                keep = positions >= 0
                gated = new_rule.gate[positions[keep]].float()
                expected = keys[keep], eligibility[keep]*gated, prediction[keep]*gated
                for actual, value in zip(new_rule.forecast, expected):
                    if not torch.equal(actual, value):
                        raise AssertionError(f'frame {frame}: gated forecast capture')
                forecast_edges += len(new_rule.forecast[0])
                open_gates += int(new_rule.gate.sum())
        old_rule.synchronize()
        new_rule.synchronize()
        pong.step(-(pong.ball[:, 1]-pong.body.position)*10)
    torch.testing.assert_close(candidate.components, weights[candidate.incoming].repeat(2, 1),
                               rtol=0, atol=0)
    torch.testing.assert_close(candidate.magnitudes, weights, rtol=0, atol=0)
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(source=str(source), source_sha256=source_sha, graph_sha256=graph.identity(),
        bounded_source_edges=out_of_bounds_source_edges,
        frames=frames, ticks=8*frames, targets=len(candidate.targets), edges=len(candidate.incoming),
        max_active_eligibility_keys=max_keys, captured_target_edge_forecasts=forecast_edges,
        open_target_gates=open_gates, exact_fields=['spikes', 'predicted', 'keys', 'values',
            'arrival_trace', 'post_trace', 'expected', 'gated target forecast', 'weights'],
        seconds=time.perf_counter()-started, script_sha256=checksum(Path(__file__)))
    output = Path('runs/full-context-m1a-v1/rule-parity.json')
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

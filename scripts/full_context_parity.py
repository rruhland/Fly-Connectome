"""Exact equal-component neural parity on pinned full-graph Pong frames."""
import argparse
import json
from pathlib import Path
import time

import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.pong import Physics, Pong
from fly_connectome.sensor import EventCamera, Retina


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/event-v1-combined-rate-initial.pt')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--output', default='runs/full-context-m1a-v1/neural-parity.json')
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('positive parity frame count required')
    torch.set_num_threads(4)
    source = Path(args.checkpoint)
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    metadata = payload['metadata']
    if metadata['config']['neural_steps'] != 8:
        raise ValueError('the bounded bridge requires eight neural ticks per frame')
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    kwargs = dict(batch=1, config=NeuronConfig(**metadata['neurons']),
                  cell_types=metadata['retina']['cell_types'])
    reference = AreaMatchedKineticsNetwork(graph, metadata['delays'], metadata['pathways'], **kwargs)
    candidate = MultiContextEfficacyNetwork(graph, metadata['delays'], metadata['pathways'],
        target_mask=retina.injected, **kwargs)
    weights = payload['state']['network']['magnitudes']
    reference.magnitudes.copy_(weights)
    candidate.set_weights(weights)
    camera = EventCamera(1, retina.spec['height'], retina.spec['width'])
    pong = Pong([1101], Physics(**metadata['physics']))
    target_arrivals = 0
    context_counts = torch.zeros(2, dtype=torch.long)
    spikes = 0
    started = time.perf_counter()
    for frame in range(args.frames):
        image = pong.render(retina.spec['height'], retina.spec['width'])
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            input_current = injection if tick == 0 else torch.zeros_like(injection)
            old = reference.step(input_current, capture_increments=True)
            new = candidate.step(input_current, capture_increments=True)
            for field in ('spikes', 'observed', 'predicted', 'arrival_environments',
                          'arrival_edges', 'feedforward_arrivals', 'sensory_input'):
                if not torch.equal(getattr(old, field), getattr(new, field)):
                    raise AssertionError(f'frame {frame} tick {tick}: {field} mismatch')
            for field in ('voltage', 'sensory_state', 'predictive_current',
                          'excitatory_prediction', 'inhibitory_prediction',
                          'adaptation', 'refractory', 'history'):
                if not torch.equal(getattr(reference, field), getattr(candidate, field)):
                    raise AssertionError(f'frame {frame} tick {tick}: {field} mismatch')
            target_arrivals += int((candidate.incoming_lookup[new.arrival_edges] >= 0).sum())
            context_counts += torch.bincount(candidate.current_context.long().flatten(), minlength=2)
            spikes += int(new.spikes.sum())
        pong.step(-(pong.ball[:, 1]-pong.body.position)*10)
    torch.testing.assert_close(candidate.components,
                               weights[candidate.incoming].repeat(2, 1), rtol=0, atol=0)
    torch.testing.assert_close(candidate.magnitudes, weights, rtol=0, atol=0)
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    report = dict(source=str(source), source_sha256=source_sha, graph_sha256=graph.identity(),
        frames=args.frames, ticks=args.frames*8, seed=1101, neurons=candidate.n,
        measured_edges=candidate.e, context_targets=len(candidate.targets),
        context_edges=len(candidate.incoming), target_arrivals=target_arrivals,
        context_counts=context_counts.tolist(), spikes=spikes,
        exact_fields=['spikes', 'observed', 'predicted', 'arrival_environments',
                      'arrival_edges', 'feedforward_arrivals', 'sensory_input',
                      'voltage', 'sensory_state', 'predictive_current',
                      'excitatory_prediction', 'inhibitory_prediction',
                      'adaptation', 'refractory', 'history', 'weights'],
        seconds=time.perf_counter()-started, script_sha256=checksum(Path(__file__)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Capped full-graph M1A throughput and state benchmark, no checkpoint write."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from full_context_efficacy import MultiContextEfficacyNetwork, MultiContextTimedPrediction
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig
from fly_connectome.pong import Physics, Pong
from fly_connectome.sensor import EventCamera, Retina


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--eta', type=float, default=1.)
    parser.add_argument('--output', default='runs/full-context-m1a-v1/benchmark-50.json')
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('positive frame count required')
    torch.set_num_threads(4)
    source = Path('checkpoints/event-v1-combined-rate-initial.pt')
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    config = replace(LearningConfig(**metadata['learning']), eta_prediction=args.eta,
                     eta_reward=0., homeostasis_rate=0.)
    source_weights = payload['state']['network']['magnitudes']
    weights = source_weights.clone().clamp_(0, config.maximum_weight)
    net = MultiContextEfficacyNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']), cell_types=metadata['retina']['cell_types'],
        target_mask=retina.injected)
    net.set_weights(weights)
    rule = MultiContextTimedPrediction(net, config, sensory_mask=retina.injected,
        sensory_gain=metadata['config']['sensory_gain'])
    camera = EventCamera(1, retina.spec['height'], retina.spec['width'])
    pong = Pong([1101], Physics(**metadata['physics']))
    phases = dict(camera=0., neural=0., plasticity=0., synchronization=0.)
    max_keys = 0
    spikes = 0
    started = time.perf_counter()
    for _ in range(args.frames):
        t = time.perf_counter()
        image = pong.render(retina.spec['height'], retina.spec['width'])
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        phases['camera'] += time.perf_counter()-t
        for tick in range(8):
            t = time.perf_counter()
            activity = net.step(injection if tick == 0 else torch.zeros_like(injection),
                                capture_increments=True)
            phases['neural'] += time.perf_counter()-t
            t = time.perf_counter()
            rule.observe(activity, torch.zeros(1))
            phases['plasticity'] += time.perf_counter()-t
            max_keys = max(max_keys, len(rule.keys))
            spikes += int(activity.spikes.sum())
        t = time.perf_counter()
        rule.synchronize()
        pong.step(-(pong.ball[:, 1]-pong.body.position)*10)
        phases['synchronization'] += time.perf_counter()-t
    seconds = time.perf_counter()-started
    if not torch.isfinite(net.voltage).all() or not torch.isfinite(net.components).all():
        raise ValueError('nonfinite neural state or context magnitude')
    outside = torch.ones(net.e, dtype=torch.bool); outside[net.incoming] = False
    torch.testing.assert_close(net.magnitudes[outside], weights[outside], rtol=0, atol=0)
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(source_sha256=source_sha, graph_sha256=graph.identity(), frames=args.frames,
        ticks=args.frames*8, eta=args.eta, seconds=seconds, camera_frames_per_second=args.frames/seconds,
        phase_seconds=phases, max_active_eligibility_keys=max_keys, spikes=spikes,
        target_magnitudes_changed=int((net.components != weights[net.incoming].repeat(2, 1)).sum()),
        non_target_magnitudes_unchanged=True,
        source_edges_clipped_at_initialization=int((source_weights != weights).sum()),
        script_sha256=checksum(Path(__file__)))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

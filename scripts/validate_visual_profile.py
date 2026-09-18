"""Fixed non-Pong validation suite for a frozen visual checkpoint, with no plasticity."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import Network
from fly_connectome.sensor import EventCamera
from fly_connectome.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = load_checkpoint(args.checkpoint, evaluation=True, seeds=[1003], warmup=False)
    conditions = [dict(kind='edge', direction=d, speed=s, polarity=p)
                  for d, s, p in itertools.product((0, 180), (.04, .16), ('on', 'off'))]
    conditions += [dict(kind='flash', polarity=p) for p in ('on', 'off')]
    conditions += [dict(kind='no_event', polarity='on')]
    types = source.retina.spec['cell_types']
    net = Network(source.network.graph, source.network.delays.tolist(),
        [('feedforward', 'predictive', 'behavioral')[p] for p in source.network.pathways.tolist()],
        config=source.network.config, batch=len(conditions), cell_types=types)
    net.magnitudes.copy_(source.network.magnitudes)
    warmup, stimulus, recovery = 500, 400, 200
    labels = ('L1','L2','L3','L4','Mi1','Tm3','Tm1','Tm2','Mi9','T4','T5','LPi','LC10')
    masks = {label: torch.tensor([t == label or (label in ('T4','T5','LPi','LC10') and t.startswith(label)) for t in types]) for label in labels}
    camera = EventCamera(len(conditions), 32, 64)
    background = torch.tensor([c['polarity'] == 'off' for c in conditions])[:,None,None].expand(-1,32,64).clone()
    camera.previous.copy_(background)
    baseline_counts = torch.zeros(len(conditions), net.n, dtype=torch.int64)
    phase_counts = [torch.zeros_like(baseline_counts) for _ in range(2)]
    bin_counts = torch.zeros_like(baseline_counts)
    traces, peak = [], 0
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    for tick in range(warmup + stimulus + recovery):
        elapsed = tick - warmup
        frame = background.clone()
        if 0 <= elapsed < stimulus:
            for index, condition in enumerate(conditions[:-1]):
                if condition['kind'] == 'edge':
                    coordinate = x if condition['direction'] == 0 else 63-x
                    foreground = coordinate < elapsed * condition['speed']
                else:
                    foreground = ((x-32)**2 + (y-16)**2 <= 4) & (elapsed < 100)
                frame[index] = foreground if condition['polarity'] == 'on' else ~foreground
        activity = net.step(source.retina.project(camera.observe(frame)) * source.config.sensory_gain)
        peak = max(peak, int(activity.spikes.sum(1).max()))
        if not torch.isfinite(net.voltage).all() or peak > net.n * .2:
            raise RuntimeError('validation exceeded preregistered numerical activity bound')
        if warmup//2 <= tick < warmup:
            baseline_counts += activity.spikes
        if elapsed >= 0:
            phase_counts[int(elapsed >= stimulus)] += activity.spikes
            bin_counts += activity.spikes
            if (elapsed+1) % 10 == 0:
                traces.append({label: (bin_counts[:,mask].float().mean(1)/(10*net.config.dt)).tolist()
                               for label, mask in masks.items() if mask.any()})
                bin_counts.zero_()
    populations = {label: dict(baseline_rate_hz=(baseline_counts[:,mask].float().mean(1)/((warmup-warmup//2)*net.config.dt)).tolist(),
        stimulus_counts=phase_counts[0][:,mask].sum(1).tolist(), recovery_counts=phase_counts[1][:,mask].sum(1).tolist())
        for label, mask in masks.items() if mask.any()}
    report = dict(checkpoint_sha256=checksum(args.checkpoint), graph_sha256=net.graph.identity(),
        neurons=asdict(net.config), conditions=conditions, seed=1003, backend='cpu',
        warmup_ticks=warmup, stimulus_ticks=stimulus, recovery_ticks=recovery,
        trace_bin_ticks=10, peak_spikes_per_tick=peak, populations=populations, rate_traces_hz=traces)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), peak_spikes_per_tick=peak, populations=populations)), flush=True)


if __name__ == '__main__':
    main()

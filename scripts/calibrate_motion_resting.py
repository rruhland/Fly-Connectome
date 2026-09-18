"""Test a preregistered subthreshold T4/T5 operating point on frozen visual probes."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import Network
from fly_connectome.sensor import EventCamera
from fly_connectome.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/motion-resting-calibration-v1.json')
    parser.add_argument('--output', default='runs/motion-resting-calibration-v1.json')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    source = load_checkpoint(spec['source_checkpoint'], evaluation=True, seeds=[spec['seed']], warmup=False)
    types = source.retina.spec['cell_types']
    labels = ('T4', 'T5', 'LPi', 'LC10')
    masks = {label: torch.tensor([t.startswith(label) for t in types]) for label in labels}
    rows = []
    for rest in spec['rest_currents']:
        params = {t: dict(p) for t,p in source.network.config.class_parameters.items()}
        params.update({t: {'rest_current': rest} for t in spec['classes']})
        cfg = replace(source.network.config, class_parameters=params)
        net = Network(source.network.graph, source.network.delays.tolist(),
            [('feedforward','predictive','behavioral')[p] for p in source.network.pathways.tolist()],
            batch=3, config=cfg, cell_types=types)
        net.magnitudes.copy_(source.network.magnitudes)
        camera = EventCamera(3, 32, 64)
        background = torch.tensor([False,True,False])[:,None,None].expand(-1,32,64).clone()
        camera.previous.copy_(background)
        counts = [torch.zeros(3, net.n, dtype=torch.int64) for _ in range(3)]
        x = torch.arange(64)[None,:].expand(32,-1)
        peak, stable = 0, True
        for tick in range(spec['warmup_ticks']+spec['stimulus_ticks']+spec['recovery_ticks']):
            elapsed = tick-spec['warmup_ticks']
            frame = background.clone()
            if 0 <= elapsed < spec['stimulus_ticks']:
                frame[0] = x < elapsed*spec['edge_pixels_per_tick']
                frame[1] = ~frame[0]
            activity = net.step(source.retina.project(camera.observe(frame))*source.config.sensory_gain)
            peak = max(peak, int(activity.spikes.sum(1).max()))
            stable = stable and bool(torch.isfinite(net.voltage).all()) and peak < net.n*spec['peak_spike_fraction_bound']
            if not stable:
                break
            if tick >= spec['warmup_ticks']//2:
                phase = 0 if elapsed < 0 else (1 if elapsed < spec['stimulus_ticks'] else 2)
                counts[phase] += activity.spikes
        populations = {label: dict(baseline_counts=counts[0][:,mask].sum(1).tolist(),
            stimulus_counts=counts[1][:,mask].sum(1).tolist(), recovery_counts=counts[2][:,mask].sum(1).tolist(),
            stimulus_changed_neurons=(counts[1][:2,mask] != counts[1][2,mask]).sum(1).tolist())
            for label,mask in masks.items()}
        qualifies = stable and all(all(v > 0 for v in populations[label]['stimulus_counts'][:2]) and
            all(v > 0 for v in populations[label]['stimulus_changed_neurons']) for label in ('T4','T5'))
        row = dict(rest_current=rest, stable=stable, qualifies=qualifies, peak_spikes_per_tick=peak,
                   populations=populations, neurons=asdict(cfg))
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'neurons'}), flush=True)
        Path(args.output).write_text(json.dumps(dict(spec=spec, config_sha256=checksum(args.config),
            checkpoint_sha256=checksum(spec['source_checkpoint']), graph_sha256=net.graph.identity(), rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()

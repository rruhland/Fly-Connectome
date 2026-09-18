"""Frozen input-current audit of measured T4/T5 pathways, outside training."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
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
    torch.set_num_threads(1)
    source = load_checkpoint(args.checkpoint, evaluation=True, seeds=[1003], warmup=False)
    types = np.asarray(source.retina.spec['cell_types'])
    net = Network(source.network.graph, source.network.delays.tolist(),
        [('feedforward', 'predictive', 'behavioral')[p] for p in source.network.pathways.tolist()],
        config=source.network.config, batch=3, cell_types=types.tolist())
    net.magnitudes.copy_(source.network.magnitudes)
    camera = EventCamera(3, 32, 64)
    background = torch.tensor([False, True, False])[:, None, None].expand(-1, 32, 64).clone()
    camera.previous.copy_(background)
    masks = {label: torch.tensor(np.char.startswith(types, label)) for label in ('T4', 'T5')}
    # Separate E/I state is evaluation instrumentation, using the very same arrivals.
    excitation, inhibition = torch.zeros_like(net.voltage), torch.zeros_like(net.voltage)
    counts = torch.zeros_like(net.voltage, dtype=torch.int64)
    stats = {label: dict(max_voltage=[-float('inf')]*3, max_excitation=[0.]*3,
                        min_inhibition=[0.]*3, max_net_current=[-float('inf')]*3,
                        summed_excitation=[0.]*3, summed_inhibition=[0.]*3) for label in masks}
    x = torch.arange(64)[None, :].expand(32, -1)
    for tick in range(1100):
        elapsed = tick - 500
        frame = background.clone()
        if 0 <= elapsed < 400:
            frame[0] = x < elapsed * .16
            frame[1] = ~frame[0]
        a = net.step(source.retina.project(camera.observe(frame)) * source.config.sensory_gain)
        indices = a.arrival_environments * net.n + net.post[a.arrival_edges]
        weights = net.magnitudes[a.arrival_edges] * net.signs[a.arrival_edges]
        excitation.mul_(net.current_decay)
        inhibition.mul_(net.current_decay)
        excitation.view(-1).index_add_(0, indices, weights.clamp(min=0))
        inhibition.view(-1).index_add_(0, indices, weights.clamp(max=0))
        if elapsed >= 0:
            counts += a.spikes
            for label, mask in masks.items():
                row = stats[label]
                for key, values, reduce in (
                    ('max_voltage', net.voltage[:, mask], 'max'),
                    ('max_excitation', excitation[:, mask], 'max'),
                    ('min_inhibition', inhibition[:, mask], 'min'),
                    ('max_net_current', (excitation+inhibition)[:, mask], 'max')):
                    prior = torch.tensor(row[key])
                    row[key] = (torch.maximum(prior, values.max(1).values) if reduce == 'max'
                                else torch.minimum(prior, values.min(1).values)).tolist()
                for key, values in (('summed_excitation', excitation), ('summed_inhibition', inhibition)):
                    row[key] = (torch.tensor(row[key]) + values[:, mask].sum(1)).tolist()
    torch.testing.assert_close(excitation+inhibition,
        net.feedforward_current+net.predictive_current+net.behavioral_current, atol=1e-5, rtol=1e-5)
    report = dict(checkpoint_sha256=checksum(args.checkpoint), graph_sha256=net.graph.identity(),
                  neurons=asdict(net.config), conditions=['on_edge', 'off_edge', 'no_event'],
                  warmup_ticks=500, stimulus_ticks=400, recovery_ticks=200, populations={})
    for label, mask in masks.items():
        edges = mask[net.post]
        pretypes = types[net.pre[edges].numpy()]
        partners = []
        for cell_type, edge_count in Counter(pretypes).most_common():
            edge_mask = edges & torch.tensor(types[net.pre.numpy()] == cell_type)
            presynaptic = net.pre[edge_mask].unique()
            partners.append(dict(cell_type=cell_type, edges=edge_count,
                contacts=int(net.graph.contacts[edge_mask.numpy()].sum()),
                signs=net.signs[edge_mask].unique().tolist(),
                presynaptic_spikes=counts[:, presynaptic].sum(1).tolist(),
                presynaptic_neurons=len(presynaptic)))
        report['populations'][label] = dict(neurons=int(mask.sum()), spikes=counts[:,mask].sum(1).tolist(),
            **stats[label], incoming_partners=partners)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({label: {k:v for k,v in row.items() if k != 'incoming_partners'}
                      for label,row in report['populations'].items()}), flush=True)


if __name__ == '__main__':
    main()

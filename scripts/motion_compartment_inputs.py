"""Read-only local T4 arm traces and fixed conductance-style scores."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/motion-stage-signal-integration-v1')
LABELS = ('T4a', 'T4b', 'T4c', 'T4d')


def frames(center, axis, direction=1, *, static=False):
    if axis not in ('horizontal', 'vertical') or direction not in (-1, 1):
        raise ValueError('axis and direction must be explicit')
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    images = []
    for frame in range(18):
        if 2 <= frame < 15:
            position = (center if axis == 'horizontal' else 16)
            if not static:
                position += direction*(frame-8)
            shape = (((x-position).abs() <= 1) & ((y-16).abs() <= 6)
                     if axis == 'horizontal' else
                     ((y-position).abs() <= 1) & ((x-center).abs() <= 6))
            images.append(shape.unsqueeze(0))
        else:
            images.append(torch.zeros(1, 32, 64, dtype=torch.bool))
    return images


def direction_index(right, left):
    return (right-left)/(right+left) if right+left > 0 else 0.


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    graph = Graph(**m['graph'], gain=m['gain'])
    net = MultiContextEfficacyNetwork(graph, m['delays'], m['pathways'],
        config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
        target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(0,
                    m['learning']['maximum_weight']))
    types = np.asarray(m['retina']['cell_types'])
    t4_nodes = torch.tensor(np.flatnonzero(np.isin(types, LABELS)))
    lookup = torch.full((net.n,), -1, dtype=torch.long)
    lookup[t4_nodes] = torch.arange(len(t4_nodes))
    edge_channel = torch.full((net.e,), -1, dtype=torch.long)
    for source, channel in (('Mi1', 0), ('Tm3', 0), ('Mi9', 1), ('Mi4', 2)):
        mask = (torch.tensor(types == source)[net.pre]
                & (lookup[net.post] >= 0) & (net.pathways == 0))
        edge_channel[mask] = channel
    if not (edge_channel == 2).any():
        raise AssertionError('measured Mi4->T4 arm missing')
    decay = net.current_decay[t4_nodes]
    zero = torch.zeros_like(net.voltage)
    trace = torch.zeros(3, len(t4_nodes))

    def update(activity, state):
        state.mul_(decay)
        edges = activity.arrival_edges
        channel = edge_channel[edges]
        keep = channel >= 0
        chosen = edges[keep]
        flat = channel[keep]*len(t4_nodes)+lookup[net.post[chosen]]
        state.view(-1).index_add_(0, flat, net.magnitudes[chosen])

    for _ in range(m['config']['warmup_steps']):
        update(net.step(zero), trace)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_trace = trace.clone()
    warm_tick = net.step_index

    def probe(images):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        state = warm_trace.clone()
        camera = EventCamera(1, 32, 64)
        values = []
        spikes = torch.zeros(len(t4_nodes), dtype=torch.int32)
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                update(activity, state)
                values.append(state.clone())
                if 3 <= frame < 15:
                    spikes.add_(activity.spikes[0, t4_nodes].int())
        return torch.stack(values), spikes

    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(m, retina, table)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    blank, blank_spikes = probe([torch.zeros(1, 32, 64, dtype=torch.bool)]*18)
    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS), graph_sha256=graph.identity(),
        trace_decay='existing per-T4 current_decay',
        score_names=('divisive', 'coincident_disinhibition'),
        groups={})
    for center in (18, 46):
        for axis in ('horizontal', 'vertical'):
            path = (((x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23))
                    if axis == 'horizontal' else
                    ((x >= center-7) & (x <= center+7) & (y >= 8) & (y <= 24)))
            bins = retina.pixel_bins[path.flatten()].unique().numpy()
            local = {label: torch.tensor((types[t4_nodes.numpy()] == label)
                                       & np.isin(columns[t4_nodes.numpy()], bins))
                     for label in LABELS}
            cases = {}
            for name, images in (
                ('right', frames(center, axis, 1)),
                ('left', frames(center, axis, -1)),
                ('static', frames(center, axis, static=True))):
                values, spikes = probe(images)
                E, I9, I4 = values[24:120].unbind(1)
                Eb, I9b, I4b = blank[24:120].unbind(1)
                divisive = (E/(1+I9+I4)-Eb/(1+I9b+I4b))
                coincident = E*(I9b-I9).clamp(min=0)
                cases[name] = dict(
                    positive_divisive=divisive.clamp(min=0).sum(0),
                    signed_divisive=divisive.sum(0),
                    coincident=coincident.sum(0),
                    excess_spikes=spikes-blank_spikes)
            row = report['groups'][f'{center}-{axis}'] = {}
            for label, mask in local.items():
                data = row[label] = dict(neurons=int(mask.sum()))
                for score in ('positive_divisive', 'signed_divisive', 'coincident'):
                    values = {name: float(case[score][mask].sum())
                              for name, case in cases.items()}
                    data[score] = dict(**values,
                        right_minus_left=values['right']-values['left'],
                        direction_index=direction_index(values['right'], values['left']))
                data['excess_spikes'] = {name: int(case['excess_spikes'][mask].sum())
                                         for name, case in cases.items()}
            print(json.dumps(dict(center=center, axis=axis, groups={label: {
                score: data[score] for score in ('positive_divisive', 'coincident')}
                for label, data in row.items()})), flush=True)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT/'compartment-inputs.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output))), flush=True)


if __name__ == '__main__':
    main()

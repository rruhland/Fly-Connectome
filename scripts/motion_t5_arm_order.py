"""Read-only causal T5 afferent-order diagnostic on measured edges."""

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS, infer_columns


OUT = Path('docs/experiments/2026-09-23-t5-arm-order-results.json')
ARMS = ('Tm1', 'Tm2', 'Tm4', 'Tm9')
SUBTYPES = ('T5c', 'T5d')
LAG = 8


def lag_order(first, second, lag):
    if first.shape != second.shape or first.ndim != 2 or not 0 < lag < len(first):
        raise ValueError('matching per-tick arm traces and a positive lag required')
    result = torch.zeros_like(first)
    result[lag:] = (first[:-lag]*second[lag:]
                    - second[:-lag]*first[lag:])
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = MultiContextEfficacyNetwork(graph, metadata['delays'],
        metadata['pathways'], config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    for cell_type in ('Tm4', 'Tm9'):
        net.rest_current[torch.tensor(types == cell_type)] = .85
    t5_nodes = torch.tensor(np.flatnonzero(
        np.char.startswith(types.astype(str), 'T5')))
    t5_lookup = torch.full((net.n,), -1, dtype=torch.long)
    t5_lookup[t5_nodes] = torch.arange(len(t5_nodes))
    edge_channel = torch.full((net.e,), -1, dtype=torch.long)
    for channel, cell_type in enumerate(ARMS):
        selected = (torch.tensor(types == cell_type)[net.pre]
                    & (t5_lookup[net.post] >= 0)
                    & (net.pathways == 0))
        edge_channel[selected] = channel
    if any(not (edge_channel == channel).any()
           for channel in range(len(ARMS))):
        raise AssertionError('missing measured T5 feedforward arm')
    arm_edges = (edge_channel >= 0).nonzero().flatten()
    if not (net.signs[arm_edges] == 1).all():
        raise AssertionError('T5 arm transmitter sign changed')

    decay = net.current_decay[t5_nodes]
    traces = torch.zeros(len(ARMS), len(t5_nodes))

    def update(activity, state):
        state.mul_(decay)
        edges = activity.arrival_edges
        channels = edge_channel[edges]
        chosen = edges[channels >= 0]
        flat = channels[channels >= 0]*len(t5_nodes)+t5_lookup[net.post[chosen]]
        state.view(-1).index_add_(0, flat,
            net.magnitudes[chosen]*net.signs[chosen])

    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        update(net.step(zero), traces)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_trace = traces.clone()
    warm_tick = net.step_index

    def capture(images):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        state = warm_trace.clone()
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        values = torch.empty((144, len(ARMS), len(t5_nodes)))
        spikes = torch.zeros(len(t5_nodes), dtype=torch.int32)
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                update(activity, state)
                values[step] = state
                if 3 <= frame < 15:
                    spikes += activity.spikes[0, t5_nodes].int()
                step += 1
        if not torch.isfinite(values).all():
            raise AssertionError('non-finite T5 arm state')
        return dict(traces=values, spikes=spikes,
                    pixel_events=pixel_events)

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, mass = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    local = {}
    for center in (18, 46):
        field = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        local[center] = {cell_type: torch.tensor(
            (types[t5_nodes.numpy()] == cell_type)
            & np.isin(columns[t5_nodes.numpy()], bins))
            for cell_type in SUBTYPES}
        if any(not mask.any() for mask in local[center].values()):
            raise AssertionError('missing inferred local T5 subtype cells')

    blank = capture([torch.ones(1, 32, 64, dtype=torch.bool)]*18)
    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS),
        graph_sha256=graph.identity(), warmup_ticks=warm_tick,
        ticks_per_frame=8, lag_ticks=LAG,
        rest_overrides={'Tm4': .85, 'Tm9': .85},
        arm_edges={name: int((edge_channel == channel).sum())
                   for channel, name in enumerate(ARMS)},
        local_groups={str(center): {name: dict(cells=int(mask.sum()),
            median_reference_contacts=float(np.median(mass[t5_nodes.numpy()][mask.numpy()])))
            for name, mask in local[center].items()}
            for center in (18, 46)},
        conditions={}, direction_contrasts={}, passes_order_screen=False)
    for center in (18, 46):
        images = dict(right=frames_for_condition(center, 1, 'off', kind='bar'),
                      left=frames_for_condition(center, -1, 'off', kind='bar'))
        static = frames_for_condition(center, 1, 'off', kind='bar')
        for frame in range(2, 15):
            static[frame] = static[8]
        images['static'] = static
        cases = {name: capture(frames) for name, frames in images.items()}
        if cases['right']['pixel_events'] != cases['left']['pixel_events']:
            raise AssertionError('opposite directions have unequal event counts')
        row = report['conditions'][str(center)] = {}
        scores = {}
        for name, case in cases.items():
            delta = case['traces']-blank['traces']
            order = lag_order(delta[:, 2], delta[:, 3], LAG)[24:120].sum(0)
            scores[name] = order
            row[name] = dict(pixel_events=case['pixel_events'], groups={})
            for subtype, mask in local[center].items():
                score = order[mask]
                arm_values = {}
                for channel, arm in enumerate(ARMS):
                    change = delta[24:120, channel][:, mask]
                    arm_values[arm] = dict(
                        mean_signed=float(change.mean()),
                        mean_absolute=float(change.abs().mean()),
                        p90_cell_absolute=float(torch.quantile(
                            change.abs().mean(0), .9)))
                row[name]['groups'][subtype] = dict(
                    neurons=int(mask.sum()),
                    order_mean=float(score.mean()),
                    order_median=float(score.median()),
                    order_p10=float(torch.quantile(score, .1)),
                    order_p90=float(torch.quantile(score, .9)),
                    positive_cells=int((score > 0).sum()),
                    negative_cells=int((score < 0).sum()),
                    excess_spikes=int((case['spikes'][mask]
                                       -blank['spikes'][mask]).sum()),
                    arm_modulation=arm_values)
        for subtype, mask in local[center].items():
            difference = scores['right'][mask]-scores['left'][mask]
            report['direction_contrasts'][f'{center}/{subtype}'] = dict(
                mean=float(difference.mean()),
                median=float(difference.median()),
                positive_cells=int((difference > 0).sum()),
                negative_cells=int((difference < 0).sum()),
                static_magnitude=abs(row['static']['groups'][subtype]['order_mean']))
        print(json.dumps(dict(center=center,
            contrasts={subtype: report['direction_contrasts'][f'{center}/{subtype}']
                       for subtype in SUBTYPES})), flush=True)
    contrasts = report['direction_contrasts']
    consistent = all(
        contrasts[f'18/{subtype}']['mean']*contrasts[f'46/{subtype}']['mean'] > 0
        for subtype in SUBTYPES)
    opposite = all(
        contrasts[f'{center}/T5c']['mean']*contrasts[f'{center}/T5d']['mean'] < 0
        for center in (18, 46))
    median_agrees = all(
        entry['mean']*entry['median'] > 0 for entry in contrasts.values())
    static_lower = all(abs(entry['mean']) > entry['static_magnitude']
                       for entry in contrasts.values())
    report['passes_order_screen'] = bool(consistent and opposite
                                          and median_agrees and static_lower)
    report['screen_parts'] = dict(consistent=consistent, opposite=opposite,
                                  median_agrees=median_agrees,
                                  static_lower=static_lower)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
                          passes_order_screen=report['passes_order_screen'],
                          screen_parts=report['screen_parts'])), flush=True)


if __name__ == '__main__':
    main()

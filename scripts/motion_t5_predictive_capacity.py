"""Frozen T5 predictive-current support for its next-frame motion gate."""

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_pong import NETWORK_STATE
from motion_graded_propagation import GRADED_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_t5_axis_aligned import moving_bar, static_bar
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-predictive-capacity-results.json')
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
PREFERRED = {'T5c': 'up', 'T5d': 'down'}


def summarize(predicted, gate, spikes, arrivals, source_counts):
    issue_ticks = torch.arange(3, 14)*8
    forecast = predicted[issue_ticks]
    recent = torch.stack([arrivals[t-7:t+1].sum(0) for t in issue_ticks])
    future_gate = torch.stack([gate[t+8:t+16].max(0).values
                               for t in issue_ticks])
    future_spikes = torch.stack([spikes[t+8:t+16].sum(0)
                                 for t in issue_ticks])
    event = future_gate > .05
    quiet = ~event
    support = forecast.abs() > 1e-6
    recent_support = recent.sum(2) > 0
    def mean(values, mask):
        return float(values[mask].float().mean()) if mask.any() else None
    return dict(cells=predicted.shape[1], frame_samples=int(event.numel()),
        future_gate_events=int(event.sum()),
        future_gate_fraction=float(event.float().mean()),
        future_spikes=int(future_spikes.sum()),
        predictive_current_support_fraction=float(support.float().mean()),
        event_current_support_fraction=mean(support, event),
        event_recent_arrival_support_fraction=mean(recent_support, event),
        event_absolute_predictive_current=mean(forecast.abs(), event),
        quiet_absolute_predictive_current=mean(forecast.abs(), quiet),
        event_signed_predictive_current=mean(forecast, event),
        quiet_signed_predictive_current=mean(forecast, quiet),
        predictive_arrivals=dict(positive=int(arrivals[24:120, :, 0].sum()),
                                 negative=int(arrivals[24:120, :, 1].sum())),
        top_predictive_source_classes=dict(sorted(source_counts.items(),
            key=lambda item: item[1], reverse=True)[:8]))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(metadata, retina, annotations)
    net = T5LocalOrderNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=.10, voltage_scale=.02,
        source_type='Tm9', source_state='current', gate_gain=8000., gate_cap=1.)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    weights = net.magnitudes.clone()
    for cell_type in ('Tm4', 'Tm9'):
        net.rest_current[torch.tensor(types == cell_type)] = .85
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    net.enable_graded()
    for _ in range(256):
        net.step(zero)
    net.enable_order()
    settled = {name: getattr(net, name).clone()
               for name in NETWORK_STATE+GRADED_STATE+ORDER_STATE}
    settled_tick = net.step_index
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def capture(images, nodes):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        all_nodes = torch.cat(tuple(nodes.values()))
        positions = {name: torch.arange(len(group))+(0 if name == 'T5c'
                     else len(nodes['T5c'])) for name, group in nodes.items()}
        lookup = torch.full((net.n,), -1, dtype=torch.long)
        lookup[all_nodes] = torch.arange(len(all_nodes))
        predicted = torch.zeros((144, len(all_nodes)))
        gate = torch.zeros_like(predicted)
        spikes = torch.zeros_like(predicted)
        arrivals = torch.zeros((144, len(all_nodes), 2), dtype=torch.int16)
        source_counts = {name: {} for name in nodes}
        pixel_events = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                t = frame_index*8+tick
                activity = net.step(sensory if tick == 0 else zero)
                predicted[t] = activity.predicted[0, all_nodes]
                gate[t] = net.last_gate_current[
                    net.t5_lookup[all_nodes]]
                spikes[t] = activity.spikes[0, all_nodes].float()
                edges = activity.arrival_edges
                keep = ((net.pathways[edges] == 1)
                        & (lookup[net.post[edges]] >= 0))
                selected = edges[keep]
                signs = net.visual_arrival_impulse(
                    activity.arrival_environments[keep], selected)
                for channel, positive in ((0, signs > 0), (1, signs < 0)):
                    chosen = selected[positive]
                    arrivals[t, :, channel].index_add_(
                        0, lookup[net.post[chosen]],
                        torch.ones(len(chosen), dtype=torch.int16))
                if 24 <= t < 120:
                    for name, group in nodes.items():
                        group_edges = selected[torch.isin(net.post[selected], group)]
                        for source in types[net.pre[group_edges].numpy()]:
                            row = source_counts[name]
                            row[source] = row.get(source, 0)+1
        if not (torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite frozen state or modified weights')
        return dict(pixel_events=pixel_events,
            groups={name: summarize(predicted[:, indices], gate[:, indices],
                                    spikes[:, indices], arrivals[:, indices],
                                    source_counts[name])
                    for name, indices in positions.items()})

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS), ticks_per_frame=8,
        gate_event_threshold=.05, predictive_support_threshold=1e-6,
        conditions={}, passes_basic_support=False)
    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    for center, speed in CASES:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins)))
            for name in PREFERRED}
        if any(len(group) == 0 for group in nodes.values()):
            raise AssertionError('missing anatomy-local T5 cells')
        cases = dict(down=capture(moving_bar('vertical', center, 1, speed), nodes),
                     up=capture(moving_bar('vertical', center, -1, speed), nodes))
        if cases['up']['pixel_events'] != cases['down']['pixel_events']:
            raise AssertionError('opposite motion has unequal camera events')
        if center == 16 and speed == 1:
            cases['blank'] = capture(bright, nodes)
            cases['static'] = capture(static_bar('vertical', center), nodes)
        key = f'{center}-s{speed}'
        report['conditions'][key] = cases
        print(json.dumps(dict(condition=key,
            supports={direction: {name: row['event_current_support_fraction']
                for name, row in case['groups'].items()}
                for direction, case in cases.items()})), flush=True)
    checks = []
    for case in report['conditions'].values():
        for direction in ('up', 'down'):
            for group in case[direction]['groups'].values():
                event_abs = group['event_absolute_predictive_current']
                quiet_abs = group['quiet_absolute_predictive_current']
                checks.append(group['future_gate_events'] > 0
                    and group['event_current_support_fraction'] is not None
                    and group['event_current_support_fraction'] >= .5
                    and event_abs is not None and quiet_abs is not None
                    and event_abs > quiet_abs)
    report['passes_basic_support'] = all(checks)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_basic_support=report['passes_basic_support'])), flush=True)


if __name__ == '__main__':
    main()

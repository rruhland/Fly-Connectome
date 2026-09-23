"""Frozen Tm4/Tm9 arm history at T5 versus its next-frame motion gate."""

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
from t5_local_order_current import LAG, ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-arm-capacity-results.json')
RECURRENT = Path('docs/experiments/2026-09-23-t5-predictive-capacity-results.json')
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
GROUPS = ('T5c', 'T5d')


def summarize(features, future_gate, future_spikes):
    issue = torch.arange(3, 14)*8
    event = torch.stack([future_gate[t+8:t+16].max(0).values
                         for t in issue]) > .05
    quiet = ~event
    future_count = torch.stack([future_spikes[t+8:t+16].sum(0)
                                for t in issue])
    at_issue = {name: values[issue] for name, values in features.items()}
    history = (at_issue['tm4_now'].abs()+at_issue['tm9_now'].abs()
               +at_issue['tm4_delay'].abs()+at_issue['tm9_delay'].abs())
    tm4 = at_issue['tm4_now'].abs()+at_issue['tm4_delay'].abs()
    tm9 = at_issue['tm9_now'].abs()+at_issue['tm9_delay'].abs()
    order = at_issue['order']
    recurrent = at_issue['recurrent']
    def mean(value, mask):
        return float(value[mask].float().mean()) if mask.any() else None
    return dict(cells=event.shape[1], frame_samples=event.numel(),
        future_gate_events=int(event.sum()),
        future_spikes=int(future_count.sum()),
        event_support=dict(
            arm_history=mean(history > .001, event),
            tm4=mean(tm4 > .001, event),
            tm9=mean(tm9 > .001, event),
            local_order=mean(order > .05, event),
            recurrent=mean(recurrent.abs() > 1e-6, event)),
        event_quiet_means=dict(
            arm_history=(mean(history, event), mean(history, quiet)),
            local_order=(mean(order, event), mean(order, quiet)),
            recurrent_absolute=(mean(recurrent.abs(), event),
                                mean(recurrent.abs(), quiet))))


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
        t5_indices = net.t5_lookup[all_nodes]
        positions = {name: torch.arange(len(group))+(0 if name == 'T5c'
                     else len(nodes['T5c'])) for name, group in nodes.items()}
        traces = {name: torch.zeros((144, len(all_nodes))) for name in (
            'tm4_now', 'tm9_now', 'tm4_delay', 'tm9_delay',
            'order', 'recurrent')}
        gate = torch.zeros((144, len(all_nodes)))
        spikes = torch.zeros_like(gate)
        pixel_events = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                t = frame_index*8+tick
                activity = net.step(sensory if tick == 0 else zero)
                absolute_tick = net.step_index-1
                now = net.residual_history[absolute_tick % (LAG+1)]
                delayed = net.residual_history[(absolute_tick-LAG) % (LAG+1)]
                traces['tm4_now'][t] = now[0, t5_indices]
                traces['tm9_now'][t] = now[1, t5_indices]
                traces['tm4_delay'][t] = delayed[0, t5_indices]
                traces['tm9_delay'][t] = delayed[1, t5_indices]
                traces['order'][t] = net.pending_gate[t5_indices]
                traces['recurrent'][t] = activity.predicted[0, all_nodes]
                gate[t] = net.last_gate_current[t5_indices]
                spikes[t] = activity.spikes[0, all_nodes].float()
        if not (torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite frozen state or changed weights')
        return dict(pixel_events=pixel_events,
            groups={name: summarize({key: value[:, indices]
                for key, value in traces.items()},
                gate[:, indices], spikes[:, indices])
                for name, indices in positions.items()})

    reference = json.loads(RECURRENT.read_text())
    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS), ticks_per_frame=8,
        future_gate_threshold=.05, arm_support_threshold=.001,
        recurrent_support_threshold=1e-6, conditions={},
        passes_upstream_capacity=False)
    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    for center, speed in CASES:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins))) for name in GROUPS}
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
        for direction in cases:
            for name, group in cases[direction]['groups'].items():
                prior = reference['conditions'][key][direction]['groups'][name]
                different = (group['future_gate_events'] != prior['future_gate_events']
                             or group['cells'] != prior['cells'])
                if prior['event_current_support_fraction'] is not None:
                    different |= (abs(group['event_support']['recurrent']
                        -prior['event_current_support_fraction']) > 1e-6)
                if different:
                    raise AssertionError('arm and recurrent audits use different samples')
        report['conditions'][key] = cases
        print(json.dumps(dict(condition=key, supports={direction: {
            name: row['event_support'] for name, row in case['groups'].items()}
            for direction, case in cases.items()})), flush=True)
    checks = []
    for case in report['conditions'].values():
        for direction in ('up', 'down'):
            for group in case[direction]['groups'].values():
                support = group['event_support']
                order_event, order_quiet = group['event_quiet_means']['local_order']
                checks.append(group['future_gate_events'] > 0
                    and support['arm_history'] is not None
                    and support['arm_history'] >= .8
                    and support['tm4'] >= .5 and support['tm9'] >= .5
                    and order_event is not None and order_quiet is not None
                    and order_event > order_quiet)
    report['passes_upstream_capacity'] = all(checks)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_upstream_capacity=report['passes_upstream_capacity'])), flush=True)


if __name__ == '__main__':
    main()

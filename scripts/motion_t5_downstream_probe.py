"""Frozen T5 spike transmission into measured LPi targets."""

import json
from pathlib import Path

import numpy as np
import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_pong import NETWORK_STATE
from motion_graded_propagation import GRADED_STATE
from motion_stage_audit import SOURCE
from motion_t5_axis_aligned import moving_bar, static_bar
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-downstream-probe-results.json')
TARGETS = ('LPi34', 'LPi43', 'LPi3412')
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = T5LocalOrderNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=.10, voltage_scale=.02,
        source_type='Tm9', source_state='current', gate_gain=8000., gate_cap=1.)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    original_weights = net.magnitudes.clone()
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
    target_nodes = {name: torch.tensor(np.flatnonzero(types == name))
                    for name in TARGETS}
    all_targets = torch.cat(tuple(target_nodes.values()))
    lookup = torch.full((net.n,), -1, dtype=torch.long)
    lookup[all_targets] = torch.arange(len(all_targets))
    target_index = {name: lookup[nodes] for name, nodes in target_nodes.items()}
    t5_source = torch.tensor(np.isin(types, ('T5c', 'T5d')))[net.pre]
    t5_edge = t5_source & (lookup[net.post] >= 0) & (net.pathways == 0)
    if not (net.signs[t5_edge] == 1).all():
        raise AssertionError('T5 transmitter sign changed')

    def capture(images, gate):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        net.order_enabled = gate
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        t5 = torch.zeros((144, len(all_targets)))
        ff = torch.zeros_like(t5)
        predicted = torch.zeros_like(t5)
        spikes = torch.zeros_like(t5, dtype=torch.bool)
        predictive_arrivals = torch.zeros(len(all_targets), dtype=torch.long)
        positive_predictive = torch.zeros_like(predictive_arrivals)
        negative_predictive = torch.zeros_like(predictive_arrivals)
        pixel_events = step = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(sensory if tick == 0 else zero,
                                    capture_increments=True)
                edges = activity.arrival_edges
                chosen = edges[t5_edge[edges]]
                predictive_edges = edges[(net.pathways[edges] == 1)
                    & (lookup[net.post[edges]] >= 0)]
                predictive_arrivals.index_add_(
                    0, lookup[net.post[predictive_edges]],
                    torch.ones_like(predictive_edges))
                for sign, count in ((1, positive_predictive),
                                    (-1, negative_predictive)):
                    signed = predictive_edges[net.signs[predictive_edges] == sign]
                    count.index_add_(0, lookup[net.post[signed]],
                        torch.ones_like(signed))
                t5[step].index_add_(0, lookup[net.post[chosen]],
                    net.magnitudes[chosen]*net.signs[chosen])
                ff[step] = activity.feedforward_arrivals[0, all_targets]
                predicted[step] = activity.predicted[0, all_targets]
                spikes[step] = activity.spikes[0, all_targets]
                step += 1
        if not (torch.isfinite(t5).all() and torch.isfinite(ff).all()
                and torch.isfinite(predicted).all()
                and torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite downstream state or changed weights')
        groups = {}
        for name in TARGETS:
            indices = target_index[name]
            a = t5[24:120, indices].sum(0)
            b = ff[24:120, indices].sum(0)
            groups[name] = dict(cells=len(indices), t5_per_cell=a.tolist(),
                ff_per_cell=b.tolist(), t5_total=float(a.sum()),
                feedforward_total=float(b.sum()),
                predictive_mean=float(predicted[24:120, indices].mean()),
                predictive_arrivals=int(predictive_arrivals[indices].sum()),
                positive_predictive_arrivals=int(positive_predictive[indices].sum()),
                negative_predictive_arrivals=int(negative_predictive[indices].sum()),
                spikes=int(spikes[24:120, indices].sum()))
        return dict(pixel_events=pixel_events, groups=groups)

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        ticks_per_frame=8, gate_gain=8000., source_release_cap=.10,
        conditions={}, screen_parts={}, passes_downstream_screen=False)
    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    for center, speed in CASES:
        key = f'{center}-s{speed}'
        rows = report['conditions'][key] = {}
        for gate in (True, False):
            label = 'gate_on' if gate else 'gate_off'
            cases = dict(blank=capture(bright, gate),
                down=capture(moving_bar('vertical', center, 1, speed), gate),
                up=capture(moving_bar('vertical', center, -1, speed), gate),
                static=capture(static_bar('vertical', center), gate))
            if cases['down']['pixel_events'] != cases['up']['pixel_events']:
                raise AssertionError('opposite directions had unequal events')
            contrasts = {}
            for name in TARGETS:
                a, b = cases['down']['groups'][name], cases['up']['groups'][name]
                t5_diff = np.asarray(a['t5_per_cell'])-np.asarray(b['t5_per_cell'])
                ff_diff = np.asarray(a['ff_per_cell'])-np.asarray(b['ff_per_cell'])
                contrasts[name] = dict(t5_total=float(t5_diff.sum()),
                    t5_median=float(np.median(t5_diff)),
                    t5_positive_cells=int((t5_diff > 0).sum()),
                    t5_negative_cells=int((t5_diff < 0).sum()),
                    feedforward_total=float(ff_diff.sum()),
                    feedforward_median=float(np.median(ff_diff)))
            for case in cases.values():
                for group in case['groups'].values():
                    group.pop('t5_per_cell')
                    group.pop('ff_per_cell')
            rows[label] = dict(cases=cases, contrasts=contrasts)
        print(json.dumps(dict(condition=key, contrasts={name: dict(
            gate_on=rows['gate_on']['contrasts'][name]['feedforward_total'],
            gate_off=rows['gate_off']['contrasts'][name]['feedforward_total'])
            for name in TARGETS})), flush=True)
    checks = {}
    for name, sign in (('LPi34', -1), ('LPi43', 1)):
        for center, speed in CASES:
            key = f'{center}-s{speed}'
            on = report['conditions'][key]['gate_on']['contrasts'][name]
            off = report['conditions'][key]['gate_off']['contrasts'][name]
            majority = (on['t5_positive_cells']-on['t5_negative_cells'])*sign > 0
            checks[f'{key}/{name}'] = bool(
                on['t5_total']*sign > 0
                and on['feedforward_total']*sign > 0
                and abs(on['feedforward_total']) >= .5*abs(on['t5_total'])
                and majority
                and (center != 16 or abs(off['t5_total']) < .5*abs(on['t5_total'])))
    report['screen_parts'] = checks
    report['passes_downstream_screen'] = all(checks.values())
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_downstream_screen=report['passes_downstream_screen'],
        screen_parts=checks)), flush=True)


if __name__ == '__main__':
    main()

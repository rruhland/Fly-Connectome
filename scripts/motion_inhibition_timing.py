"""Read-only tick timing of existing signed currents at the frozen T5 stage."""

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


OUT = Path('docs/experiments/2026-09-23-motion-inhibition-timing-results.json')
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
GROUPS = ('T5c', 'T5d', 'LPi34', 'LPi43', 'VS')
PREFERRED = {'T5c': 'up', 'T5d': 'down', 'LPi34': 'up', 'LPi43': 'down'}


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
        release_cap=.10, voltage_scale=.02, source_type='Tm9',
        source_state='current', gate_gain=8000., gate_cap=1.)
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

    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    static_nodes = {name: torch.tensor(np.flatnonzero(types == name))
                    for name in GROUPS[2:]}

    def capture(images, nodes):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        net.order_enabled = True
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        all_nodes = torch.cat(tuple(nodes.values()))
        if (net.target_lookup[all_nodes] >= 0).any():
            raise AssertionError('timing targets must not use sensory context stores')
        lookup = torch.full((net.n,), -1, dtype=torch.long)
        lookup[all_nodes] = torch.arange(len(all_nodes))
        indices = {name: lookup[group] for name, group in nodes.items()}
        arrays = {name: torch.zeros((144, len(all_nodes)))
                  for name in ('exc', 'inh', 'ff', 'ff_arrival', 'gate')}
        spikes = torch.zeros((144, len(all_nodes)), dtype=torch.bool)
        pred_arrivals = torch.zeros((2, 144, len(all_nodes)), dtype=torch.int16)
        pixel_events = step = 0
        graded_predictive = net.graded_edges[net.pathways[net.graded_edges] == 1]
        graded_lookup = lookup[net.post[graded_predictive]]
        local_graded = graded_lookup >= 0
        graded_predictive = graded_predictive[local_graded]
        graded_lookup = graded_lookup[local_graded]
        graded_sources = net.graded_sources[
            net.pathways[net.graded_edges] == 1][local_graded]
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                if len(graded_predictive):
                    due = net.release_history[(net.step_index-net.delays[
                        graded_predictive]).remainder(net.history_length),
                        graded_sources]
                    active = due > 0
                    signs = net.visual_impulse(graded_predictive)
                    for channel, signed in ((0, signs > 0), (1, signs < 0)):
                        selected = active & signed
                        pred_arrivals[channel, step].index_add_(
                            0, graded_lookup[selected],
                            torch.ones_like(graded_lookup[selected], dtype=torch.int16))
                activity = net.step(sensory if tick == 0 else zero,
                                    capture_increments=True)
                arrays['exc'][step] = net.excitatory_prediction[0, all_nodes]
                arrays['inh'][step] = net.inhibitory_prediction[0, all_nodes]
                if not torch.allclose(arrays['exc'][step]+arrays['inh'][step],
                                      activity.predicted[0, all_nodes], atol=1e-5):
                    raise AssertionError('signed stores differ from physical prediction')
                arrays['ff'][step] = net.feedforward_current[0, all_nodes]
                arrays['ff_arrival'][step] = activity.feedforward_arrivals[0, all_nodes]
                t5_positions = net.t5_lookup[all_nodes]
                chosen = t5_positions >= 0
                arrays['gate'][step, chosen] = net.last_gate_current[
                    t5_positions[chosen]]
                spikes[step] = activity.spikes[0, all_nodes]
                edges = activity.arrival_edges
                predictive = edges[(net.pathways[edges] == 1)
                                   & (lookup[net.post[edges]] >= 0)]
                signs = net.visual_arrival_impulse(
                    activity.arrival_environments[
                        (net.pathways[edges] == 1)
                        & (lookup[net.post[edges]] >= 0)], predictive)
                for channel, signed in ((0, signs > 0), (1, signs < 0)):
                    selected = predictive[signed]
                    pred_arrivals[channel, step].index_add_(
                        0, lookup[net.post[selected]],
                        torch.ones_like(selected, dtype=torch.int16))
                step += 1
        if not (torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite state or modified long-term weights')
        groups = {}
        for name in GROUPS:
            positions = indices[name]
            active = slice(24, 120)
            inhibition = -arrays['inh'][active, positions]
            phases = [int(spikes[k:120:8, positions].sum()) for k in range(24, 32)]
            groups[name] = dict(cells=len(positions),
                inhibition_per_cell=inhibition.mean(0).tolist(),
                inhibition_mean=float(inhibition.mean()),
                excitation_mean=float(arrays['exc'][active, positions].mean()),
                feedforward_mean=float(arrays['ff'][active, positions].mean()),
                feedforward_arrival_total=float(arrays['ff_arrival'][active, positions].sum()),
                gate_mean=float(arrays['gate'][active, positions].mean()),
                spikes=int(spikes[active, positions].sum()), spike_phase=phases,
                inhibitory_phase=[float((-arrays['inh'][k:120:8, positions]).mean())
                                  for k in range(24, 32)],
                inhibitory_arrival_phase=[int(pred_arrivals[1, k:120:8, positions].sum())
                                          for k in range(24, 32)],
                predictive_arrivals=dict(positive=int(pred_arrivals[0, active, positions].sum()),
                                         negative=int(pred_arrivals[1, active, positions].sum())))
        return dict(pixel_events=pixel_events, groups=groups)

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS), ticks_per_frame=8,
        source_release_cap=.10, gate_gain=8000., conditions={},
        passes_existing_null_bias_screen={})
    for center, speed in CASES:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins)))
            for name in GROUPS[:2]}
        nodes.update(static_nodes)
        if any(len(group) == 0 for group in nodes.values()):
            raise AssertionError('missing anatomical motion-stage group')
        bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
        cases = dict(blank=capture(bright, nodes),
                     down=capture(moving_bar('vertical', center, 1, speed), nodes),
                     up=capture(moving_bar('vertical', center, -1, speed), nodes),
                     static=capture(static_bar('vertical', center), nodes))
        if cases['down']['pixel_events'] != cases['up']['pixel_events']:
            raise AssertionError('opposite motion has unequal camera events')
        contrasts = {}
        for name, direction in PREFERRED.items():
            preferred = cases[direction]['groups'][name]
            null = cases['up' if direction == 'down' else 'down']['groups'][name]
            a = np.asarray(preferred['inhibition_per_cell'])
            b = np.asarray(null['inhibition_per_cell'])
            participating = (a+b) > 1e-6
            difference = b-a
            majority = int((difference[participating] > 1e-6).sum())
            contrasts[name] = dict(preferred=direction,
                null_minus_preferred_inhibition=float(difference.mean()),
                participating_cells=int(participating.sum()),
                null_greater_cells=majority,
                fraction_null_greater=float(majority/max(int(participating.sum()), 1)),
                null_bias=bool(difference.mean() > 0
                               and majority > int(participating.sum())/2))
        for case in cases.values():
            for group in case['groups'].values():
                group.pop('inhibition_per_cell')
        key = f'{center}-s{speed}'
        report['conditions'][key] = dict(cases=cases, contrasts=contrasts)
        print(json.dumps(dict(condition=key, contrasts={name: dict(
            null_minus_preferred_inhibition=row['null_minus_preferred_inhibition'],
            fraction_null_greater=row['fraction_null_greater'])
            for name, row in contrasts.items()})), flush=True)
    for name in ('T5c', 'T5d'):
        report['passes_existing_null_bias_screen'][name] = all(
            row['contrasts'][name]['null_bias']
            for row in report['conditions'].values())
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_existing_null_bias_screen=report['passes_existing_null_bias_screen'])),
        flush=True)


if __name__ == '__main__':
    main()

"""Blank-calibrated Tm9 graded output on its measured T5 edges."""

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
from graded_visual import GradedVisualNetwork
from motion_graded_propagation import GRADED_STATE, select_cap, summarize_target
from motion_source_signal_audit import summarize_feature
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns
from motion_t5_arm_order import lag_order


OUT = Path('docs/experiments/2026-09-23-tm9-graded-results.json')
CAPS = (.01, .03, .10)
SUBTYPES = ('T5c', 'T5d')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = GradedVisualNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=CAPS[0], voltage_scale=.02,
        source_type='Tm9', source_state='current')
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    original_weights = net.magnitudes.clone()
    for cell_type in ('Tm4', 'Tm9'):
        net.rest_current[torch.tensor(types == cell_type)] = .85
    if (len(net.graded_edges) != int((types[graph.pre] == 'Tm9').sum())
            or not (net.signs[net.graded_edges] == 1).all()):
        raise AssertionError('Tm9 measured edge roster or transmitter sign changed')
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone()
            for name in NETWORK_STATE+GRADED_STATE}
    warm_blank_residual = net.blank_residual.clone()
    warm_blank_count = net.blank_count
    warm_tick = net.step_index
    t5_all = torch.tensor(np.char.startswith(types.astype(str), 'T5'))
    selected_targets = torch.tensor(np.flatnonzero(np.isin(types, SUBTYPES)))
    target_lookup = np.full(net.n, -1, dtype=np.int64)
    target_lookup[selected_targets.numpy()] = np.arange(len(selected_targets))
    tm4_edges = (torch.tensor(types == 'Tm4')[net.pre]
                 & torch.tensor(target_lookup[net.post.numpy()] >= 0)
                 & (net.pathways == 0))
    tm9_to_targets = net.graded_edge_mask & torch.tensor(
        target_lookup[net.post.numpy()] >= 0)
    if not (net.pathways[tm9_to_targets] == 0).all():
        raise AssertionError('graded Tm9->T5c/d order trace requires feedforward edges')
    target_positions = torch.tensor(target_lookup)
    target_decay = net.current_decay[selected_targets]

    def advance_arms(activity, tm4_state, tm9_state):
        tm4_state.mul_(target_decay)
        tm9_state.mul_(target_decay).add_(
            net.last_graded_impulse[0, selected_targets])
        edges = activity.arrival_edges
        selected = edges[tm4_edges[edges]]
        tm4_state.index_add_(0, target_positions[net.post[selected]],
            net.magnitudes[selected]*net.signs[selected])

    calibration = []
    for cap in CAPS:
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        net.blank_residual.copy_(warm_blank_residual)
        net.blank_count = warm_blank_count
        net.graded_enabled = False
        net.release_cap = cap
        net.enable_graded()
        impulses = []
        blank_spikes = 0
        for tick in range(256):
            activity = net.step(zero)
            if tick >= 128:
                impulses.append(net.last_graded_impulse[0, t5_all].abs().clone())
                blank_spikes += int(activity.spikes[0, t5_all].sum())
        row = dict(cap=cap,
            blank_p99_impulse=float(torch.quantile(torch.stack(impulses), .99)),
            blank_spike_rate=blank_spikes/(128*int(t5_all.sum())),
            finite=bool(torch.isfinite(net.voltage).all()
                        and torch.isfinite(net.last_graded_impulse).all()))
        calibration.append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    selected_cap = select_cap(calibration)
    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS),
        graph_sha256=graph.identity(), graded_edges=len(net.graded_edges),
        graded_contacts=int(graph.contacts[net.graded_edges.numpy()].sum()),
        graded_source_cells=len(net.graded_nodes),
        warmup_ticks=warm_tick, settle_ticks=256,
        ticks_per_frame=8, source_state='positive_fast_current',
        current_scale=.02, rest_overrides={'Tm4': .85, 'Tm9': .85},
        calibration=calibration, selected_cap=selected_cap,
        conditions={}, partial_tm9_t5_screen=False)
    if selected_cap is None:
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(OUT), selected_cap=None)), flush=True)
        return

    for name, value in warm.items():
        getattr(net, name).copy_(value)
    net.step_index = warm_tick
    net.blank_residual.copy_(warm_blank_residual)
    net.blank_count = warm_blank_count
    net.graded_enabled = False
    net.release_cap = selected_cap
    net.enable_graded()
    tm4_state = torch.zeros(len(selected_targets))
    tm9_state = torch.zeros_like(tm4_state)
    for _ in range(256):
        advance_arms(net.step(zero), tm4_state, tm9_state)
    settled = {name: getattr(net, name).clone()
               for name in NETWORK_STATE+GRADED_STATE}
    settled_tick = net.step_index
    settled_tm4 = tm4_state.clone()
    settled_tm9 = tm9_state.clone()

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_columns = annotated_columns(metadata, retina, annotations)
    target_columns, _ = infer_columns(metadata, retina, annotations)
    source_lookup = np.full(net.n, -1, dtype=np.int64)
    source_lookup[net.graded_nodes.numpy()] = np.arange(len(net.graded_nodes))
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    source_regions, target_regions = {}, {}
    for center in (18, 46):
        field = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        source_regions[center] = torch.tensor(source_lookup[
            np.flatnonzero((types == 'Tm9') & np.isin(source_columns, bins))])
        target_regions[center] = {subtype: torch.tensor(target_lookup[
            np.flatnonzero((types == subtype) & np.isin(target_columns, bins))])
            for subtype in SUBTYPES}
        if (len(source_regions[center]) == 0 or any(
                len(mask) == 0 for mask in target_regions[center].values())):
            raise AssertionError('missing anatomically local Tm9/T5 cells')

    def capture(images):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        tm4_state = settled_tm4.clone()
        tm9_state = settled_tm9.clone()
        release = torch.empty((144, len(net.graded_nodes)))
        traces = {name: torch.empty((144, len(selected_targets)))
                  for name in ('impulse', 'current', 'voltage')}
        traces['spikes'] = torch.zeros((144, len(selected_targets)), dtype=torch.bool)
        traces['tm4_arm'] = torch.empty((144, len(selected_targets)))
        traces['tm9_arm'] = torch.empty((144, len(selected_targets)))
        source_spikes = 0
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                advance_arms(activity, tm4_state, tm9_state)
                release[step] = net.release_history[
                    (net.step_index-1) % net.history_length]
                traces['impulse'][step] = net.last_graded_impulse[0, selected_targets]
                traces['current'][step] = (
                    net.feedforward_current[0, selected_targets]
                    +net.predictive_current[0, selected_targets]
                    +net.behavioral_current[0, selected_targets])
                traces['voltage'][step] = net.voltage[0, selected_targets]
                traces['spikes'][step] = activity.spikes[0, selected_targets]
                traces['tm4_arm'][step] = tm4_state
                traces['tm9_arm'][step] = tm9_state
                if 3 <= frame < 15:
                    source_spikes += int(activity.spikes[0, net.graded_nodes].sum())
                step += 1
        if not (torch.isfinite(release).all()
                and all(torch.isfinite(traces[name]).all()
                        for name in ('impulse', 'current', 'voltage',
                                     'tm4_arm', 'tm9_arm'))):
            raise AssertionError('non-finite Tm9 propagation state')
        return dict(release=release, source_spikes=source_spikes,
                    pixel_events=pixel_events, **traces)

    blank = capture([torch.ones((1, 32, 64), dtype=torch.bool)]*18)
    repeated_blank = capture([torch.ones((1, 32, 64), dtype=torch.bool)]*18)
    if any(not torch.equal(blank[name], repeated_blank[name])
           for name in ('release', 'impulse', 'current', 'voltage',
                        'spikes', 'tm4_arm', 'tm9_arm')):
        raise AssertionError('restored graded state changed repeated blank')
    report['state_restoration_reproducible'] = True
    order_scores = {}
    for center in (18, 46):
        other = 46 if center == 18 else 18
        images = dict(right=frames_for_condition(center, 1, 'off', kind='bar'),
                      left=frames_for_condition(center, -1, 'off', kind='bar'))
        static = frames_for_condition(center, 1, 'off', kind='bar')
        for frame in range(2, 15):
            static[frame] = static[8]
        images['static'] = static
        for direction, frames in images.items():
            observed = capture(frames)
            tm4_delta = observed['tm4_arm']-blank['tm4_arm']
            tm9_delta = observed['tm9_arm']-blank['tm9_arm']
            order = lag_order(tm4_delta, tm9_delta, 8)[24:120].sum(0)
            order_scores[f'{center}-{direction}'] = order
            source = summarize_feature(observed['release'], blank['release'],
                                       source_regions[center], source_regions[other])
            targets = {subtype: summarize_target(
                observed, blank, target_regions[center][subtype])
                for subtype in SUBTYPES}
            report['conditions'][f'{center}-{direction}'] = dict(
                pixel_events=observed['pixel_events'],
                source_spikes=observed['source_spikes'],
                source=source, targets=targets,
                order={subtype: dict(
                    mean=float(order[target_regions[center][subtype]].mean()),
                    median=float(order[target_regions[center][subtype]].median()))
                    for subtype in SUBTYPES})
            print(json.dumps(dict(center=center, condition=direction,
                source_fraction=source['local_fraction_above_blank_p99'],
                target_impulses={subtype: targets[subtype]['impulse']
                                 ['mean_absolute_change'] for subtype in SUBTYPES})),
                flush=True)
    report['partial_tm9_t5_screen'] = bool(all(
        (row := report['conditions'][f'{center}-{direction}'])['source']
            ['local_fraction_above_blank_p99'] >= .05
        and all(row['targets'][subtype]['impulse']['mean_absolute_change'] >= .0005
                and row['targets'][subtype]['voltage']['per_cell_absolute_p90'] >= .002
                for subtype in SUBTYPES)
        for center in (18, 46) for direction in ('right', 'left')))
    report['order_contrasts'] = {}
    for center in (18, 46):
        for subtype in SUBTYPES:
            mask = target_regions[center][subtype]
            contrast = (order_scores[f'{center}-right'][mask]
                        -order_scores[f'{center}-left'][mask])
            static = order_scores[f'{center}-static'][mask]
            report['order_contrasts'][f'{center}/{subtype}'] = dict(
                mean=float(contrast.mean()), median=float(contrast.median()),
                positive_cells=int((contrast > 0).sum()),
                negative_cells=int((contrast < 0).sum()),
                static_magnitude=abs(float(static.mean())))
    contrasts = report['order_contrasts']
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
    report['order_screen_parts'] = dict(consistent=consistent,
        opposite=opposite, median_agrees=median_agrees,
        static_lower=static_lower)
    report['passes_order_screen'] = bool(consistent and opposite
                                          and median_agrees and static_lower)
    if (not torch.equal(net.magnitudes, original_weights)
            or checksum(SOURCE) != source_sha):
        raise AssertionError('frozen weights or checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), selected_cap=selected_cap,
                          partial_tm9_t5_screen=report['partial_tm9_t5_screen'],
                          passes_order_screen=report['passes_order_screen'],
                          order_parts=report['order_screen_parts'])),
          flush=True)


if __name__ == '__main__':
    main()

"""Frozen CT1 current around each T5 cell's own feedforward arrival."""

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from ct1_column_shadow import SHADOW_STATE
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_pong import NETWORK_STATE
from motion_ct1_column_shadow import measured_ct1_shadow, route_shadow_by_synapse_geometry
from motion_graded_propagation import GRADED_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_t5_axis_aligned import moving_bar
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-ct1-event-aligned-results.json')
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
PREFERRED = {'T5c': 'up', 'T5d': 'down'}
OFFSETS = torch.arange(-8, 9)
SPIKE_OFFSETS = torch.arange(-4, 5)


def aligned(values, anchors, cells, offsets):
    return values[anchors[cells, None]+offsets[None, :], cells[:, None]]


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
    shadow, contacts = measured_ct1_shadow(net, metadata, retina, annotations)
    geometry = route_shadow_by_synapse_geometry(shadow, net, metadata)
    for _ in range(256):
        shadow.begin_tick(net, output_enabled=False)
        net.step(zero)
    shadow.enable()
    network_state = {name: getattr(net, name).clone()
                     for name in NETWORK_STATE+GRADED_STATE+ORDER_STATE}
    shadow_state = {name: getattr(shadow, name).clone()
                    for name in SHADOW_STATE}
    settled_tick = net.step_index
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def capture(images, nodes, *, with_ct1):
        for name, value in network_state.items():
            getattr(net, name).copy_(value)
        for name, value in shadow_state.items():
            getattr(shadow, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        arrays = {name: {field: torch.zeros((144, len(group)))
                         for field in ('ff', 'inh', 'spikes')}
                  for name, group in nodes.items()}
        events_count = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                events_count += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                t = frame_index*8+tick
                shadow.begin_tick(net, output_enabled=with_ct1)
                activity = net.step(sensory if tick == 0 else zero,
                                    capture_increments=True)
                for name, group in nodes.items():
                    arrays[name]['ff'][t] = activity.feedforward_arrivals[0, group]
                    arrays[name]['inh'][t] = -shadow.last_impulse[group]
                    arrays[name]['spikes'][t] = activity.spikes[0, group].float()
        if not (torch.isfinite(net.voltage).all()
                and torch.isfinite(shadow.current).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite or altered frozen CT1 trial')
        return events_count, arrays

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS),
        ct1_body=10009, contacts=contacts, geometry_route=geometry,
        anchor='off-arm maximum positive non-CT1 feedforward increment',
        minimum_anchor_increment=.005, offsets=OFFSETS.tolist(),
        conditions={}, passes_event_aligned_screen=False)
    for center, speed in CASES:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins)))
            for name in PREFERRED}
        if any(len(group) == 0 for group in nodes.values()):
            raise AssertionError('empty local T5 event group')
        trials = {}
        for direction, sign in (('down', 1), ('up', -1)):
            frames = moving_bar('vertical', center, sign, speed)
            trials[direction] = {}
            for arm, enabled in (('off', False), ('on', True)):
                count, arrays = capture(frames, nodes, with_ct1=enabled)
                trials[direction][arm] = dict(events=count, arrays=arrays)
        if (trials['down']['off']['events'] != trials['up']['off']['events']
                or any(trials[direction]['on']['events']
                       != trials[direction]['off']['events']
                       for direction in ('up', 'down'))):
            raise AssertionError('event camera differs across matched trials')
        groups = {}
        for name, preferred in PREFERRED.items():
            null = 'up' if preferred == 'down' else 'down'
            pref_off = trials[preferred]['off']['arrays'][name]
            null_off = trials[null]['off']['arrays'][name]
            pref_on = trials[preferred]['on']['arrays'][name]
            null_on = trials[null]['on']['arrays'][name]
            pref_peak, pref_anchor = pref_off['ff'][24:120].clamp(min=0).max(0)
            null_peak, null_anchor = null_off['ff'][24:120].clamp(min=0).max(0)
            pref_anchor += 24
            null_anchor += 24
            valid = (pref_peak >= .005) & (null_peak >= .005)
            cells = valid.nonzero().flatten()
            if not len(cells):
                raise AssertionError('no shared feedforward-anchored T5 cells')
            pref_profile = aligned(pref_on['inh'], pref_anchor, cells, OFFSETS)
            null_profile = aligned(null_on['inh'], null_anchor, cells, OFFSETS)
            pref_pre = pref_profile[:, 4:9].mean(1)
            null_pre = null_profile[:, 4:9].mean(1)
            participating = (pref_pre+null_pre) > 1e-8
            bias = null_pre-pref_pre
            null_greater = int((bias[participating] > 1e-8).sum())
            null_off_spikes = aligned(null_off['spikes'], null_anchor,
                                      cells, SPIKE_OFFSETS).sum()
            null_on_spikes = aligned(null_on['spikes'], null_anchor,
                                     cells, SPIKE_OFFSETS).sum()
            pref_off_spikes = aligned(pref_off['spikes'], pref_anchor,
                                      cells, SPIKE_OFFSETS).sum()
            pref_on_spikes = aligned(pref_on['spikes'], pref_anchor,
                                     cells, SPIKE_OFFSETS).sum()
            row = groups[name] = dict(cells=len(nodes[name]),
                anchored_cells=len(cells), participating_cells=int(participating.sum()),
                pre_event_inhibition=dict(preferred=float(pref_pre.mean()),
                    null=float(null_pre.mean()),
                    null_minus_preferred=float(bias.mean()),
                    null_greater_cells=null_greater),
                post_event_inhibition=dict(preferred=float(pref_profile[:, 9:13].mean()),
                    null=float(null_profile[:, 9:13].mean())),
                profile_preferred=pref_profile.mean(0).tolist(),
                profile_null=null_profile.mean(0).tolist(),
                profile_peak_offset=dict(preferred=int(OFFSETS[
                    pref_profile.mean(0).argmax()]), null=int(OFFSETS[
                    null_profile.mean(0).argmax()])),
                anchor_feedforward=dict(preferred_off=float(pref_peak[cells].mean()),
                    null_off=float(null_peak[cells].mean()),
                    preferred_on=float(pref_on['ff'][pref_anchor[cells], cells].mean()),
                    null_on=float(null_on['ff'][null_anchor[cells], cells].mean())),
                event_spikes=dict(preferred_off=int(pref_off_spikes),
                    preferred_on=int(pref_on_spikes), null_off=int(null_off_spikes),
                    null_on=int(null_on_spikes)),
                null_reduction=float((null_off_spikes-null_on_spikes)
                                     /null_off_spikes.clamp(min=1)),
                preferred_retention=float(pref_on_spikes
                                          /pref_off_spikes.clamp(min=1)))
            row['passes'] = bool(row['pre_event_inhibition']['null_minus_preferred'] > 0
                and null_greater > int(participating.sum())/2
                and row['null_reduction'] >= .10
                and row['preferred_retention'] >= .90)
        key = f'{center}-s{speed}'
        report['conditions'][key] = dict(pixel_events=trials['down']['off']['events'],
                                          groups=groups)
        print(json.dumps(dict(condition=key, groups={name: dict(
            anchored_cells=row['anchored_cells'],
            null_minus_pre=row['pre_event_inhibition']['null_minus_preferred'],
            null_reduction=row['null_reduction'], passes=row['passes'])
            for name, row in groups.items()})), flush=True)
    report['passes_event_aligned_screen'] = all(row['passes']
        for case in report['conditions'].values() for row in case['groups'].values())
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_event_aligned_screen=report['passes_event_aligned_screen'])), flush=True)


if __name__ == '__main__':
    main()

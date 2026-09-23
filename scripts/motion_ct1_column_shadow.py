"""Frozen off/on test of measured CT1 contacts in inferred local compartments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch

from ct1_column_shadow import CT1ColumnShadow, SHADOW_STATE
from ct1_geometry_routing import route_output_contacts
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from full_context_pong import NETWORK_STATE
from motion_ct1_wiring import TRANSMITTERS, WEIGHTS
from motion_ct1_synapse_geometry import CACHE
from motion_graded_propagation import GRADED_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns
from motion_t5_axis_aligned import moving_bar, static_bar
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-ct1-column-shadow-results.json')
GEOMETRY_OUT = Path('docs/experiments/2026-09-23-ct1-geometry-shadow-results.json')
COLUMN_RESULT = OUT
CT1 = 10009
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
PREFERRED = {'T5c': 'up', 'T5d': 'down'}


def measured_ct1_shadow(net, metadata, retina, annotations):
    ids = np.asarray(metadata['graph']['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    source_columns = annotated_columns(metadata, retina, annotations,
                                        classes=('Tm1', 'Tm9'))
    target_columns, _ = infer_columns(metadata, retina, annotations)
    source_info = {int(ids[i]): (int(i), types[i], int(source_columns[i]))
                   for i in np.flatnonzero(np.isin(types, ('Tm1', 'Tm9')))}
    target_info = {int(ids[i]): (int(i), int(target_columns[i]))
                   for i in np.flatnonzero(np.char.startswith(types.astype(str), 'T5'))}
    graded_lookup = {int(node): index
                     for index, node in enumerate(net.graded_nodes.tolist())}
    inputs = {'Tm1': ([], [], []), 'Tm9': ([], [], [])}
    output = ([], [], [])
    with pa.memory_map(str(WEIGHTS), 'r') as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            weights = batch.column('weight').to_numpy()
            for row in np.flatnonzero(post == CT1):
                info = source_info.get(int(pre[row]))
                if info is None or weights[row] < 3:
                    continue
                node, cell_type, column = info
                if column < 0 or net.graph.signs[node] != 1:
                    raise AssertionError('unmapped or nonexcitatory CT1 source')
                nodes, columns, contacts = inputs[cell_type]
                nodes.append(node if cell_type == 'Tm1' else graded_lookup[node])
                columns.append(column)
                contacts.append(int(weights[row]))
            for row in np.flatnonzero(pre == CT1):
                info = target_info.get(int(post[row]))
                if info is None or weights[row] < 3:
                    continue
                node, column = info
                if column < 0:
                    raise AssertionError('unmapped CT1 T5 target')
                for store, value in zip(output, (node, column, int(weights[row]))):
                    store.append(value)
    transmitter = feather.read_table(TRANSMITTERS,
        columns=['body', 'ground_truth'])
    row = transmitter.filter(pc.equal(transmitter['body'], CT1)).to_pylist()
    if len(row) != 1 or row[0]['ground_truth'] != 'gaba':
        raise AssertionError('measured CT1 transmitter must be GABA')
    if (sum(len(value[0]) for value in inputs.values()) != 1715
            or len(output[0]) != 3355):
        raise AssertionError('CT1 threshold-three route differs from roster audit')
    tm1, tm9 = inputs['Tm1'], inputs['Tm9']
    shadow = CT1ColumnShadow(net, len(retina.spec['hex_columns']),
        tm1_nodes=tm1[0], tm1_columns=tm1[1],
        tm1_weights=np.asarray(tm1[2])*metadata['gain'],
        tm9_indices=tm9[0], tm9_columns=tm9[1],
        tm9_weights=np.asarray(tm9[2])*metadata['gain'],
        output_posts=output[0], output_columns=output[1],
        output_weights=np.asarray(output[2])*metadata['gain'])
    counts = dict(input_edges={name: len(value[0]) for name, value in inputs.items()},
                  output_edges=len(output[0]),
                  input_contacts={name: sum(value[2]) for name, value in inputs.items()},
                  output_contacts=sum(output[2]))
    return shadow, counts


def route_shadow_by_synapse_geometry(shadow, net, metadata):
    if not CACHE.exists():
        raise FileNotFoundError('run motion_ct1_synapse_geometry.py first')
    ids = np.asarray(metadata['graph']['body_ids'])
    types = np.asarray(metadata['retina']['cell_types'])
    source_columns = {
        int(ids[node]): int(column)
        for node, column in zip(shadow.tm1_nodes.tolist(),
                                shadow.tm1_columns.tolist())}
    source_columns.update({
        int(ids[net.graded_nodes[index]]): int(column)
        for index, column in zip(shadow.tm9_indices.tolist(),
                                 shadow.tm9_columns.tolist())})
    target_nodes = {int(ids[i]): int(i)
                    for i in np.flatnonzero(np.isin(types, ('T5c', 'T5d')))}
    with np.load(CACHE) as saved:
        posts, columns, contacts = route_output_contacts(
            saved['input_body'], saved['input_xyz'],
            saved['output_body'], saved['output_xyz'],
            source_columns, target_nodes)
    if int(contacts.sum()) != 37186:
        raise AssertionError('geometry route changed measured T5c/d contact mass')
    shadow.output_posts = torch.tensor(posts, dtype=torch.long)
    shadow.output_columns = torch.tensor(columns, dtype=torch.long)
    shadow.output_weights = torch.tensor(contacts*metadata['gain'],
                                         dtype=torch.float32)
    return dict(output_edges=len(posts), output_contacts=int(contacts.sum()),
                cached_sites_sha256=checksum(CACHE))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-routed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    target_columns, _ = infer_columns(metadata, retina, annotations)
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
    shadow, contact_counts = measured_ct1_shadow(net, metadata, retina, annotations)
    if args.geometry_routed:
        contact_counts['geometry_route'] = route_shadow_by_synapse_geometry(
            shadow, net, metadata)
    for _ in range(256):
        shadow.begin_tick(net, output_enabled=False)
        net.step(zero)
    shadow.enable()
    state = {name: getattr(net, name).clone()
             for name in NETWORK_STATE+GRADED_STATE+ORDER_STATE}
    shadow_state = {name: getattr(shadow, name).clone() for name in SHADOW_STATE}
    settled_tick = net.step_index
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def capture(images, nodes, bins, *, with_ct1):
        for name, value in state.items():
            getattr(net, name).copy_(value)
        for name, value in shadow_state.items():
            getattr(shadow, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        counts = {name: 0 for name in nodes}
        impulse = {name: 0. for name in nodes}
        current = {name: 0. for name in nodes}
        pixel_events = 0
        release_occupancy = 0.
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                shadow.begin_tick(net, output_enabled=with_ct1)
                activity = net.step(sensory if tick == 0 else zero)
                if 24 <= frame_index*8+tick < 120:
                    release_occupancy += float((shadow.last_release[bins] > 0).float().mean())/96
                    for name, group in nodes.items():
                        counts[name] += int(activity.spikes[0, group].sum())
                        impulse[name] += float((-shadow.last_impulse[group]).mean())/96
                        current[name] += float(net.feedforward_current[0, group].mean())/96
        if not (torch.isfinite(net.voltage).all()
                and torch.isfinite(shadow.current).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite shadow dynamics or altered weights')
        return dict(pixel_events=pixel_events, spikes=counts,
            ct1_inhibition_per_cell_tick=impulse,
            feedforward_current_per_cell_tick=current,
            local_release_occupancy=release_occupancy)

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS), weights_sha256=checksum(WEIGHTS),
        ct1_body=CT1, contacts=contact_counts, ticks_per_frame=8,
        output_route='nearest_input_site' if args.geometry_routed else 'same_target_column',
        release_scale=.02, release_cap=.10, conditions={}, passes_rescue=False)
    reference = json.loads(COLUMN_RESULT.read_text()) if args.geometry_routed else None
    for center, speed in CASES:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(target_columns, bins.numpy())))
            for name in PREFERRED}
        if any(len(group) == 0 for group in nodes.values()):
            raise AssertionError('missing anatomy-local T5 group')
        bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
        arms = {}
        for with_ct1 in (False, True):
            arms['on' if with_ct1 else 'off'] = dict(
                blank=capture(bright, nodes, bins, with_ct1=with_ct1),
                down=capture(moving_bar('vertical', center, 1, speed),
                             nodes, bins, with_ct1=with_ct1),
                up=capture(moving_bar('vertical', center, -1, speed),
                           nodes, bins, with_ct1=with_ct1),
                static=capture(static_bar('vertical', center),
                               nodes, bins, with_ct1=with_ct1))
        if any(arms[arm]['up']['pixel_events'] != arms[arm]['down']['pixel_events']
               for arm in arms):
            raise AssertionError('opposite motions have unequal events')
        key = f'{center}-s{speed}'
        if reference is not None:
            for case in ('blank', 'down', 'up', 'static'):
                if (arms['off'][case]['spikes'] != reference['conditions'][key]
                        ['arms']['off'][case]['spikes']):
                    raise AssertionError('geometry off arm differs from fixed comparator')
        screen = {}
        for name, preferred in PREFERRED.items():
            null = 'up' if preferred == 'down' else 'down'
            pref_off = arms['off'][preferred]['spikes'][name]
            pref_on = arms['on'][preferred]['spikes'][name]
            null_off = arms['off'][null]['spikes'][name]
            null_on = arms['on'][null]['spikes'][name]
            inhibition_pref = arms['on'][preferred]['ct1_inhibition_per_cell_tick'][name]
            inhibition_null = arms['on'][null]['ct1_inhibition_per_cell_tick'][name]
            screen[name] = dict(preferred=preferred,
                null_reduction=(null_off-null_on)/max(null_off, 1),
                preferred_retention=pref_on/max(pref_off, 1),
                null_greater_ct1_current=inhibition_null > inhibition_pref,
                blank_safe=arms['on']['blank']['spikes'][name]
                           <= arms['off']['blank']['spikes'][name])
            row = screen[name]
            row['passes'] = bool(row['null_reduction'] >= .10
                and row['preferred_retention'] >= .90
                and pref_on > null_on and row['null_greater_ct1_current']
                and row['blank_safe'])
        report['conditions'][key] = dict(arms=arms, screen=screen)
        print(json.dumps(dict(condition=key, screen=screen)), flush=True)
    report['passes_rescue'] = all(row['screen'][name]['passes']
        for row in report['conditions'].values() for name in PREFERRED)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output = GEOMETRY_OUT if args.geometry_routed else OUT
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), passes_rescue=report['passes_rescue'])),
        flush=True)


if __name__ == '__main__':
    main()

"""Frozen Tm4 afferent source state at missing T5 arm-history events."""

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


OUT = Path('docs/experiments/2026-09-23-tm4-subthreshold-results.json')
GROUPS = ('T5c', 'T5d')


def category(values, mask):
    if not mask.any():
        return dict(samples=0)
    positive = values['current_deviation'].clamp(min=0)[mask]
    return dict(samples=int(mask.sum()),
        mean_source_current_deviation=float(values['current_deviation'][mask].mean()),
        mean_positive_source_current_deviation=float(positive.mean()),
        positive_deviation_above_01_fraction=float((
            values['current_deviation'][mask] > .01).float().mean()),
        mean_absolute_voltage_deviation=float(
            values['voltage_deviation'][mask].abs().mean()),
        weighted_source_spike_fraction=float(values['source_spike'][mask].mean()))


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
    field = ((y >= 16-8) & (y <= 16+8)
             & (x >= 24) & (x <= 40))
    bins = retina.pixel_bins[field.flatten()].unique().numpy()
    nodes = {name: torch.tensor(np.flatnonzero(
        (types == name) & np.isin(columns, bins))) for name in GROUPS}
    if any(len(group) == 0 for group in nodes.values()):
        raise AssertionError('missing local T5 targets')
    afferents = {}
    for name, group in nodes.items():
        edges = (net.tm4_arm_edges & torch.isin(net.post, group)).nonzero().flatten()
        lookup = torch.full((net.n,), -1, dtype=torch.long)
        lookup[group] = torch.arange(len(group))
        position = lookup[net.post[edges]]
        edge_weights = net.magnitudes[edges]
        mass = torch.zeros(len(group))
        mass.index_add_(0, position, edge_weights)
        afferents[name] = dict(pre=net.pre[edges], positions=position,
            weights=edge_weights, mass=mass, no_contacts=int((mass == 0).sum()))

    def capture(images):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        arrays = {name: {field: torch.zeros((144, len(group)))
                         for field in ('source_current', 'source_voltage',
                                       'source_spike', 'arm', 'gate')}
                  for name, group in nodes.items()}
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
                source_current = (net.feedforward_current[0]
                    +net.predictive_current[0]+net.behavioral_current[0])
                for name, group in nodes.items():
                    row = arrays[name]
                    index = net.t5_lookup[group]
                    row['arm'][t] = (now[0, index].abs()
                                     +delayed[0, index].abs())
                    row['gate'][t] = net.last_gate_current[index]
                    edge = afferents[name]
                    for key, source in (('source_current', source_current),
                                        ('source_voltage', net.voltage[0]),
                                        ('source_spike', activity.spikes[0].float())):
                        row[key][t].index_add_(0, edge['positions'],
                            edge['weights']*source[edge['pre']])
                        row[key][t] /= edge['mass'].clamp(min=1e-20)
        if not (torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        return pixel_events, arrays

    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    blank_count, blank = capture(bright)
    trials = {}
    for name, images in (('down', moving_bar('vertical', 16, 1, 2)),
                         ('up', moving_bar('vertical', 16, -1, 2)),
                         ('static', static_bar('vertical', 16))):
        count, arrays = capture(images)
        trials[name] = dict(pixel_events=count, arrays=arrays)
    if (blank_count != 0 or trials['up']['pixel_events']
            != trials['down']['pixel_events']):
        raise AssertionError('blank or opposite event-camera count mismatch')
    issue = torch.arange(3, 14)*8
    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS), center=16, speed=2,
        next_frame_gate_threshold=.05, tm4_arm_support_threshold=.001,
        source_current_deviation_threshold=.01,
        source_state='signed synaptic current, paired against bright blank',
        groups={}, passes_latent_tm4_screen=False)
    for condition, trial in trials.items():
        groups = report['groups'][condition] = {}
        for name in GROUPS:
            row = trial['arrays'][name]
            base = blank[name]
            event = torch.stack([row['gate'][t+8:t+16].max(0).values
                                 for t in issue]) > .05
            supported = row['arm'][issue] > .001
            values = dict(current_deviation=(row['source_current'][issue]
                -base['source_current'][issue]),
                voltage_deviation=(row['source_voltage'][issue]
                -base['source_voltage'][issue]),
                source_spike=row['source_spike'][issue])
            groups[name] = dict(cells=len(nodes[name]),
                cells_without_tm4_contacts=afferents[name]['no_contacts'],
                future_gate_events=int(event.sum()),
                missing_arm_events=category(values, event & ~supported),
                supported_arm_events=category(values, event & supported),
                quiet=category(values, ~event))
        print(json.dumps(dict(condition=condition, groups={name: dict(
            missing=row['missing_arm_events']['samples'],
            positive_deviation_fraction=row['missing_arm_events'].get(
                'positive_deviation_above_01_fraction'))
            for name, row in groups.items()})), flush=True)
    missing = report['groups']['up']['T5d']['missing_arm_events']
    quiet = report['groups']['up']['T5d']['quiet']
    report['passes_latent_tm4_screen'] = bool(missing['samples'] > 0
        and quiet['samples'] > 0
        and missing['positive_deviation_above_01_fraction'] > .5
        and missing['mean_positive_source_current_deviation']
            >= 2*quiet['mean_positive_source_current_deviation'])
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passes_latent_tm4_screen=report['passes_latent_tm4_screen'])), flush=True)


if __name__ == '__main__':
    main()

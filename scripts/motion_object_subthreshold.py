"""Frozen per-tick object responses in the measured T2/T2a/T3 pathway."""
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
from motion_stage_locality import ANNOTATIONS
from motion_t3_target import infer_t3_columns, static_dot


OUT = Path('docs/experiments/2026-09-23-object-pathway-subthreshold-results.json')
TRACES = Path('docs/experiments/2026-09-23-object-pathway-subthreshold-traces.npz')
CLASSES = ('T2', 'T2a', 'T3')
CENTERS = (18, 46)
TRANSIT = slice(3*8, 15*8)


def summarize_signal(stimulus, blank, local, remote):
    """Compare matched trials for one class, retaining a small causal trace."""
    def trace(values):
        return np.round(values.numpy().astype(np.float64), 7).tolist()

    result = {}
    for name in ('voltage', 'current'):
        change = stimulus[name] - blank[name]
        local_trace = change[:, local]
        remote_trace = change[:, remote]
        local_absolute = local_trace.abs().mean(1)
        local_signed = local_trace.mean(1)
        remote_absolute = remote_trace.abs().mean(1)
        transit_absolute = local_absolute[TRANSIT]
        result[name] = dict(
            local_mean_absolute=float(transit_absolute.mean()),
            local_mean_signed=float(local_signed[TRANSIT].mean()),
            remote_mean_absolute=float(remote_absolute[TRANSIT].mean()),
            peak_local_absolute=float(transit_absolute.max()),
            peak_tick=int(transit_absolute.argmax())+TRANSIT.start,
            local_cells_above_002=int((local_trace[TRANSIT].abs().mean(0) > .002).sum()),
            local_absolute_trace=trace(local_absolute),
            local_signed_trace=trace(local_signed),
            remote_absolute_trace=trace(remote_absolute))
    result['local_spikes'] = int(stimulus['spikes'][TRANSIT][:, local].sum())
    result['blank_local_spikes'] = int(blank['spikes'][TRANSIT][:, local].sum())
    result['local_neurons'] = int(local.sum())
    result['remote_neurons'] = int(remote.sum())
    result['pixel_events'] = stimulus['pixel_events']
    return result


def passes_object_screen(rows):
    """At each center/polarity, at least one class meets the fixed signal screen."""
    for center in CENTERS:
        for polarity in ('on', 'off'):
            if not any(
                (row := rows[f'{center}-{polarity}-dot'][cell_type])['voltage']
                ['local_mean_absolute'] >= .002
                and row['voltage']['local_mean_absolute'] >= 2 *
                    row['voltage']['remote_mean_absolute']
                and row['voltage']['local_cells_above_002'] >= 5
                for cell_type in CLASSES
            ):
                return False
    return True


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    retina = Retina(**metadata['retina'])
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    net = MultiContextEfficacyNetwork(
        graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_tick = net.step_index

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    types = np.asarray(metadata['retina']['cell_types'])
    columns, masses = {}, {}
    for cell_type in CLASSES:
        sources = ('Tm2', 'Mi1') if cell_type == 'T2' else ('Mi1', 'Tm1')
        columns[cell_type], masses[cell_type] = infer_t3_columns(
            metadata, retina, annotations, target_type=cell_type,
            source_types=sources)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    bins = {}
    for center in CENTERS:
        region = (x >= center-8) & (x <= center+8) & (y >= 13) & (y <= 19)
        bins[center] = retina.pixel_bins[region.flatten()].unique().numpy()
    masks = {cell_type: {center: (types == cell_type) &
             np.isin(columns[cell_type], bins[center]) for center in CENTERS}
             for cell_type in CLASSES}
    indices = np.flatnonzero(np.logical_or.reduce(
        [mask for by_center in masks.values() for mask in by_center.values()]))
    selected = torch.tensor(indices)
    map_to_selected = np.full(net.n, -1, dtype=np.int64)
    map_to_selected[indices] = np.arange(len(indices))
    local_columns = {cell_type: {center: map_to_selected[np.flatnonzero(
        masks[cell_type][center])] for center in CENTERS}
        for cell_type in CLASSES}
    if any(len(local_columns[cell_type][center]) == 0
           for cell_type in CLASSES for center in CENTERS):
        raise AssertionError('missing anatomically inferred local object cells')

    def probe(frames, polarity):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        voltage = torch.empty((144, len(indices)))
        current = torch.empty_like(voltage)
        spikes = torch.empty((144, len(indices)), dtype=torch.bool)
        pixel_events = 0
        step = 0
        for frame, image in enumerate(frames):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                voltage[step] = net.voltage[0, selected]
                current[step] = (net.feedforward_current[0, selected] +
                                 net.predictive_current[0, selected] +
                                 net.behavioral_current[0, selected])
                spikes[step] = activity.spikes[0, selected]
                step += 1
        if not torch.isfinite(voltage).all() or not torch.isfinite(current).all():
            raise AssertionError('non-finite object-pathway state')
        return dict(voltage=voltage, current=current, spikes=spikes,
                    pixel_events=pixel_events)

    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(),
                  warmup_ticks=warm_tick, ticks_per_frame=8,
                  inferred={cell_type: dict(
                      neurons=int((types == cell_type).sum()),
                      coverage=int(((columns[cell_type] >= 0) &
                                    (types == cell_type)).sum()),
                      median_input_contacts=float(np.median(
                          masses[cell_type][types == cell_type])),
                      local_cells={str(center): len(local_columns[cell_type][center])
                                   for center in CENTERS})
                      for cell_type in CLASSES},
                  conditions={}, passes_screen=False)
    blanks = {polarity: probe(
        [torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)]*18,
        polarity) for polarity in ('on', 'off')}
    for center in CENTERS:
        opposite = 46 if center == 18 else 18
        for polarity in ('on', 'off'):
            for kind in ('dot', 'bar', 'static'):
                frames = (static_dot(center, polarity) if kind == 'static'
                          else frames_for_condition(center, 1, polarity, kind=kind))
                observed = probe(frames, polarity)
                key = f'{center}-{polarity}-{kind}'
                report['conditions'][key] = {}
                for cell_type in CLASSES:
                    local = torch.tensor(local_columns[cell_type][center])
                    remote = torch.tensor(local_columns[cell_type][opposite])
                    report['conditions'][key][cell_type] = summarize_signal(
                        observed, blanks[polarity], local, remote)
                print(json.dumps(dict(condition=key,
                    local_voltage={label: report['conditions'][key][label]
                                   ['voltage']['local_mean_absolute']
                                   for label in CLASSES})), flush=True)
    report['passes_screen'] = passes_object_screen(report['conditions'])
    traces = {}
    for condition, groups in report['conditions'].items():
        for cell_type, row in groups.items():
            for signal in ('voltage', 'current'):
                for trace_name in ('local_absolute_trace', 'local_signed_trace',
                                   'remote_absolute_trace'):
                    key = f'{condition}/{cell_type}/{signal}/{trace_name}'
                    traces[key] = np.asarray(row[signal].pop(trace_name),
                                             dtype=np.float32)
    np.savez_compressed(TRACES, **traces)
    report['traces_file'] = str(TRACES)
    report['traces_sha256'] = checksum(TRACES)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), passes_screen=report['passes_screen'])),
          flush=True)


if __name__ == '__main__':
    main()

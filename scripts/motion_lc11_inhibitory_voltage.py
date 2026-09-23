"""Read-only dot/bar voltage signal in LC11's silent inhibitory sources."""
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
from motion_lc11_wiring import WEIGHTS, collect_target_edges
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from motion_t3_target import static_dot


OUT = Path('docs/experiments/2026-09-23-lc11-inhibitory-voltage-results.json')
CLASSES = ('Li15', 'MeLo10', 'Li26', 'TmY19a', 'TmY19b',
           'MeLo12', 'Li25', 'MeLo8', 'TmY15', 'Y14')
TRANSIT = slice(3*8, 15*8)


def summarize_class(stimulus, blank, source_mask, mass):
    change = stimulus['voltage'][TRANSIT][:, source_mask] - (
        blank['voltage'][TRANSIT][:, source_mask])
    weights = mass[source_mask]
    per_cell_signed = change.mean(0)
    per_cell_absolute = change.abs().mean(0)
    return dict(neurons=int(source_mask.sum()),
                contacts=int(weights.sum()),
                mean_absolute_voltage=float(per_cell_absolute.mean()),
                mean_signed_voltage=float(per_cell_signed.mean()),
                weighted_absolute_voltage=float(np.average(
                    per_cell_absolute.numpy(), weights=weights)),
                weighted_signed_voltage=float(np.average(
                    per_cell_signed.numpy(), weights=weights)),
                cells_above_002=int((per_cell_absolute > .002).sum()),
                positive_cells_above_002=int((per_cell_signed > .002).sum()),
                stimulus_spikes=int(stimulus['spikes'][TRANSIT][:, source_mask].sum()),
                blank_spikes=int(blank['spikes'][TRANSIT][:, source_mask].sum()),
                pixel_events=stimulus['pixel_events'])


def inhibitory_candidate(rows):
    for cell_type in CLASSES:
        for polarity in ('on', 'off'):
            if all(
                rows[f'{center}-{polarity}-bar'][cell_type]
                ['positive_cells_above_002'] >= 5
                and (rows[f'{center}-{polarity}-bar'][cell_type]
                     ['weighted_signed_voltage'] -
                     rows[f'{center}-{polarity}-dot'][cell_type]
                     ['weighted_signed_voltage']) >= .002
                for center in (18, 46)
            ):
                return dict(cell_type=cell_type, polarity=polarity)
    return None


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    ids = np.asarray(metadata['graph']['body_ids'], dtype=np.int64)
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'type', 'somaSide', 'status'])
    targets = np.array([int(row['bodyId']) for row in table.to_pylist()
                        if row['type'] == 'LC11' and row['somaSide'] == 'R'
                        and row['status'] == 'Traced'], dtype=np.int64)
    pre, post, contacts = collect_target_edges(WEIGHTS, targets)
    chosen_ids = ids[np.isin(types, CLASSES)]
    keep = np.isin(pre, chosen_ids) & np.isin(post, targets)
    pre, contacts = pre[keep], contacts[keep]
    sources = np.unique(pre)
    source_nodes = np.searchsorted(ids, sources)
    if not np.all(graph.signs[source_nodes] == -1):
        raise AssertionError('inhibitory source signs changed')
    mass = np.bincount(np.searchsorted(sources, pre), weights=contacts,
                       minlength=len(sources))
    masks = {cell_type: torch.tensor(types[source_nodes] == cell_type)
             for cell_type in CLASSES}
    if any(not mask.any() for mask in masks.values()):
        raise AssertionError('missing measured inhibitory source class')

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
    selected = torch.tensor(source_nodes)

    def capture(images, polarity):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        voltage = torch.empty((144, len(sources)))
        spikes = torch.empty((144, len(sources)), dtype=torch.bool)
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                voltage[step] = net.voltage[0, selected]
                spikes[step] = activity.spikes[0, selected]
                step += 1
        return dict(voltage=voltage, spikes=spikes,
                    pixel_events=pixel_events)

    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  weights_sha256=checksum(WEIGHTS),
                  graph_sha256=graph.identity(),
                  source_neurons=len(sources), contacts=int(contacts.sum()),
                  conditions={}, candidate=None)
    blanks = {polarity: capture(
        [torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)]*18,
        polarity) for polarity in ('on', 'off')}
    for center in (18, 46):
        for polarity in ('on', 'off'):
            for kind in ('dot', 'bar', 'static'):
                images = (static_dot(center, polarity) if kind == 'static'
                          else frames_for_condition(center, 1, polarity, kind=kind))
                observed = capture(images, polarity)
                key = f'{center}-{polarity}-{kind}'
                report['conditions'][key] = {
                    cell_type: summarize_class(observed, blanks[polarity],
                                               masks[cell_type], mass)
                    for cell_type in CLASSES}
                print(json.dumps(dict(condition=key,
                    weighted_signed={label: report['conditions'][key][label]
                                     ['weighted_signed_voltage']
                                     for label in CLASSES})), flush=True)
    report['candidate'] = inhibitory_candidate(report['conditions'])
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), candidate=report['candidate'])),
          flush=True)


if __name__ == '__main__':
    main()

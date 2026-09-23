"""Approved, frozen rest-current calibration for silent motion afferents."""
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


OUT = Path('runs/motion-stage-recovery-v1')
CLASSES = ('Mi4', 'Tm4', 'Tm9')
CURRENTS = (0., .85, .95, 1.05, 1.15)


def choose_current(rows):
    """Smallest candidate meeting the preregistered local/blank/arrival checks."""
    for row in rows:
        if (row['blank_rate'] < .01 and row['changed_local_cells'] >= 5
                and row['target_arrivals'] >= 1 and row['finite']):
            return row['rest_current']
    return None


def annotated_columns(metadata, retina, table, classes=CLASSES):
    """Place named, anatomically annotated afferents in the fixed retina grid."""
    types = np.asarray(metadata['retina']['cell_types'])
    ids = np.asarray(metadata['graph']['body_ids'])
    body, q, r = (table[k].to_numpy(zero_copy_only=False)
                  for k in ('bodyId', 'assignedOlHex1', 'assignedOlHex2'))
    locations = {int(b): (float(x), float(y)) for b, x, y in zip(body, q, r)
                 if np.isfinite(x) and np.isfinite(y)}
    selected = np.flatnonzero(np.isin(types, classes))
    hexes = np.asarray(retina.spec['hex_columns'])
    centers = np.column_stack((hexes[:, 0]+hexes[:, 1]/2,
                               hexes[:, 1]*math.sqrt(3)/2))
    inferred = np.full(len(ids), -1, dtype=np.int64)
    for batch in np.array_split(selected, max(1, math.ceil(len(selected)/500))):
        points = np.asarray([locations.get(int(ids[i]), (np.nan, np.nan))
                             for i in batch])
        valid = np.isfinite(points).all(1)
        xy = np.column_stack((points[valid, 0]+points[valid, 1]/2,
                              points[valid, 1]*math.sqrt(3)/2))
        inferred[batch[valid]] = ((xy[:, None, :]-centers[None, :, :])**2).sum(2).argmin(1)
    return inferred


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    graph = Graph(**m['graph'], gain=m['gain'])
    net = MultiContextEfficacyNetwork(graph, m['delays'], m['pathways'],
        config=NeuronConfig(**m['neurons']), cell_types=m['retina']['cell_types'],
        target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(0,
                    m['learning']['maximum_weight']))
    initial = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    base_rest = net.rest_current.clone()
    zero = torch.zeros_like(net.voltage)
    types = np.asarray(m['retina']['cell_types'])
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns = annotated_columns(m, retina, table)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    region = (x >= 10) & (x <= 26) & (y >= 9) & (y <= 23)
    bins = retina.pixel_bins[region.flatten()].unique().numpy()
    class_mask = {label: torch.tensor(types == label) for label in CLASSES}
    local_mask = {label: torch.tensor((types == label) & np.isin(columns, bins))
                  for label in CLASSES}
    target_mask = {'Mi4': torch.tensor(np.char.startswith(types.astype(str), 'T4')),
                   'Tm4': torch.tensor(np.char.startswith(types.astype(str), 'T5')),
                   'Tm9': torch.tensor(np.char.startswith(types.astype(str), 'T5'))}
    edge_masks = {label: class_mask[label][net.pre] & target_mask[label][net.post]
                  for label in CLASSES}
    if any(not local_mask[label].any() for label in CLASSES):
        raise AssertionError('annotated local afferents missing')

    def probe(warm, warm_tick, frames, polarity, label):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        spikes = torch.zeros(net.n, dtype=torch.int32)
        arrivals = 0
        recipients = torch.zeros(net.n, dtype=torch.bool)
        peak_fraction = 0.
        for frame, image in enumerate(frames):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                if 2 <= frame < 15:
                    spikes.add_(activity.spikes[0].int())
                    edges = activity.arrival_edges[edge_masks[label][activity.arrival_edges]]
                    arrivals += int(len(edges))
                    recipients[net.post[edges]] = True
                    peak_fraction = max(peak_fraction, float(activity.spikes.float().mean()))
        return dict(spikes=spikes, target_arrivals=arrivals,
                    recipient_cells=int(recipients.sum()), peak_fraction=peak_fraction,
                    finite=bool(torch.isfinite(net.voltage).all()
                                and torch.isfinite(net.feedforward_current).all()))

    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(), center=18,
                  currents=CURRENTS, results={}, selected={})
    for label in CLASSES:
        rows = []
        for current in CURRENTS:
            net.rest_current.copy_(base_rest)
            net.rest_current[class_mask[label]] = current
            for name, value in initial.items():
                getattr(net, name).copy_(value)
            net.step_index = 0
            for _ in range(m['config']['warmup_steps']):
                net.step(zero)
            warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
            tick = net.step_index
            blank_on = probe(warm, tick, [torch.zeros(1, 32, 64, dtype=torch.bool)]*18,
                             'on', label)
            blank_off = probe(warm, tick, [torch.ones(1, 32, 64, dtype=torch.bool)]*18,
                              'off', label)
            on = probe(warm, tick, frames_for_condition(18, 1, 'on', kind='bar'),
                       'on', label)
            off = probe(warm, tick, frames_for_condition(18, 1, 'off', kind='bar'),
                        'off', label)
            local = local_mask[label]
            changed_on = int((on['spikes'][local] != blank_on['spikes'][local]).sum())
            changed_off = int((off['spikes'][local] != blank_off['spikes'][local]).sum())
            blank_rate = max(int(blank_on['spikes'][class_mask[label]].sum()),
                             int(blank_off['spikes'][class_mask[label]].sum())) / (
                                 int(class_mask[label].sum())*13*8)
            rows.append(dict(rest_current=current, local_neurons=int(local.sum()),
                changed_local_cells=max(changed_on, changed_off),
                changed_on=changed_on, changed_off=changed_off,
                on_local_spikes=int(on['spikes'][local].sum()),
                off_local_spikes=int(off['spikes'][local].sum()),
                blank_local_spikes=int(blank_on['spikes'][local].sum()),
                blank_rate=blank_rate,
                target_arrivals=max(on['target_arrivals'], off['target_arrivals']),
                blank_target_arrivals=max(blank_on['target_arrivals'],
                                          blank_off['target_arrivals']),
                target_recipient_cells=max(on['recipient_cells'], off['recipient_cells']),
                peak_network_fraction=max(x['peak_fraction'] for x in
                                          (blank_on, blank_off, on, off)),
                finite=all(x['finite'] for x in (blank_on, blank_off, on, off))))
            print(json.dumps(dict(cell_class=label, **rows[-1])), flush=True)
        report['results'][label] = rows
        report['selected'][label] = choose_current(rows)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'calibration.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path), selected=report['selected'])), flush=True)


if __name__ == '__main__':
    main()

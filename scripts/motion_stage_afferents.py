"""Frozen local T4/T5 arrival and membrane audit for the fixed moving bar."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS, infer_columns
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--center', type=int, choices=(18, 46), default=18)
    args = parser.parse_args()
    center = args.center
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
    zero = torch.zeros_like(net.voltage)
    for _ in range(m['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_tick = net.step_index
    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(m, retina, annotations)
    types = np.asarray(m['retina']['cell_types'])
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    path = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
    bins = retina.pixel_bins[path.flatten()].unique().numpy()
    groups = {}
    for label in ('T4', 'T5'):
        groups[label] = torch.tensor(np.char.startswith(types.astype(str), label)
            & np.isin(columns, bins))
        if not groups[label].any():
            raise AssertionError(f'no local {label} cells')
    camera = EventCamera(1, 32, 64)
    cases = [('blank-on', None, 'on'), ('blank-off', None, 'off'),
             ('on-right', 1, 'on'),
             ('on-left', -1, 'on'), ('off-right', 1, 'off'), ('off-left', -1, 'off')]
    report = dict(source_sha256=source_sha, annotations_sha256=checksum(ANNOTATIONS),
        center=center, stimulus='3x13 moving bar', groups={
            label: dict(neurons=int(mask.sum())) for label, mask in groups.items()},
        cases={})
    for case, direction, polarity in cases:
        for name, state in warm.items():
            getattr(net, name).copy_(state)
        net.step_index = warm_tick
        camera.previous.fill_(polarity == 'off')
        frames = (frames_for_condition(center, direction, polarity, kind='bar')
                  if direction is not None else
                  [torch.full((1, 32, 64), polarity == 'off')]*18)
        stats = {label: dict(spikes=0, peak_voltage=-float('inf'),
            max_feedforward_current=-float('inf'),
            min_feedforward_current=float('inf'),
            arrival_edges=0, positive_arrival_magnitude=0.,
            negative_arrival_magnitude=0., afferent_arrivals={})
            for label in groups}
        for frame, image in enumerate(frames):
            events = camera.observe(image)
            injection = retina.project(events)*m['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                if not 2 <= frame < 15:
                    continue
                edges = activity.arrival_edges
                pretypes = types[net.pre[edges].numpy()]
                impulses = net.magnitudes[edges]*net.signs[edges]
                for label, mask in groups.items():
                    row = stats[label]
                    row['spikes'] += int(activity.spikes[0, mask].sum())
                    row['peak_voltage'] = max(row['peak_voltage'],
                                              float(net.voltage[0, mask].max()))
                    current = net.feedforward_current[0, mask]
                    row['max_feedforward_current'] = max(
                        row['max_feedforward_current'], float(current.max()))
                    row['min_feedforward_current'] = min(
                        row['min_feedforward_current'], float(current.min()))
                    selected = mask[net.post[edges]]
                    row['arrival_edges'] += int(selected.sum())
                    row['positive_arrival_magnitude'] += float(impulses[selected].clamp(min=0).sum())
                    row['negative_arrival_magnitude'] += float(impulses[selected].clamp(max=0).sum())
                    unique, counts = np.unique(pretypes[selected.numpy()], return_counts=True)
                    for name, count in zip(unique, counts):
                        row['afferent_arrivals'][str(name)] = (
                            row['afferent_arrivals'].get(str(name), 0)+int(count))
        report['cases'][case] = stats
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    path = Path(f'runs/motion-stage-audit-v1/afferents-{center}.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path), groups=report['groups'],
                          cases=report['cases'])), flush=True)


if __name__ == '__main__':
    main()

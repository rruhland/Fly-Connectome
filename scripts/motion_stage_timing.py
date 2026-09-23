"""Frozen per-tick audit of ON input timing and inferred Tm3 columns."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from full_context_efficacy import MultiContextEfficacyNetwork
from full_context_pong import NETWORK_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina


def infer_tm3_columns(metadata, retina):
    """Use only measured L1->Tm3 contacts and mapped L1 columns."""
    graph = metadata['graph']
    types = np.asarray(metadata['retina']['cell_types'])
    pre, post, contacts = (np.asarray(graph[k]) for k in ('pre', 'post', 'contacts'))
    source_columns = np.asarray(metadata['retina']['neuron_columns'])
    selected = ((types[pre] == 'L1') & (types[post] == 'Tm3')
                & (source_columns[pre] >= 0))
    hexes = np.asarray(retina.spec['hex_columns'])
    source_hex = hexes[source_columns[pre[selected]]]
    n = len(types)
    mass = np.bincount(post[selected], weights=contacts[selected], minlength=n)
    center = np.column_stack([
        np.bincount(post[selected], weights=contacts[selected]*source_hex[:, axis],
                    minlength=n)/np.maximum(mass, 1)
        for axis in (0, 1)])
    inferred = np.full(n, -1, dtype=np.int64)
    valid = np.flatnonzero((types == 'Tm3') & (mass > 0))
    inferred[valid] = ((center[valid, None, :]-hexes[None, :, :])**2).sum(2).argmin(1)
    spread = np.zeros(n)
    spread_weight = contacts[selected]*(
        (source_hex-center[post[selected]])**2).sum(1)
    spread[valid] = np.sqrt(np.bincount(post[selected], weights=spread_weight,
                                        minlength=n)[valid]/mass[valid])
    return inferred, mass, spread


def frames(center, kind):
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    positions = {'blank': (), 'flash': (center,),
                 'right': (center-1, center, center+1),
                 'left': (center+1, center, center-1)}[kind]
    output = [torch.zeros(1, 32, 64, dtype=torch.bool) for _ in range(8)]
    for frame, position in enumerate(positions, start=2):
        output[frame] = (((x-position).abs() <= 1)
                         & ((y-16).abs() <= 6)).unsqueeze(0)
    return output


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
    zero = torch.zeros_like(net.voltage)
    for _ in range(m['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    tick0 = net.step_index
    table = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns = annotated_columns(m, retina, table,
                                classes=('Mi4', 'Mi9', 'Mi1'))
    tm3, mass, spread = infer_tm3_columns(m, retina)
    columns[tm3 >= 0] = tm3[tm3 >= 0]
    t4, _ = infer_columns(m, retina, table)
    columns[t4 >= 0] = t4[t4 >= 0]
    types = np.asarray(m['retina']['cell_types'])
    tm3_mask = types == 'Tm3'
    report = dict(source_sha256=source_sha,
        annotations_sha256=checksum(ANNOTATIONS),
        tm3_column_inference=dict(total=int(tm3_mask.sum()),
            inferred=int(((tm3 >= 0) & tm3_mask).sum()),
            median_l1_contacts=float(np.median(mass[tm3_mask & (tm3 >= 0)])),
            median_l1_axial_spread=float(np.median(spread[tm3_mask & (tm3 >= 0)])),
            p90_l1_axial_spread=float(np.quantile(spread[tm3_mask & (tm3 >= 0)], .9))),
        probes={})
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    for center in (18, 46):
        region = (x >= center-5) & (x <= center+5) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[region.flatten()].unique().numpy()
        masks = {name: torch.tensor((types == name) & np.isin(columns, bins))
                 for name in ('Mi4', 'Mi9', 'Mi1', 'Tm3')}
        masks['T4'] = torch.tensor(np.char.startswith(types.astype(str), 'T4')
                                   & np.isin(columns, bins))
        target = masks['T4']
        if any(not mask.any() for mask in masks.values()):
            raise AssertionError('local timing group empty')
        for kind in ('blank', 'flash', 'right', 'left'):
            for name, value in warm.items():
                getattr(net, name).copy_(value)
            net.step_index = tick0
            camera = EventCamera(1, 32, 64)
            ticks = []
            for frame, image in enumerate(frames(center, kind)):
                events = camera.observe(image)
                injection = retina.project(events)*m['config']['sensory_gain']
                for tick in range(8):
                    activity = net.step(injection if tick == 0 else zero)
                    edges = activity.arrival_edges
                    selected = target[net.post[edges]]
                    edges = edges[selected]
                    source = types[net.pre[edges].numpy()]
                    signed = (net.magnitudes[edges]*net.signs[edges]).numpy()
                    arrivals = {label: dict(count=int((source == label).sum()),
                        signed_magnitude=float(signed[source == label].sum()))
                        for label in ('Mi4', 'Mi9', 'Mi1', 'Tm3')}
                    ticks.append(dict(frame=frame, tick=tick, pixel_events=(
                        int(len(events.pixels)) if tick == 0 else 0),
                        mi4_mean_voltage=float(net.voltage[0, masks['Mi4']].mean()),
                        t4_mean_voltage=float(net.voltage[0, target].mean()),
                        spikes={name: int(activity.spikes[0, mask].sum())
                                for name, mask in masks.items()},
                        arrivals=arrivals))
            report['probes'][f'{center}-{kind}'] = dict(
                local_neurons={name: int(mask.sum()) for name, mask in masks.items()},
                ticks=ticks)
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    path = Path('runs/motion-stage-signal-integration-v1/timing.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    summary = dict(source_sha256=source_sha, detailed_trace_sha256=checksum(path),
                   tm3_column_inference=report['tm3_column_inference'], probes={})
    for center in (18, 46):
        blank = report['probes'][f'{center}-blank']['ticks']
        for kind in ('flash', 'right', 'left'):
            ticks = report['probes'][f'{center}-{kind}']['ticks']
            difference = np.asarray([row['mi4_mean_voltage']-base['mi4_mean_voltage']
                                     for row, base in zip(ticks, blank)])
            summary['probes'][f'{center}-{kind}'] = dict(
                local_neurons=report['probes'][f'{center}-{kind}']['local_neurons'],
                minimum_mi4_voltage_difference=float(difference.min()),
                minimum_tick=int(difference.argmin()),
                maximum_mi4_voltage_difference=float(difference.max()),
                maximum_tick=int(difference.argmax()),
                t4_excess_spikes=sum(row['spikes']['T4']-base['spikes']['T4']
                                     for row, base in zip(ticks, blank)))
    summary_path = path.with_name('timing-summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path), tm3=report['tm3_column_inference'])), flush=True)


if __name__ == '__main__':
    main()

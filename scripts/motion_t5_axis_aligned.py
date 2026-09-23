"""Frozen axis-aligned T5 motion probes for the opt-in local current."""

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
from motion_stage_recovery import annotated_columns
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-axis-aligned-results.json')
STATE = NETWORK_STATE+GRADED_STATE+ORDER_STATE
AXES = {'horizontal': ((18, 46), 32, ('T5a', 'T5b')),
        'vertical': ((10, 22), 16, ('T5c', 'T5d'))}


def moving_bar(axis, center, direction, speed):
    if axis not in AXES or direction not in (-1, 1) or speed not in (1, 2):
        raise ValueError('known axis, direction and speed required')
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    frames = []
    for frame in range(18):
        if 2 <= frame < 15:
            position = center+direction*speed*(frame-8)
            if axis == 'horizontal':
                bar = ((x-position).abs() <= 1) & ((y-16).abs() <= 6)
            else:
                bar = ((y-position).abs() <= 1) & ((x-32).abs() <= 6)
            frames.append((~bar).unsqueeze(0))
        else:
            frames.append(torch.ones((1, 32, 64), dtype=torch.bool))
    return frames


def static_bar(axis, center):
    frames = moving_bar(axis, center, 1, 1)
    for frame in range(2, 15):
        frames[frame] = frames[8]
    return frames


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
    settled = {name: getattr(net, name).clone() for name in STATE}
    settled_tick = net.step_index

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    source_columns = annotated_columns(metadata, retina, annotations)
    target_columns, _ = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    local = {}
    for axis, (centers, holdout, subtypes) in AXES.items():
        for center in (*centers, holdout):
            if axis == 'horizontal':
                field = ((x >= center-8) & (x <= center+8)
                         & (y >= 9) & (y <= 23))
            else:
                field = ((y >= center-8) & (y <= center+8)
                         & (x >= 24) & (x <= 40))
            bins = retina.pixel_bins[field.flatten()].unique().numpy()
            local[(axis, center)] = dict(
                targets={subtype: torch.tensor(np.flatnonzero(
                    (types[net.t5_nodes.numpy()] == subtype)
                    & np.isin(target_columns[net.t5_nodes.numpy()], bins)))
                    for subtype in subtypes},
                sources=torch.tensor(np.flatnonzero(np.isin(
                    source_columns[net.graded_nodes.numpy()], bins))))
            group = local[(axis, center)]
            if not len(group['sources']) or any(
                    not len(indices) for indices in group['targets'].values()):
                raise AssertionError('missing axis-aligned local sources or T5 cells')

    def capture(images, axis, center, *, with_gate=True):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        net.order_enabled = with_gate
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        spikes = torch.zeros((144, len(net.t5_nodes)), dtype=torch.bool)
        gates = torch.zeros((144, len(net.t5_nodes)))
        release = torch.zeros(144)
        pixel_events = 0
        step = 0
        group = local[(axis, center)]
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                spikes[step] = activity.spikes[0, net.t5_nodes]
                gates[step] = net.last_gate_current
                release[step] = net.release_history[
                    (net.step_index-1) % net.history_length,
                    group['sources']].mean()
                step += 1
        if not (torch.isfinite(net.voltage).all()
                and torch.isfinite(gates).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        targets = {}
        for subtype, indices in group['targets'].items():
            count = spikes[24:120, indices].sum(0).int()
            current = gates[24:120, indices]
            targets[subtype] = dict(spikes=int(count.sum()),
                per_cell=count.tolist(),
                gate_fraction=float((current > 0).float().mean()),
                gate_mean=float(current.mean()),
                gate_p95=float(torch.quantile(current, .95)))
        return dict(pixel_events=pixel_events, targets=targets,
            blank_t5_rate=float(spikes[16:].float().mean()),
            mean_source_release=float(release[24:120].mean()))

    def evaluate(axis, center, speed, *, with_gate=True):
        subtypes = AXES[axis][2]
        cases = dict(blank=capture([torch.ones((1, 32, 64), dtype=torch.bool)]*18,
                                   axis, center, with_gate=with_gate),
            positive=capture(moving_bar(axis, center, 1, speed), axis, center,
                             with_gate=with_gate),
            negative=capture(moving_bar(axis, center, -1, speed), axis, center,
                             with_gate=with_gate),
            static=capture(static_bar(axis, center), axis, center,
                           with_gate=with_gate))
        if cases['positive']['pixel_events'] != cases['negative']['pixel_events']:
            raise AssertionError('opposite directions have unequal camera events')
        contrasts = {}
        for subtype in subtypes:
            positive = np.asarray(cases['positive']['targets'][subtype]['per_cell'])
            negative = np.asarray(cases['negative']['targets'][subtype]['per_cell'])
            difference = positive-negative
            preferred = max(int(positive.sum()), int(negative.sum()))
            static_count = cases['static']['targets'][subtype]['spikes']
            contrasts[subtype] = dict(
                positive_minus_negative=int(difference.sum()),
                per_cell_median=float(np.median(difference)),
                positive_cells=int((difference > 0).sum()),
                negative_cells=int((difference < 0).sum()),
                preferred_spikes=preferred, static_spikes=static_count,
                moving_over_static=preferred/max(static_count, 1))
        for case in cases.values():
            for entry in case['targets'].values():
                counts = np.asarray(entry.pop('per_cell'))
                values, frequencies = np.unique(counts, return_counts=True)
                entry['per_cell_spike_histogram'] = {
                    str(int(value)): int(frequency)
                    for value, frequency in zip(values, frequencies)}
        result = dict(axis=axis, center=center, speed=speed,
                      with_gate=with_gate, cases=cases,
                      contrasts=contrasts,
                      blank_safe=cases['blank']['blank_t5_rate'] < .001)
        print(json.dumps(dict(axis=axis, center=center, speed=speed,
            with_gate=with_gate,
            contrasts={subtype: contrasts[subtype]['positive_minus_negative']
                       for subtype in subtypes}, blank_safe=result['blank_safe'])),
            flush=True)
        return result

    def passes(row, signs=None):
        if not row['blank_safe']:
            return False
        values = [row['contrasts'][subtype]['positive_minus_negative']
                  for subtype in AXES[row['axis']][2]]
        if values[0]*values[1] >= 0:
            return False
        for subtype, value in zip(AXES[row['axis']][2], values):
            entry = row['contrasts'][subtype]
            if abs(value) < 10 or entry['moving_over_static'] < 1.5:
                return False
            majority = entry['positive_cells']-entry['negative_cells']
            if majority*value <= 0:
                return False
            if signs is not None and value*signs[subtype] <= 0:
                return False
        return True

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS),
        ticks_per_frame=8, lag_ticks=8, gate_gain=8000., gate_cap=1.,
        source_release_cap=.10, current_scale=.02,
        rest_overrides={'Tm4': .85, 'Tm9': .85}, axes={})
    for axis, (centers, holdout, subtypes) in AXES.items():
        calibration = [evaluate(axis, center, 1) for center in centers]
        first = calibration[0]
        signs = {subtype: int(np.sign(
            first['contrasts'][subtype]['positive_minus_negative']))
                 for subtype in subtypes}
        passed = (passes(first) and passes(calibration[1], signs))
        row = report['axes'][axis] = dict(calibration=calibration,
            calibration_signs=signs, passes_calibration=passed,
            holdout=[], passes_holdout=False)
        if passed:
            row['holdout'] = [evaluate(axis, holdout, speed) for speed in (1, 2)]
            row['passes_holdout'] = all(passes(item, signs) for item in row['holdout'])
    report['gate_off_vertical_control'] = [
        evaluate('vertical', center, speed, with_gate=False)
        for center, speed in ((10, 1), (22, 1), (16, 1), (16, 2))]
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), screens={axis: dict(
        calibration=row['passes_calibration'], holdout=row['passes_holdout'])
        for axis, row in report['axes'].items()})), flush=True)


if __name__ == '__main__':
    main()

"""Opt-in causal T5 membrane-current test of the fixed Tm4/Tm9 order cue."""

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
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_stage_recovery import annotated_columns
from motion_tm9_graded import moving_bar_frames
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-local-order-current-results.json')
GAINS = (500., 2000., 8000.)
SUBTYPES = ('T5c', 'T5d')
STATE = NETWORK_STATE+GRADED_STATE+ORDER_STATE
BLANK = [torch.ones((1, 32, 64), dtype=torch.bool)]*18


def stimulus(center, direction, speed):
    if speed == 1:
        return frames_for_condition(center, direction, 'off', kind='bar')
    return moving_bar_frames(center, direction, speed)


def static_bar(center):
    frames = frames_for_condition(center, 1, 'off', kind='bar')
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
        source_type='Tm9', source_state='current',
        gate_gain=GAINS[0], gate_cap=1.)
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
    for center in (18, 46, 32):
        field = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        local[center] = dict(
            targets={subtype: torch.tensor(np.flatnonzero(
                (types[net.t5_nodes.numpy()] == subtype)
                & np.isin(target_columns[net.t5_nodes.numpy()], bins)))
                for subtype in SUBTYPES},
            sources=torch.tensor(np.flatnonzero(np.isin(
                source_columns[net.graded_nodes.numpy()], bins))))
        if not len(local[center]['sources']) or any(
                not len(indices) for indices in local[center]['targets'].values()):
            raise AssertionError('missing anatomically local source or T5 group')

    def capture(images, center, *, with_gate=True):
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
                    local[center]['sources']].mean()
                step += 1
        if not (torch.isfinite(gates).all() and torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        groups = {}
        for subtype, indices in local[center]['targets'].items():
            per_cell = spikes[24:120, indices].sum(0).int()
            local_gate = gates[24:120, indices]
            groups[subtype] = dict(spikes=int(per_cell.sum()),
                per_cell_spikes=per_cell.tolist(),
                per_frame_spikes=spikes[:, indices].reshape(
                    18, 8, len(indices)).sum((1, 2)).int().tolist(),
                gate_fraction=float((local_gate > 0).float().mean()),
                gate_p95=float(torch.quantile(local_gate, .95)),
                gate_mean=float(local_gate.mean()))
        return dict(pixel_events=pixel_events, groups=groups,
            blank_all_t5_rate=float(spikes[16:].float().mean()),
            mean_source_release=float(release[24:120].mean()))

    def evaluate(gain, center, speed):
        net.gate_gain = gain
        prefix = f'{center}-s{speed}'
        cases = dict(blank=capture(BLANK, center),
            right=capture(stimulus(center, 1, speed), center),
            left=capture(stimulus(center, -1, speed), center),
            static=capture(static_bar(center), center))
        if cases['right']['pixel_events'] != cases['left']['pixel_events']:
            raise AssertionError('opposite motions have unequal pixel event counts')
        contrast = {}
        for subtype in SUBTYPES:
            groups = {name: case['groups'][subtype] for name, case in cases.items()}
            per_cell = np.asarray(groups['right']['per_cell_spikes'])-np.asarray(
                groups['left']['per_cell_spikes'])
            contrast[subtype] = dict(
                right_minus_left=int(groups['right']['spikes']-groups['left']['spikes']),
                per_cell_median=float(np.median(per_cell)),
                positive_cells=int((per_cell > 0).sum()),
                negative_cells=int((per_cell < 0).sum()),
                static_excess=int(groups['static']['spikes']-groups['blank']['spikes']))
        for case in cases.values():
            for group in case['groups'].values():
                counts = np.asarray(group.pop('per_cell_spikes'))
                values, frequencies = np.unique(counts, return_counts=True)
                group['per_cell_spike_histogram'] = {
                    str(int(value)): int(frequency)
                    for value, frequency in zip(values, frequencies)}
        row = dict(gain=gain, center=center, speed=speed,
            blank_safe=cases['blank']['blank_all_t5_rate'] < .001,
            cases=cases, contrast=contrast)
        print(json.dumps(dict(condition=prefix, gain=gain,
            blank_rate=cases['blank']['blank_all_t5_rate'],
            contrasts={s: contrast[s]['right_minus_left'] for s in SUBTYPES})),
            flush=True)
        return row

    def passes(row, *, holdout):
        if not row['blank_safe']:
            return False
        for subtype, sign in (('T5c', -1), ('T5d', 1)):
            entry = row['contrast'][subtype]
            value = entry['right_minus_left']
            if value*sign < 5 or abs(value) <= abs(entry['static_excess']):
                return False
            if holdout and entry['per_cell_median']*sign <= 0:
                return False
        return True

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS),
        ticks_per_frame=8, lag_ticks=8, source_release_cap=.10,
        current_scale=.02, rest_overrides={'Tm4': .85, 'Tm9': .85},
        gate_cap=1., tested_gains=list(GAINS),
        local_cells={str(c): {s: len(local[c]['targets'][s]) for s in SUBTYPES}
                     for c in local},
        gate_off_calibration={}, calibration=[], selected_gain=None, holdout=[],
        passes_calibration=False, passes_holdout=False)
    for center in (18, 46):
        gate_off = dict(
            blank=capture(BLANK, center, with_gate=False),
            right=capture(stimulus(center, 1, 1), center, with_gate=False),
            left=capture(stimulus(center, -1, 1), center, with_gate=False),
            static=capture(static_bar(center), center, with_gate=False))
        for case in gate_off.values():
            for group in case['groups'].values():
                counts = np.asarray(group.pop('per_cell_spikes'))
                values, frequencies = np.unique(counts, return_counts=True)
                group['per_cell_spike_histogram'] = {
                    str(int(value)): int(frequency)
                    for value, frequency in zip(values, frequencies)}
        report['gate_off_calibration'][str(center)] = gate_off
    for gain in GAINS:
        rows = [evaluate(gain, center, 1) for center in (18, 46)]
        report['calibration'].append(dict(gain=gain, rows=rows,
            passed=all(passes(row, holdout=False) for row in rows)))
        if report['calibration'][-1]['passed']:
            report['selected_gain'] = gain
            report['passes_calibration'] = True
            break
    if report['passes_calibration']:
        report['holdout'] = [evaluate(report['selected_gain'], 32, speed)
                             for speed in (1, 2)]
        report['passes_holdout'] = all(passes(row, holdout=True)
                                       for row in report['holdout'])
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        selected_gain=report['selected_gain'],
        passes_calibration=report['passes_calibration'],
        passes_holdout=report['passes_holdout'])), flush=True)


if __name__ == '__main__':
    main()

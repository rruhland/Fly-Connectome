"""Frozen source-local current/voltage screen for measured T4/T5 afferents."""

import json
import math
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
from motion_source_signal_audit import FEATURES, summarize_feature
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from motion_stage_recovery import annotated_columns


OUT = Path('docs/experiments/2026-09-23-motion-source-current-results.json')
CLASSES = ('Mi4', 'Tm4', 'Tm9')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    retina = Retina(**metadata['retina'])
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    net = MultiContextEfficacyNetwork(graph, metadata['delays'],
        metadata['pathways'], config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone() for name in NETWORK_STATE}
    warm_tick = net.step_index

    types = np.asarray(metadata['retina']['cell_types'])
    selected_nodes = np.flatnonzero(np.isin(types, CLASSES))
    selected = torch.tensor(selected_nodes)
    lookup = np.full(net.n, -1, dtype=np.int64)
    lookup[selected_nodes] = np.arange(len(selected_nodes))
    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns = annotated_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    regions = {}
    for center in (18, 46):
        field = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        regions[center] = {cell_type: torch.tensor(lookup[
            np.flatnonzero((types == cell_type) & np.isin(columns, bins))])
            for cell_type in CLASSES}
        if any(len(regions[center][cell_type]) == 0 for cell_type in CLASSES):
            raise AssertionError('missing annotated local motion source cells')

    decay = math.exp(-net.config.dt/.250)

    def capture(images, polarity):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        baseline_v = net.voltage[0, selected].clone()
        baseline_i = (net.feedforward_current[0, selected]
                      +net.predictive_current[0, selected]
                      +net.behavioral_current[0, selected])
        traces = {feature: torch.empty((144, len(selected_nodes)))
                  for feature in FEATURES}
        spikes = torch.zeros(len(selected_nodes), dtype=torch.int32)
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                voltage = net.voltage[0, selected]
                current = (net.feedforward_current[0, selected]
                           +net.predictive_current[0, selected]
                           +net.behavioral_current[0, selected])
                fast_v, fast_i = voltage-baseline_v, current-baseline_i
                traces['voltage'][step] = voltage
                traces['current'][step] = current
                traces['positive_fast_voltage'][step] = fast_v.clamp(min=0)
                traces['negative_fast_voltage'][step] = (-fast_v).clamp(min=0)
                traces['positive_fast_current'][step] = fast_i.clamp(min=0)
                traces['negative_fast_current'][step] = (-fast_i).clamp(min=0)
                baseline_v.mul_(decay).add_(voltage, alpha=1-decay)
                baseline_i.mul_(decay).add_(current, alpha=1-decay)
                if 3 <= frame < 15:
                    spikes += activity.spikes[0, selected].int()
                step += 1
        if any(not torch.isfinite(trace).all() for trace in traces.values()):
            raise AssertionError('non-finite motion source state')
        return dict(traces=traces, spikes=spikes, pixel_events=pixel_events)

    blanks = {polarity: capture(
        [torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)]*18,
        polarity) for polarity in ('on', 'off')}
    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(),
                  warmup_ticks=warm_tick, ticks_per_frame=8,
                  fast_baseline_ms=250, conditions={}, candidate_screens={})
    for center in (18, 46):
        other = 46 if center == 18 else 18
        for polarity in ('on', 'off'):
            for kind in ('bar', 'static'):
                frames = frames_for_condition(center, 1, polarity, kind='bar')
                if kind == 'static':
                    for frame in range(2, 15):
                        frames[frame] = frames[8]
                observed = capture(frames, polarity)
                row = dict(pixel_events=observed['pixel_events'], classes={})
                for cell_type in CLASSES:
                    local = regions[center][cell_type]
                    remote = regions[other][cell_type]
                    group = dict(local_neurons=len(local),
                                 remote_neurons=len(remote),
                                 local_spikes=int(observed['spikes'][local].sum()),
                                 blank_local_spikes=int(blanks[polarity]['spikes'][local].sum()),
                                 features={})
                    for feature in FEATURES:
                        result = summarize_feature(
                            observed['traces'][feature],
                            blanks[polarity]['traces'][feature], local, remote)
                        difference = (observed['traces'][feature][:, local]
                                      -blanks[polarity]['traces'][feature][:, local])
                        result['peak_positive_tick'] = int(
                            difference.mean(1).argmax())
                        group['features'][feature] = result
                    row['classes'][cell_type] = group
                report['conditions'][f'{center}-{polarity}-{kind}'] = row
                print(json.dumps(dict(condition=f'{center}-{polarity}-{kind}',
                    Mi4_current=row['classes']['Mi4']['features']['current']
                                  ['local_mean_absolute_change'])), flush=True)
    for cell_type in CLASSES:
        polarity = 'on' if cell_type == 'Mi4' else 'off'
        for feature in FEATURES[2:]:
            rows = [report['conditions'][f'{center}-{polarity}-bar']
                    ['classes'][cell_type]['features'][feature]
                    for center in (18, 46)]
            report['candidate_screens'][f'{cell_type}/{feature}'] = bool(all(
                row['finite']
                and row['local_fraction_above_blank_p99'] >= .05
                and row['blank_fraction_above_blank_p99'] <= .02
                and row['local_mean_absolute_change'] >=
                    2*row['remote_mean_absolute_change']
                for row in rows))
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
        passing_candidates=[name for name, passed in
                            report['candidate_screens'].items() if passed])),
        flush=True)


if __name__ == '__main__':
    main()

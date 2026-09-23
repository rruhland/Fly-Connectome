"""Frozen source-local release-signal audit for measured object pathways."""

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
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from motion_t3_target import infer_t3_columns, static_dot


OUT = Path('docs/experiments/2026-09-23-source-signal-results.json')
CLASSES = ('T2', 'T2a', 'T3')
CENTERS = (18, 46)
FEATURES = ('voltage', 'current', 'positive_fast_voltage',
            'negative_fast_voltage', 'positive_fast_current',
            'negative_fast_current')


def summarize_feature(stimulus, blank, local, remote, *, start=24, end=120):
    local_stimulus = stimulus[start:end, local]
    local_blank = blank[start:end, local]
    local_change = local_stimulus-local_blank
    remote_change = stimulus[start:end, remote]-blank[start:end, remote]
    floor = torch.quantile(local_blank, .99, dim=0)
    per_cell = local_change.mean(0)
    return dict(
        local_mean_absolute_change=float(local_change.abs().mean()),
        local_mean_signed_change=float(local_change.mean()),
        remote_mean_absolute_change=float(remote_change.abs().mean()),
        local_fraction_above_blank_p99=float((local_stimulus > floor).float().mean()),
        blank_fraction_above_blank_p99=float((local_blank > floor).float().mean()),
        per_cell_signed_change_p10=float(torch.quantile(per_cell, .1)),
        per_cell_signed_change_p50=float(torch.quantile(per_cell, .5)),
        per_cell_signed_change_p90=float(torch.quantile(per_cell, .9)),
        finite=bool(torch.isfinite(stimulus).all() and torch.isfinite(blank).all()))


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

    types = np.asarray(metadata['retina']['cell_types'])
    indices = np.flatnonzero(np.isin(types, CLASSES))
    selected = torch.tensor(indices)
    lookup = np.full(net.n, -1, dtype=np.int64)
    lookup[indices] = np.arange(len(indices))
    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    bins = {}
    for center in CENTERS:
        region = (x >= center-8) & (x <= center+8) & (y >= 13) & (y <= 19)
        bins[center] = retina.pixel_bins[region.flatten()].unique().numpy()
    regions = {}
    for cell_type in CLASSES:
        source_types = ('Tm2', 'Mi1') if cell_type == 'T2' else ('Mi1', 'Tm1')
        columns, _ = infer_t3_columns(metadata, retina, annotations,
            target_type=cell_type, source_types=source_types)
        regions[cell_type] = {
            center: torch.tensor(lookup[np.flatnonzero(
                (types == cell_type) & np.isin(columns, bins[center]))])
            for center in CENTERS}
        if any(len(regions[cell_type][center]) == 0 for center in CENTERS):
            raise AssertionError(f'missing local {cell_type} source cells')

    decay = math.exp(-net.config.dt/.250)

    def capture(images, polarity):
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        baseline_v = net.voltage[0, selected].clone()
        baseline_i = (net.feedforward_current[0, selected]
                      + net.predictive_current[0, selected]
                      + net.behavioral_current[0, selected])
        traces = {feature: torch.empty((144, len(indices)))
                  for feature in FEATURES}
        source_spikes = torch.zeros(len(indices), dtype=torch.int32)
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
                           + net.predictive_current[0, selected]
                           + net.behavioral_current[0, selected])
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
                    source_spikes += activity.spikes[0, selected].int()
                step += 1
        if any(not torch.isfinite(trace).all() for trace in traces.values()):
            raise AssertionError('non-finite source-local state')
        return dict(traces=traces, spikes=source_spikes,
                    pixel_events=pixel_events)

    blanks = {polarity: capture(
        [torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)]*18,
        polarity) for polarity in ('on', 'off')}
    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(), warmup_ticks=warm_tick,
                  ticks_per_frame=8, fast_baseline_ms=250,
                  conditions={}, candidate_screens={})
    for center in CENTERS:
        other = 46 if center == 18 else 18
        for polarity in ('on', 'off'):
            for kind in ('dot', 'bar', 'static'):
                frames = (static_dot(center, polarity) if kind == 'static'
                          else frames_for_condition(center, 1, polarity,
                                                    kind=kind))
                observed = capture(frames, polarity)
                row = dict(pixel_events=observed['pixel_events'], classes={})
                for cell_type in CLASSES:
                    local, remote = (regions[cell_type][center],
                                     regions[cell_type][other])
                    group = dict(local_neurons=len(local),
                                 remote_neurons=len(remote),
                                 local_spikes=int(observed['spikes'][local].sum()),
                                 blank_local_spikes=int(blanks[polarity]['spikes'][local].sum()),
                                 features={})
                    for feature in FEATURES:
                        group['features'][feature] = summarize_feature(
                            observed['traces'][feature],
                            blanks[polarity]['traces'][feature], local, remote)
                    row['classes'][cell_type] = group
                report['conditions'][f'{center}-{polarity}-{kind}'] = row
                print(json.dumps(dict(condition=f'{center}-{polarity}-{kind}',
                    T2_voltage=row['classes']['T2']['features']['voltage']
                                   ['local_mean_absolute_change'])), flush=True)
    for cell_type in CLASSES:
        for feature in FEATURES[2:]:
            cases = [report['conditions'][f'{center}-{polarity}-dot']
                     ['classes'][cell_type]['features'][feature]
                     for center in CENTERS for polarity in ('on', 'off')]
            report['candidate_screens'][f'{cell_type}/{feature}'] = bool(all(
                row['finite']
                and row['local_fraction_above_blank_p99'] >= .05
                and row['local_mean_absolute_change'] >=
                    2*row['remote_mean_absolute_change']
                for row in cases))
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT),
                          passing_candidates=[key for key, passed in
                              report['candidate_screens'].items() if passed])),
          flush=True)


if __name__ == '__main__':
    main()

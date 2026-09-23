"""Blank-calibrated T2 graded-output test across measured visual edges."""

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
from full_context_pong import NETWORK_STATE
from graded_visual import GradedVisualNetwork
from motion_source_signal_audit import summarize_feature
from motion_stage_audit import SOURCE, frames_for_condition
from motion_stage_locality import ANNOTATIONS
from motion_t3_target import infer_t3_columns, static_dot


OUT = Path('docs/experiments/2026-09-23-t2-graded-propagation-results.json')
LI15_OUT = Path('docs/experiments/2026-09-23-li15-local-release-results.json')
LI15_LOCAL_OUT = Path('docs/experiments/2026-09-23-li15-afferent-locality-results.json')
CAPS = (.01, .03, .10)
TARGET_TYPES = ('Li15', 'MeLo10', 'TmY19a', 'MeLo8')
GRADED_STATE = ('release_history', 'graded_baseline', 'last_graded_impulse')


def select_cap(rows):
    passing = [row['cap'] for row in rows
               if row['finite'] and row['blank_p99_impulse'] <= .05
               and row['blank_spike_rate'] < .01]
    return max(passing, default=None)


def summarize_target(stimulus, blank, indices, *, start=24, end=120):
    result = {}
    for name in ('impulse', 'current', 'voltage'):
        change = (stimulus[name][start:end, indices]
                  - blank[name][start:end, indices])
        per_cell = change.abs().mean(0)
        result[name] = dict(
            mean_signed_change=float(change.mean()),
            mean_absolute_change=float(change.abs().mean()),
            per_cell_absolute_p10=float(torch.quantile(per_cell, .1)),
            per_cell_absolute_p50=float(torch.quantile(per_cell, .5)),
            per_cell_absolute_p90=float(torch.quantile(per_cell, .9)))
    result['stimulus_spikes'] = int(stimulus['spikes'][start:end, indices].sum())
    result['blank_spikes'] = int(blank['spikes'][start:end, indices].sum())
    return result


def causal_highpass(trace, initial, *, decay):
    baseline = initial.clone()
    positive = torch.empty_like(trace)
    negative = torch.empty_like(trace)
    for tick, values in enumerate(trace):
        difference = values-baseline
        positive[tick] = difference.clamp(min=0)
        negative[tick] = (-difference).clamp(min=0)
        baseline.mul_(decay).add_(values, alpha=1-decay)
    return positive, negative


def summarize_local_candidate(stimulus, blank, indices, *, start=24, end=120):
    observed = stimulus[start:end, indices]
    reference = blank[start:end, indices]
    difference = observed-reference
    floor = torch.quantile(reference, .99, dim=0)
    per_cell = difference.mean(0)
    return dict(
        neurons=len(indices),
        fraction_above_blank_p99=float((observed > floor).float().mean()),
        blank_fraction_above_blank_p99=float((reference > floor).float().mean()),
        mean_absolute_change=float(difference.abs().mean()),
        mean_signed_change=float(difference.mean()),
        per_cell_signed_p10=float(torch.quantile(per_cell, .1)),
        per_cell_signed_p50=float(torch.quantile(per_cell, .5)),
        per_cell_signed_p90=float(torch.quantile(per_cell, .9)),
        finite=bool(torch.isfinite(observed).all()
                    and torch.isfinite(reference).all()))


def dominant_target_masks(pre, post, contacts, source_regions, target_mask):
    targets = np.flatnonzero(target_mask)
    masses = {}
    for center, sources in source_regions.items():
        selected = sources[pre] & target_mask[post]
        all_mass = np.bincount(post[selected], weights=contacts[selected],
                               minlength=len(target_mask))
        masses[center] = all_mass[targets]
    centers = tuple(source_regions)
    if len(centers) != 2:
        raise ValueError('compare exactly two anatomical source regions')
    groups = {center: targets[masses[center] > masses[other]]
              for center, other in ((centers[0], centers[1]),
                                    (centers[1], centers[0]))}
    return groups, masses


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = GradedVisualNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=CAPS[0], voltage_scale=.01)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    original_weights = net.magnitudes.clone()
    if (len(net.graded_edges) != int((types[graph.pre] == 'T2').sum())
            or not (net.signs[net.graded_edges] == 1).all()):
        raise AssertionError('T2 edge roster or recorded transmitter signs changed')
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    warm = {name: getattr(net, name).clone()
            for name in NETWORK_STATE+GRADED_STATE}
    warm_tick = net.step_index
    warm_blank_residual = net.blank_residual.clone()
    warm_blank_count = net.blank_count

    target_indices = np.flatnonzero(np.isin(types, TARGET_TYPES))
    selected_targets = torch.tensor(target_indices)
    target_lookup = np.full(net.n, -1, dtype=np.int64)
    target_lookup[target_indices] = np.arange(len(target_indices))
    target_groups = {cell_type: torch.tensor(target_lookup[
        np.flatnonzero(types == cell_type)]) for cell_type in TARGET_TYPES}
    li15 = torch.tensor(types == 'Li15')
    if not li15.any():
        raise AssertionError('no measured Li15 target cells')

    calibration = []
    for cap in CAPS:
        for name, value in warm.items():
            getattr(net, name).copy_(value)
        net.step_index = warm_tick
        net.blank_residual.copy_(warm_blank_residual)
        net.blank_count = warm_blank_count
        net.graded_enabled = False
        net.release_cap = cap
        net.enable_graded()
        impulses = []
        blank_spikes = 0
        max_global_fraction = 0.
        for tick in range(256):
            activity = net.step(zero)
            if tick >= 128:
                impulses.append(net.last_graded_impulse[0, li15].abs().clone())
                blank_spikes += int(activity.spikes[0, li15].sum())
                max_global_fraction = max(max_global_fraction,
                    float(activity.spikes.float().mean()))
        row = dict(cap=cap,
            blank_p99_impulse=float(torch.quantile(torch.stack(impulses), .99)),
            blank_spike_rate=blank_spikes/(128*int(li15.sum())),
            max_global_spike_fraction=max_global_fraction,
            finite=bool(torch.isfinite(net.voltage).all()
                        and torch.isfinite(net.last_graded_impulse).all()))
        calibration.append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    selected_cap = select_cap(calibration)
    report = dict(source_sha256=source_sha,
                  annotations_sha256=checksum(ANNOTATIONS),
                  graph_sha256=graph.identity(), graded_edges=len(net.graded_edges),
                  graded_contacts=int(graph.contacts[net.graded_edges.numpy()].sum()),
                  graded_source_cells=len(net.graded_nodes),
                  warmup_ticks=warm_tick, settle_ticks=256,
                  ticks_per_frame=8, voltage_scale=.01,
                  calibration=calibration, selected_cap=selected_cap,
                  conditions={}, partial_t2_li15_screen=False)
    if selected_cap is None:
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(OUT), selected_cap=None)), flush=True)
        return

    for name, value in warm.items():
        getattr(net, name).copy_(value)
    net.step_index = warm_tick
    net.blank_residual.copy_(warm_blank_residual)
    net.blank_count = warm_blank_count
    net.graded_enabled = False
    net.release_cap = selected_cap
    net.enable_graded()
    for _ in range(256):
        net.step(zero)
    settled = {name: getattr(net, name).clone()
               for name in NETWORK_STATE+GRADED_STATE}
    settled_tick = net.step_index

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_t3_columns(metadata, retina, annotations,
                                 target_type='T2', source_types=('Tm2', 'Mi1'))
    source_lookup = np.full(net.n, -1, dtype=np.int64)
    source_lookup[net.graded_nodes.numpy()] = np.arange(len(net.graded_nodes))
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    regions = {}
    source_regions = {}
    for center in (18, 46):
        region = (x >= center-8) & (x <= center+8) & (y >= 13) & (y <= 19)
        bins = retina.pixel_bins[region.flatten()].unique().numpy()
        source_regions[center] = (types == 'T2') & np.isin(columns, bins)
        regions[center] = torch.tensor(source_lookup[
            np.flatnonzero(source_regions[center])])
    li15_groups, li15_mass = dominant_target_masks(
        graph.pre, graph.post, graph.contacts, source_regions,
        types == 'Li15')
    local_li15 = {center: torch.tensor(target_lookup[li15_groups[center]])
                  for center in (18, 46)}
    if any(len(local_li15[center]) == 0 for center in (18, 46)):
        raise AssertionError('no anatomy-dominant Li15 targets')

    def capture(images, polarity):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(polarity == 'off')
        release = torch.empty((144, len(net.graded_nodes)))
        traces = {name: torch.empty((144, len(target_indices)))
                  for name in ('impulse', 'current', 'voltage')}
        traces['spikes'] = torch.zeros((144, len(target_indices)), dtype=torch.bool)
        source_spikes = 0
        pixel_events = 0
        step = 0
        for frame, image in enumerate(images):
            events = camera.observe(image)
            injection = retina.project(events)*metadata['config']['sensory_gain']
            if 3 <= frame < 15:
                pixel_events += len(events.pixels)
            for tick in range(8):
                activity = net.step(injection if tick == 0 else zero)
                release[step] = net.release_history[
                    (net.step_index-1) % net.history_length]
                traces['impulse'][step] = net.last_graded_impulse[0, selected_targets]
                traces['current'][step] = (
                    net.feedforward_current[0, selected_targets]
                    +net.predictive_current[0, selected_targets]
                    +net.behavioral_current[0, selected_targets])
                traces['voltage'][step] = net.voltage[0, selected_targets]
                traces['spikes'][step] = activity.spikes[0, selected_targets]
                if 3 <= frame < 15:
                    source_spikes += int(activity.spikes[0, net.graded_nodes].sum())
                step += 1
        if not (torch.isfinite(release).all()
                and all(torch.isfinite(traces[name]).all()
                        for name in ('impulse', 'current', 'voltage'))):
            raise AssertionError('non-finite graded propagation state')
        decay = math.exp(-net.config.dt/.250)
        initial_current = (settled['feedforward_current'][0, selected_targets]
                           +settled['predictive_current'][0, selected_targets]
                           +settled['behavioral_current'][0, selected_targets])
        positive_voltage, negative_voltage = causal_highpass(
            traces['voltage'], settled['voltage'][0, selected_targets],
            decay=decay)
        positive_current, negative_current = causal_highpass(
            traces['current'], initial_current, decay=decay)
        candidates = dict(positive_voltage=positive_voltage,
                          negative_voltage=negative_voltage,
                          positive_current=positive_current,
                          negative_current=negative_current)
        return dict(release=release, source_spikes=source_spikes,
                    pixel_events=pixel_events, candidates=candidates, **traces)

    blanks = {polarity: capture(
        [torch.full((1, 32, 64), polarity == 'off', dtype=torch.bool)]*18,
        polarity) for polarity in ('on', 'off')}
    repeated_blank = capture([torch.zeros((1, 32, 64), dtype=torch.bool)]*18,
                             'on')
    if any(not torch.equal(blanks['on'][name], repeated_blank[name])
           for name in ('release', 'impulse', 'current', 'voltage', 'spikes')):
        raise AssertionError('restored graded state changed a repeated blank')
    report['state_restoration_reproducible'] = True
    li15_report = dict(source_sha256=source_sha,
                       graph_sha256=graph.identity(), selected_t2_cap=selected_cap,
                       baseline_ms=250, li15_cells=len(target_groups['Li15']),
                       conditions={}, candidate_screens={})
    li15_local_report = dict(source_sha256=source_sha,
        graph_sha256=graph.identity(), selected_t2_cap=selected_cap,
        baseline_ms=250, local_groups={str(center): dict(
            cells=len(local_li15[center]),
            body_ids=np.asarray(graph.body_ids)[li15_groups[center]].astype(int).tolist(),
            t2_contact_mass=li15_mass[center].astype(int).tolist())
            for center in (18, 46)},
        conditions={}, candidate_screens={})
    for center in (18, 46):
        other = 46 if center == 18 else 18
        for polarity in ('on', 'off'):
            for kind in ('dot', 'bar', 'static'):
                images = (static_dot(center, polarity) if kind == 'static'
                          else frames_for_condition(center, 1, polarity,
                                                    kind=kind))
                observed = capture(images, polarity)
                source = summarize_feature(
                    observed['release'], blanks[polarity]['release'],
                    regions[center], regions[other])
                downstream = {cell_type: summarize_target(
                    observed, blanks[polarity], target_groups[cell_type])
                    for cell_type in TARGET_TYPES}
                key = f'{center}-{polarity}-{kind}'
                report['conditions'][key] = dict(
                    pixel_events=observed['pixel_events'],
                    source_spikes=observed['source_spikes'],
                    source=source, downstream=downstream)
                li15_report['conditions'][key] = {
                    feature: summarize_local_candidate(
                        observed['candidates'][feature],
                        blanks[polarity]['candidates'][feature],
                        target_groups['Li15'])
                    for feature in observed['candidates']}
                li15_local_report['conditions'][key] = {
                    feature: summarize_local_candidate(
                        observed['candidates'][feature],
                        blanks[polarity]['candidates'][feature],
                        local_li15[center])
                    for feature in observed['candidates']}
                print(json.dumps(dict(condition=key,
                    source_fraction=source['local_fraction_above_blank_p99'],
                    Li15=downstream['Li15']['voltage']['per_cell_absolute_p90'])),
                    flush=True)
    report['partial_t2_li15_screen'] = bool(all(
        (row := report['conditions'][f'{center}-{polarity}-dot'])['source']
            ['local_fraction_above_blank_p99'] >= .05
        and row['downstream']['Li15']['impulse']['mean_absolute_change'] >= .002
        and row['downstream']['Li15']['voltage']['per_cell_absolute_p90'] >= .002
        for center in (18, 46) for polarity in ('on', 'off')))
    for feature in ('positive_voltage', 'negative_voltage',
                    'positive_current', 'negative_current'):
        rows = [li15_report['conditions'][f'{center}-{polarity}-dot'][feature]
                for center in (18, 46) for polarity in ('on', 'off')]
        li15_report['candidate_screens'][feature] = bool(all(
            row['finite'] and row['fraction_above_blank_p99'] >= .03
            and row['blank_fraction_above_blank_p99'] <= .02
            and row['mean_absolute_change'] >= .0005 for row in rows))
        local_rows = [li15_local_report['conditions']
                      [f'{center}-{polarity}-dot'][feature]
                      for center in (18, 46) for polarity in ('on', 'off')]
        li15_local_report['candidate_screens'][feature] = bool(all(
            row['finite'] and row['fraction_above_blank_p99'] >= .03
            and row['blank_fraction_above_blank_p99'] <= .02
            and row['mean_absolute_change'] >= .0005 for row in local_rows))
    if (not torch.equal(net.magnitudes, original_weights)
            or checksum(SOURCE) != source_sha):
        raise AssertionError('frozen checkpoint or synaptic weights changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    LI15_OUT.write_text(json.dumps(li15_report, indent=2, allow_nan=False)+'\n')
    LI15_LOCAL_OUT.write_text(json.dumps(li15_local_report, indent=2,
                                         allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), selected_cap=selected_cap,
                          partial_t2_li15_screen=report['partial_t2_li15_screen'],
                          li15_candidates=[name for name, passed in
                              li15_report['candidate_screens'].items() if passed],
                          local_li15_candidates=[name for name, passed in
                              li15_local_report['candidate_screens'].items()
                              if passed])),
          flush=True)


if __name__ == '__main__':
    main()

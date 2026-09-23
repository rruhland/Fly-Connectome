"""Frozen T5 motion check for the bounded opt-in Tm4 current supplement."""

import argparse
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
from motion_t5_axis_aligned import moving_bar, static_bar
from t5_local_order_current import LAG, ORDER_STATE
from tm4_supplemented_t5 import TM4_STATE, Tm4SupplementedT5Network


OUT = Path('docs/experiments/2026-09-23-tm4-graded-results.json')
FAST_OUT = Path('docs/experiments/2026-09-23-tm4-fast-release-results.json')
FORECAST_OUT = Path('docs/experiments/2026-09-23-t5-future-afferent-results.json')
PAIRED_OUT = Path('docs/experiments/2026-09-23-t5-paired-afferent-results.json')
SPIKE_OUT = Path('docs/experiments/2026-09-23-t5-future-spike-results.json')
STATE = NETWORK_STATE+GRADED_STATE+ORDER_STATE+TM4_STATE
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
FAST_CASES = CASES+((14, 1), (18, 2))
SUBTYPES = ('T5c', 'T5d')
CAPS = (.01, .03, .10)
ISSUE = torch.arange(3, 14)*8


def roc_auc(scores, events):
    scores, events = np.asarray(scores).ravel(), np.asarray(events).ravel()
    positives = int(events.sum())
    negatives = len(events)-positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind='mergesort')
    scores, events = scores[order], events[order]
    wins = 0.
    lower_negatives = 0
    start = 0
    while start < len(scores):
        end = np.searchsorted(scores, scores[start], side='right')
        tied_positive = int(events[start:end].sum())
        tied_negative = end-start-tied_positive
        wins += tied_positive*(lower_negatives+.5*tied_negative)
        lower_negatives += tied_negative
        start = end
    return wins/(positives*negatives)


def future_events(gate):
    return torch.stack([gate[t+8:t+16].max(0).values for t in ISSUE]) > .05


def future_deviation_events(stimulus, blank, *, threshold):
    deviation = stimulus-blank
    return torch.stack([deviation[t+8:t+16].max(0).values
                        for t in ISSUE]) > threshold


def arm_support(now, delayed, events):
    if not events.any():
        return None
    return float(((now[ISSUE].abs()+delayed[ISSUE].abs()) > .001)[events]
                 .float().mean())


@torch.no_grad()
def main(*, fast_release=False, forecast_audit=False, paired_audit=False,
         spike_audit=False):
    torch.set_num_threads(4)
    forecast_audit |= paired_audit or spike_audit
    fast_release |= forecast_audit
    tau = .050 if fast_release else .250
    output = (SPIKE_OUT if spike_audit else
              PAIRED_OUT if paired_audit else
              FORECAST_OUT if forecast_audit else
              FAST_OUT if fast_release else OUT)
    cases = FAST_CASES if fast_release else CASES
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = Tm4SupplementedT5Network(graph, metadata['delays'],
        metadata['pathways'], config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=.10, voltage_scale=.02, source_type='Tm9',
        source_state='current', gate_gain=8000., gate_cap=1.,
        tm4_release_cap=CAPS[0], tm4_current_scale=.02,
        tm4_release_tau=tau)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight']))
    weights = net.magnitudes.clone()
    for cell_type in ('Tm4', 'Tm9'):
        net.rest_current[torch.tensor(types == cell_type)] = .85
    if not (net.pathways[net.tm4_supplement_edges] == 0).all():
        raise AssertionError('Tm4 supplement left measured feedforward path')
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    net.enable_graded()
    for _ in range(256):
        net.step(zero)
    prestate = {name: getattr(net, name).clone() for name in STATE}
    pre_tick = net.step_index
    pre_tm4_count = net.tm4_blank_count
    pre_order_count = net.order_blank_count

    def restore(values, tick):
        for name, value in values.items():
            getattr(net, name).copy_(value)
        net.step_index = tick

    t5_cd = torch.tensor(np.isin(types, SUBTYPES))
    calibration = []
    for cap in CAPS:
        restore(prestate, pre_tick)
        net.tm4_blank_count = pre_tm4_count
        net.order_blank_count = pre_order_count
        net.tm4_enabled = False
        net.order_enabled = False
        net.tm4_release_cap = cap
        net.enable_tm4()
        impulses = []
        blank_spikes = 0
        for tick in range(256):
            activity = net.step(zero)
            if tick >= 128:
                impulses.append(net.last_tm4_impulse[0, t5_cd].abs().clone())
                blank_spikes += int(activity.spikes[0, t5_cd].sum())
        row = dict(cap=cap,
            blank_p99_added_t5_impulse=float(torch.quantile(
                torch.stack(impulses), .99)),
            blank_t5_spike_rate=blank_spikes/(128*int(t5_cd.sum())),
            finite=bool(torch.isfinite(net.voltage).all()
                        and torch.isfinite(net.last_tm4_impulse).all()))
        calibration.append(row)
        print(json.dumps(dict(calibration=row)), flush=True)
    passing = [row['cap'] for row in calibration if row['finite']
        and row['blank_p99_added_t5_impulse'] <= .05
        and row['blank_t5_spike_rate'] < .01]
    selected_cap = max(passing) if passing else None
    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        annotations_sha256=checksum(ANNOTATIONS),
        tm4_source_cells=len(net.tm4_nodes),
        tm4_t5_edges=len(net.tm4_supplement_edges),
        tm4_t5_contacts=int(graph.contacts[net.tm4_supplement_edges.numpy()].sum()),
        tm4_edge_signs={str(sign): int((net.signs[net.tm4_supplement_edges]
                        == sign).sum()) for sign in (-1, 1)},
        tm4_delay_ticks=dict(min=int(net.tm4_delays.min()),
                             max=int(net.tm4_delays.max())),
        current_scale=.02, tm4_release_tau_ms=int(tau*1000),
        tm9_release_cap=.10,
        tm4_rest_current=.85, tm9_rest_current=.85,
        ticks_per_frame=8, calibration=calibration,
        selected_cap=selected_cap, conditions={},
        passes_frozen_feature_gate=False)
    if selected_cap is None:
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output), selected_cap=None)), flush=True)
        return

    settled = {}
    for mode in ('baseline', 'supplement'):
        restore(prestate, pre_tick)
        net.tm4_blank_count = pre_tm4_count
        net.order_blank_count = pre_order_count
        net.tm4_enabled = False
        net.order_enabled = False
        net.tm4_release_cap = selected_cap
        if mode == 'supplement':
            net.enable_tm4()
        for _ in range(256):
            net.step(zero)
        net.enable_order()
        for _ in range(128):
            net.step(zero)
        settled[mode] = (net.step_index,
                         {name: getattr(net, name).clone() for name in STATE})

    annotations = feather.read_table(ANNOTATIONS,
        columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def capture(mode, images, nodes):
        tick, state = settled[mode]
        restore(state, tick)
        net.tm4_enabled = mode == 'supplement'
        net.order_enabled = True
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(True)
        joined = torch.cat(tuple(nodes.values()))
        index = net.t5_lookup[joined]
        traces = {name: torch.zeros((144, len(joined))) for name in
                  ('gate', 'tm4_now', 'tm4_delayed', 'spikes', 'tm4_impulse')}
        if forecast_audit:
            traces['afferent'] = torch.zeros_like(traces['gate'])
            traces['pending_order'] = torch.zeros_like(traces['gate'])
        pixel_events = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for step in range(8):
                t = frame_index*8+step
                activity = net.step(sensory if step == 0 else zero,
                                    capture_increments=forecast_audit)
                absolute_tick = net.step_index-1
                now = net.residual_history[absolute_tick % (LAG+1)]
                delayed = net.residual_history[(absolute_tick-LAG) % (LAG+1)]
                traces['gate'][t] = net.last_gate_current[index]
                traces['tm4_now'][t] = now[0, index]
                traces['tm4_delayed'][t] = delayed[0, index]
                traces['spikes'][t] = activity.spikes[0, joined].float()
                traces['tm4_impulse'][t] = net.last_tm4_impulse[0, joined]
                if forecast_audit:
                    traces['afferent'][t] = activity.feedforward_arrivals[0, joined]
                    traces['pending_order'][t] = net.pending_gate[index]
        if not (torch.isfinite(net.voltage).all()
                and torch.isfinite(net.last_tm4_impulse).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        return pixel_events, traces

    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    checks = []
    for center, speed in cases:
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        nodes = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins))) for name in SUBTYPES}
        if any(len(group) == 0 for group in nodes.values()):
            raise AssertionError('missing anatomy-local T5 targets')
        spans = {}
        offset = 0
        for name in SUBTYPES:
            spans[name] = slice(offset, offset+len(nodes[name]))
            offset += len(nodes[name])
        images = dict(blank=bright,
            up=moving_bar('vertical', center, -1, speed),
            down=moving_bar('vertical', center, 1, speed),
            static=static_bar('vertical', center))
        captures = {mode: {condition: capture(mode, frames, nodes)
                    for condition, frames in images.items()}
                    for mode in ('baseline', 'supplement')}
        for mode in captures:
            if captures[mode]['up'][0] != captures[mode]['down'][0]:
                raise AssertionError('opposite motion has unequal event counts')
            if captures[mode]['blank'][0] != 0:
                raise AssertionError('bright blank generated camera events')
        condition = report['conditions'][f'{center}-s{speed}'] = {}
        if forecast_audit:
            audit = report.setdefault('forecast_audit', {})[f'{center}-s{speed}'] = {}
        if paired_audit:
            paired = report.setdefault('paired_forecast_audit', {})[
                f'{center}-s{speed}'] = {}
        if spike_audit:
            spike = report.setdefault('spike_forecast_audit', {})[
                f'{center}-s{speed}'] = {}
        for name in SUBTYPES:
            local = spans[name]
            baseline_event = {direction: future_events(
                captures['baseline'][direction][1]['gate'][:, local])
                for direction in ('up', 'down')}
            rows = condition[name] = {}
            if forecast_audit:
                audit[name] = {}
            if paired_audit:
                paired[name] = {}
            if spike_audit:
                spike[name] = {}
            for mode in ('baseline', 'supplement'):
                if forecast_audit:
                    audit[name][mode] = {}
                    for direction in ('up', 'down', 'static', 'blank'):
                        trace = captures[mode][direction][1]
                        event = torch.stack([trace['afferent'][t+8:t+16,
                            local].max(0).values for t in ISSUE]) > .01
                        order = trace['pending_order'][ISSUE, local]
                        audit[name][mode][direction] = dict(
                            future_afferent_events=int(event.sum()),
                            samples=event.numel(),
                            event_fraction=float(event.float().mean()),
                            order_event_mean=float(order[event].mean())
                                if event.any() else None,
                            order_quiet_mean=float(order[~event].mean())
                                if (~event).any() else None,
                            order_auc=roc_auc(order.numpy(), event.numpy()),
                            order_active_fraction=float((order > .05)
                                                        .float().mean()))
                if paired_audit:
                    paired[name][mode] = {}
                    blank_trace = captures[mode]['blank'][1]['afferent'][:, local]
                    for direction in ('up', 'down', 'static'):
                        trace = captures[mode][direction][1]
                        event = future_deviation_events(
                            trace['afferent'][:, local], blank_trace,
                            threshold=.01)
                        order = trace['pending_order'][ISSUE, local]
                        paired[name][mode][direction] = dict(
                            future_deviation_events=int(event.sum()),
                            samples=event.numel(),
                            event_fraction=float(event.float().mean()),
                            order_event_mean=float(order[event].mean())
                                if event.any() else None,
                            order_quiet_mean=float(order[~event].mean())
                                if (~event).any() else None,
                            order_auc=roc_auc(order.numpy(), event.numpy()))
                if spike_audit:
                    spike[name][mode] = {}
                    for direction in ('up', 'down', 'static', 'blank'):
                        trace = captures[mode][direction][1]
                        events = future_events(trace['spikes'][:, local])
                        order = trace['pending_order'][ISSUE, local]
                        recent = torch.stack([trace['spikes'][t-7:t+1,
                            local].sum(0) for t in ISSUE])
                        spike[name][mode][direction] = dict(
                            future_spike_events=int(events.sum()),
                            samples=events.numel(),
                            event_fraction=float(events.float().mean()),
                            order_event_mean=float(order[events].mean())
                                if events.any() else None,
                            order_quiet_mean=float(order[~events].mean())
                                if (~events).any() else None,
                            order_auc=roc_auc(order.numpy(), events.numpy()),
                            recent_spike_auc=roc_auc(recent.numpy(),
                                                     events.numpy()))
                count = {direction: captures[mode][direction][1]['spikes'][
                    24:120, local].sum(0) for direction in ('up', 'down', 'static')}
                contrast = count['down']-count['up']
                preferred = max(float(count['up'].sum()),
                                float(count['down'].sum()))
                static = float(count['static'].sum())
                blank_rate = float(captures[mode]['blank'][1]['spikes'][
                    16:, local].mean())
                support = {}
                for direction in ('up', 'down'):
                    trace = captures[mode][direction][1]
                    support[direction] = dict(
                        baseline_future_gate_events=int(
                            baseline_event[direction].sum()),
                        tm4_support=arm_support(trace['tm4_now'][:, local],
                            trace['tm4_delayed'][:, local],
                            baseline_event[direction]),
                        graded_impulse_mean=float(trace['tm4_impulse'][
                            24:120, local].abs().mean()))
                rows[mode] = dict(cells=len(nodes[name]),
                    contrast_down_minus_up=int(contrast.sum()),
                    positive_cells=int((contrast > 0).sum()),
                    negative_cells=int((contrast < 0).sum()),
                    preferred_moving_spikes=int(preferred),
                    static_spikes=int(static),
                    moving_over_static=preferred/max(static, 1),
                    blank_spike_rate=blank_rate,
                    event_support=support)
            base, trial = rows['baseline'], rows['supplement']
            sign = -1 if name == 'T5c' else 1
            contrast = trial['contrast_down_minus_up']
            majority = (trial['negative_cells'] > trial['positive_cells']
                        if sign < 0 else
                        trial['positive_cells'] > trial['negative_cells'])
            checks.append(contrast*sign >= 10 and majority
                and trial['moving_over_static'] >= 1.5
                and trial['moving_over_static']
                    >= .9*base['moving_over_static']
                and trial['blank_spike_rate'] < .001)
        print(json.dumps(dict(condition=f'{center}-s{speed}',
            T5c={mode: condition['T5c'][mode]['contrast_down_minus_up']
                 for mode in ('baseline', 'supplement')},
            T5d={mode: condition['T5d'][mode]['contrast_down_minus_up']
                 for mode in ('baseline', 'supplement')})), flush=True)
    target = report['conditions']['16-s2']['T5d']
    base_support = target['baseline']['event_support']['up']['tm4_support']
    trial_support = target['supplement']['event_support']['up']['tm4_support']
    report['fast_tm4_support'] = dict(baseline=base_support,
        supplement=trial_support, improvement=trial_support-base_support,
        passes=bool(trial_support >= .55 and trial_support-base_support >= .08))
    report['passes_frozen_feature_gate'] = bool(
        report['fast_tm4_support']['passes'] and all(checks))
    if forecast_audit:
        forecast_checks = []
        for center, speed in cases:
            rows = report['forecast_audit'][f'{center}-s{speed}']
            for name, preferred in (('T5c', 'up'), ('T5d', 'down')):
                entry = rows[name]['supplement']
                moving = entry[preferred]
                static = entry['static']
                blank = entry['blank']
                forecast_checks.append(
                    moving['future_afferent_events'] >= 10
                    and moving['order_event_mean'] is not None
                    and moving['order_quiet_mean'] is not None
                    and moving['order_event_mean']
                        >= 2*moving['order_quiet_mean']
                    and moving['order_auc'] is not None
                    and moving['order_auc'] >= .70
                    and static['order_active_fraction']
                        <= .5*moving['order_active_fraction']
                    and blank['event_fraction'] < .001)
        report['passes_future_afferent_preflight'] = bool(all(forecast_checks))
    if paired_audit:
        paired_checks = []
        for center, speed in cases:
            rows = report['paired_forecast_audit'][f'{center}-s{speed}']
            for name, preferred in (('T5c', 'up'), ('T5d', 'down')):
                entry = rows[name]['supplement']
                moving, static = entry[preferred], entry['static']
                paired_checks.append(
                    moving['future_deviation_events'] >= 10
                    and moving['order_event_mean'] is not None
                    and moving['order_quiet_mean'] is not None
                    and moving['order_event_mean']
                        >= 2*moving['order_quiet_mean']
                    and moving['order_auc'] is not None
                    and moving['order_auc'] >= .70
                    and static['event_fraction']
                        <= .5*moving['event_fraction'])
        report['passes_paired_future_afferent_preflight'] = bool(
            all(paired_checks))
    if spike_audit:
        spike_checks = []
        for center, speed in cases:
            rows = report['spike_forecast_audit'][f'{center}-s{speed}']
            for name, preferred in (('T5c', 'up'), ('T5d', 'down')):
                entry = rows[name]['supplement']
                moving, static, blank = (entry[preferred], entry['static'],
                                         entry['blank'])
                spike_checks.append(
                    moving['future_spike_events'] >= 10
                    and moving['order_event_mean'] is not None
                    and moving['order_quiet_mean'] is not None
                    and moving['order_event_mean']
                        >= 2*moving['order_quiet_mean']
                    and moving['order_auc'] is not None
                    and moving['recent_spike_auc'] is not None
                    and moving['order_auc'] >= .70
                    and moving['order_auc']
                        >= moving['recent_spike_auc']+.05
                    and static['event_fraction']
                        < .5*moving['event_fraction']
                    and blank['event_fraction'] < .001)
        report['passes_future_spike_preflight'] = bool(all(spike_checks))
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), selected_cap=selected_cap,
        fast_tm4_support=report['fast_tm4_support'],
        passes_frozen_feature_gate=report['passes_frozen_feature_gate'],
        passes_future_afferent_preflight=report.get(
            'passes_future_afferent_preflight'),
        passes_paired_future_afferent_preflight=report.get(
            'passes_paired_future_afferent_preflight'),
        passes_future_spike_preflight=report.get(
            'passes_future_spike_preflight'))),
        flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast-release', action='store_true')
    parser.add_argument('--forecast-audit', action='store_true')
    parser.add_argument('--paired-audit', action='store_true')
    parser.add_argument('--spike-audit', action='store_true')
    args = parser.parse_args()
    main(fast_release=args.fast_release, forecast_audit=args.forecast_audit,
         paired_audit=args.paired_audit, spike_audit=args.spike_audit)

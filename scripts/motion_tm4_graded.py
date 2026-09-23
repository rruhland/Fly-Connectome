"""Frozen T5 motion check for the bounded opt-in Tm4 current supplement."""

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
STATE = NETWORK_STATE+GRADED_STATE+ORDER_STATE+TM4_STATE
CASES = ((10, 1), (22, 1), (16, 1), (16, 2))
SUBTYPES = ('T5c', 'T5d')
CAPS = (.01, .03, .10)
ISSUE = torch.arange(3, 14)*8


def future_events(gate):
    return torch.stack([gate[t+8:t+16].max(0).values for t in ISSUE]) > .05


def arm_support(now, delayed, events):
    if not events.any():
        return None
    return float(((now[ISSUE].abs()+delayed[ISSUE].abs()) > .001)[events]
                 .float().mean())


@torch.no_grad()
def main():
    torch.set_num_threads(4)
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
        tm4_release_cap=CAPS[0], tm4_current_scale=.02)
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
        current_scale=.02, tm9_release_cap=.10,
        tm4_rest_current=.85, tm9_rest_current=.85,
        ticks_per_frame=8, calibration=calibration,
        selected_cap=selected_cap, conditions={},
        passes_frozen_feature_gate=False)
    if selected_cap is None:
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(OUT), selected_cap=None)), flush=True)
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
        pixel_events = 0
        for frame_index, frame in enumerate(images):
            events = camera.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for step in range(8):
                t = frame_index*8+step
                activity = net.step(sensory if step == 0 else zero)
                absolute_tick = net.step_index-1
                now = net.residual_history[absolute_tick % (LAG+1)]
                delayed = net.residual_history[(absolute_tick-LAG) % (LAG+1)]
                traces['gate'][t] = net.last_gate_current[index]
                traces['tm4_now'][t] = now[0, index]
                traces['tm4_delayed'][t] = delayed[0, index]
                traces['spikes'][t] = activity.spikes[0, joined].float()
                traces['tm4_impulse'][t] = net.last_tm4_impulse[0, joined]
        if not (torch.isfinite(net.voltage).all()
                and torch.isfinite(net.last_tm4_impulse).all()
                and torch.equal(net.magnitudes, weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        return pixel_events, traces

    bright = [torch.ones((1, 32, 64), dtype=torch.bool)]*18
    checks = []
    for center, speed in CASES:
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
        for name in SUBTYPES:
            local = spans[name]
            baseline_event = {direction: future_events(
                captures['baseline'][direction][1]['gate'][:, local])
                for direction in ('up', 'down')}
            rows = condition[name] = {}
            for mode in ('baseline', 'supplement'):
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
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), selected_cap=selected_cap,
        fast_tm4_support=report['fast_tm4_support'],
        passes_frozen_feature_gate=report['passes_frozen_feature_gate'])),
        flush=True)


if __name__ == '__main__':
    main()

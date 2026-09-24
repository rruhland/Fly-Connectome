"""Opt-in frozen motion-front-end comparison on matched event streams."""

import argparse
import json
import time
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
from t5_local_order_current import ORDER_STATE
from tm4_supplemented_t5 import TM4_STATE, Tm4SupplementedT5Network


OUT = Path('docs/experiments/2026-09-23-m1a-motion-front-end-results.json')
EXTENDED_OUT = Path('docs/experiments/2026-09-23-m1a-motion-front-end-extension-results.json')
DOT_OUT = Path('docs/experiments/2026-09-23-m1a-square-dot-results.json')
STATE = NETWORK_STATE + GRADED_STATE + ORDER_STATE + TM4_STATE
SUBTYPES = tuple(f'T{stage}{direction}' for stage in (4, 5)
                 for direction in 'abcd')


def event_centroid(events, occupancy, *, on, width):
    """Reconstruct a uniform-background object's centroid from events only."""
    occupancy[events.pixels] = events.on
    pixels = (occupancy == on).nonzero().flatten()
    if not len(pixels):
        return None
    return (float((pixels // width).float().mean()),
            float((pixels % width).float().mean()))


def median_displacement(positions, *, axis=0):
    differences = [positions[t+1][axis] - positions[t][axis]
                   for t in range(3, 14)
                   if positions[t] is not None and positions[t+1] is not None]
    return float(np.median(differences)) if differences else None


def opponent_score(spikes, cells):
    return spikes['T5d'] / max(cells['T5d'], 1) - spikes['T5c'] / max(cells['T5c'], 1)


def moving_square(axis, center, direction, speed):
    if axis not in ('vertical', 'horizontal') or direction not in (-1, 1):
        raise ValueError('axis and direction required')
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    frames = []
    for frame in range(18):
        if 2 <= frame < 15:
            position = center+direction*speed*(frame-8)
            square = (((y-position).abs() <= 1) & ((x-32).abs() <= 1)
                      if axis == 'vertical' else
                      ((x-position).abs() <= 1) & ((y-16).abs() <= 1))
            frames.append((~square).unsqueeze(0))
        else:
            frames.append(torch.ones((1, 32, 64), dtype=torch.bool))
    return frames


def static_square(axis, center):
    frames = moving_square(axis, center, 1, 1)
    for frame in range(2, 15):
        frames[frame] = frames[8]
    return frames


def images(axis, center, direction, speed, on, *, dot=False):
    frames = (moving_square(axis, center, direction, speed) if dot else
              moving_bar(axis, center, direction, speed))
    return [~frame for frame in frames] if on else frames


@torch.no_grad()
def main(*, horizontal_extension=False, dot_contest=False):
    torch.set_num_threads(4)
    horizontal_extension |= dot_contest
    started = time.perf_counter()
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    net = Tm4SupplementedT5Network(
        graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=.10, voltage_scale=.02, source_type='Tm9',
        source_state='current', gate_gain=8000., gate_cap=1.,
        tm4_release_cap=.10, tm4_current_scale=.02, tm4_release_tau=.050)
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
    net.enable_tm4()
    for _ in range(256):
        net.step(zero)
    net.enable_order()
    for _ in range(128):
        net.step(zero)
    settled = {name: getattr(net, name).clone() for name in STATE}
    settled_tick = net.step_index

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')

    def local_nodes(axis, center):
        if axis == 'vertical':
            field = (y >= center-8) & (y <= center+8) & (x >= 24) & (x <= 40)
        else:
            field = (x >= center-8) & (x <= center+8) & (y >= 9) & (y <= 23)
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        groups = {name: torch.tensor(np.flatnonzero(
            (types == name) & np.isin(columns, bins))) for name in SUBTYPES}
        if any(len(groups[name]) == 0 for name in SUBTYPES):
            raise AssertionError('missing anatomy-local T4/T5 subtype')
        return groups

    def capture(frames, *, axis, center, on):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.step_index = settled_tick
        camera = EventCamera(1, 32, 64)
        camera.previous.fill_(not on)
        occupancy = torch.full((32*64,), not on, dtype=torch.bool)
        groups = local_nodes(axis, center)
        joined = torch.cat(tuple(groups.values()))
        counts = torch.zeros(len(joined), dtype=torch.int32)
        positions = []
        event_count = 0
        blank_spikes = torch.zeros((), dtype=torch.int32)
        for frame_index, frame in enumerate(frames):
            events = camera.observe(frame)
            positions.append(event_centroid(events, occupancy, on=on, width=64))
            if 3 <= frame_index < 15:
                event_count += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(sensory if tick == 0 else zero)
                if 3 <= frame_index < 15:
                    counts += activity.spikes[0, joined].int()
                if frame_index >= 2:
                    blank_spikes += activity.spikes[0, net.t5_nodes].sum().int()
        if not (torch.isfinite(net.voltage).all()
                and torch.equal(net.magnitudes, original_weights)):
            raise AssertionError('nonfinite state or changed long-term weights')
        by_type = {}
        offset = 0
        for name, nodes in groups.items():
            values = counts[offset:offset+len(nodes)]
            by_type[name] = values.numpy()
            offset += len(nodes)
        return dict(events=event_count,
                    displacement=median_displacement(
                        positions, axis=0 if axis == 'vertical' else 1),
                    spikes=by_type,
                    cells={name: len(nodes) for name, nodes in groups.items()},
                    t5_spike_rate=float(blank_spikes/(16*8*len(net.t5_nodes))))

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
                  annotations_sha256=checksum(ANNOTATIONS),
                  ticks_per_frame=8, frozen_weights=True,
                  shape='square' if dot_contest else 'bar',
                  calibration=[], conditions=[], blanks={})

    def pair(axis, center, speed, on, *, calibration=False):
        positive = capture(images(axis, center, 1, speed, on, dot=dot_contest),
                           axis=axis, center=center, on=on)
        negative = capture(images(axis, center, -1, speed, on, dot=dot_contest),
                           axis=axis, center=center, on=on)
        stationary = (static_square(axis, center) if dot_contest else
                      static_bar(axis, center))
        if on:
            stationary = [~frame for frame in stationary]
        static = capture(stationary, axis=axis, center=center, on=on)
        if positive['events'] != negative['events']:
            raise AssertionError('opposite directions have unequal camera events')
        subtypes = {}
        for name in SUBTYPES:
            pos, neg, still = (item['spikes'][name] for item in
                               (positive, negative, static))
            difference = pos-neg
            subtypes[name] = dict(cells=positive['cells'][name],
                positive_spikes=int(pos.sum()), negative_spikes=int(neg.sum()),
                static_spikes=int(still.sum()),
                positive_minus_negative=int(difference.sum()),
                positive_cells=int((difference > 0).sum()),
                negative_cells=int((difference < 0).sum()))
        row = dict(axis=axis, center=center, speed=speed, polarity='on' if on else 'off',
            calibration=calibration, pixel_events=positive['events'],
            event_tracker=dict(positive_displacement=positive['displacement'],
                negative_displacement=negative['displacement'],
                static_displacement=static['displacement']),
            t5_opponent=dict(positive=opponent_score(
                {name: entry['positive_spikes'] for name, entry in subtypes.items()},
                positive['cells']), negative=opponent_score(
                {name: entry['negative_spikes'] for name, entry in subtypes.items()},
                positive['cells'])), subtypes=subtypes)
        print(json.dumps(dict(axis=axis, center=center, speed=speed,
              polarity=row['polarity'], calibration=calibration,
              t5_opponent=row['t5_opponent'],
              event_tracker=row['event_tracker'])), flush=True)
        return row

    for center in (10, 22):
        report['calibration'].append(pair('vertical', center, 1, False,
                                          calibration=True))
    if dot_contest:
        for center in (18, 46):
            report['calibration'].append(pair('horizontal', center, 1, False,
                                              calibration=True))
    vertical_calibration = [row for row in report['calibration']
                            if row['axis'] == 'vertical']
    up_scores = [row['t5_opponent']['negative'] for row in vertical_calibration]
    down_scores = [row['t5_opponent']['positive'] for row in vertical_calibration]
    threshold = (max(up_scores)+min(down_scores))/2
    report['t5_decoder'] = dict(threshold=threshold,
        calibration_separated=bool(max(up_scores) < min(down_scores)))

    cases = ([('vertical', center, speed, False)
              for center in (14, 18) for speed in (1, 2)]
             + [('horizontal', 32, speed, False) for speed in (1, 2)]
             + [('vertical', 16, 1, True), ('horizontal', 32, 1, True)])
    if horizontal_extension and not dot_contest:
        cases += [('horizontal', center, 1, False) for center in (18, 46)]
    for axis, center, speed, on in cases:
        row = pair(axis, center, speed, on)
        if axis == 'vertical' and not on:
            row['t5_opponent']['positive_correct'] = bool(
                row['t5_opponent']['positive'] > threshold)
            row['t5_opponent']['negative_correct'] = bool(
                row['t5_opponent']['negative'] < threshold)
        report['conditions'].append(row)

    if horizontal_extension and not dot_contest:
        center = next(row for row in report['conditions']
                      if row['axis'] == 'horizontal' and row['center'] == 32
                      and row['speed'] == 1 and row['polarity'] == 'off')
        right = center['t5_opponent']['positive']
        left = center['t5_opponent']['negative']
        threshold = (right+left)/2
        report['horizontal_extension'] = dict(
            center_threshold=threshold, center_separated=bool(right > left))
        for row in report['conditions']:
            if row['axis'] == 'horizontal' and row['center'] in (18, 46):
                row['t5_opponent']['positive_correct'] = bool(
                    row['t5_opponent']['positive'] > threshold)
                row['t5_opponent']['negative_correct'] = bool(
                    row['t5_opponent']['negative'] < threshold)

    for on in (False, True):
        background = torch.full((1, 32, 64), not on, dtype=torch.bool)
        blank = capture([background]*18, axis='vertical', center=16, on=on)
        report['blanks']['on' if on else 'off'] = dict(
            pixel_events=blank['events'], t5_spike_rate=blank['t5_spike_rate'],
            subtype_spikes={name: int(values.sum())
                            for name, values in blank['spikes'].items()})
    report['elapsed_seconds'] = time.perf_counter()-started
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output = DOT_OUT if dot_contest else EXTENDED_OUT if horizontal_extension else OUT
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), elapsed_seconds=report['elapsed_seconds'],
        t5_decoder=report['t5_decoder'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizontal-extension', action='store_true')
    parser.add_argument('--dot-contest', action='store_true')
    args = parser.parse_args()
    main(horizontal_extension=args.horizontal_extension,
         dot_contest=args.dot_contest)

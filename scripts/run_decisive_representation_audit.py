"""One-shot frozen motion-code transfer audit with paired T4/T5 responses."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn.functional as F

from event_centric_forecast import MotionWithSeparateSurface
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.sensor import EventCamera, Retina
from gap_timing_transfer import MODEL_OUT
from generic_motion_probe import events_map
from generic_native_cadence import scene_events
from m1a_motion_front_end import STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from run_event_centric_competition import train_state
from run_observation_horizon_forecast import sensory_sequence
from tm4_supplemented_t5 import Tm4SupplementedT5Network


OUT = Path('docs/experiments/2026-09-25-decisive-representation-audit-results.json')
DIRS = ('up', 'down', 'left', 'right')
SUBTYPES = tuple(f'T{stage}{direction}' for stage in (4, 5)
                 for direction in 'abcd')
ACTIVE = range(4, 13)
SHAPES = {'square': ((y, x) for y in range(-1, 2)
                     for x in range(-1, 2)),
          'plus': ((0, 0), (-2, 0), (-1, 0), (1, 0), (2, 0),
                   (0, -2), (0, -1), (0, 1), (0, 2))}
SHAPES['square'] = tuple(SHAPES['square'])
VELOCITY = dict(up=(-1, 0), down=(1, 0),
                left=(0, -1), right=(0, 1))


def obj(shape, center, direction, speed):
    return dict(shape=shape, center=center,
                direction=direction, speed=speed)


def make_cases():
    cases = []
    for direction in DIRS:
        for polarity in ('dark', 'bright'):
            for center in ((12, 24), (20, 40)):
                cases.append(dict(split='calibration', polarity=polarity,
                                  objects=[obj('square', center, direction, 1)]))
                cases.append(dict(split='speed', polarity=polarity,
                                  objects=[obj('square', center, direction, 2)]))
            for center in ((16, 32), (16, 48)):
                cases.append(dict(split='position', polarity=polarity,
                                  objects=[obj('square', center, direction, 1)]))
                cases.append(dict(split='shape', polarity=polarity,
                                  objects=[obj('plus', center, direction, 1)]))
    for first, second in (('up', 'down'), ('left', 'right'),
                          ('up', 'right'), ('down', 'left')):
        for polarity in ('dark', 'bright'):
            cases.append(dict(split='separated', polarity=polarity,
                              objects=[obj('square', (12, 20), first, 1),
                                       obj('square', (20, 44), second, 1)]))
    for polarity in ('dark', 'bright'):
        cases.append(dict(split='crossing', polarity=polarity,
                          objects=[obj('square', (16, 30), 'right', 1),
                                   obj('square', (16, 34), 'left', 1)]))
    return cases


def object_mask(item, frame):
    mask = torch.zeros((32, 64), dtype=torch.bool)
    if frame not in ACTIVE:
        return mask
    dy, dx = VELOCITY[item['direction']]
    cy, cx = item['center']
    cy += dy*item['speed']*(frame-8)
    cx += dx*item['speed']*(frame-8)
    for oy, ox in SHAPES[item['shape']]:
        y, x = cy+oy, cx+ox
        if 0 <= y < 32 and 0 <= x < 64:
            mask[y, x] = True
    return mask


def case_frames(case):
    background = case['polarity'] == 'dark'
    images = []
    paths = [torch.zeros((32, 64), dtype=torch.bool)
             for _ in case['objects']]
    for frame in range(18):
        combined = torch.zeros((32, 64), dtype=torch.bool)
        for index, item in enumerate(case['objects']):
            mask = object_mask(item, frame)
            combined |= mask
            paths[index] |= mask
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, combined] = not background
        images.append(image)
    regions = [F.max_pool2d(path.float()[None, None], 5,
                            stride=1, padding=2)[0, 0].bool()
               for path in paths]
    return images, regions


def normalized(value):
    vector = torch.as_tensor(value, dtype=torch.float32)
    return vector/vector.norm().clamp(min=1e-8)


def fit_centroids(rows, key):
    return {direction: normalized(torch.stack([
        normalized(row[key]) for row in rows
        if row['direction'] == direction]).mean(0))
        for direction in DIRS}


def predict_direction(centroids, feature):
    vector = normalized(feature)
    return min(DIRS, key=lambda direction: float(
        (vector-centroids[direction]).square().sum()))


def score(rows, centroids, key):
    predictions = [predict_direction(centroids, row[key]) for row in rows]
    correct = [guess == row['direction']
               for row, guess in zip(rows, predictions)]
    active = [bool(torch.as_tensor(row[key]).abs().sum()) for row in rows]
    return dict(correct=sum(correct), total=len(rows),
                accuracy=sum(correct)/max(len(rows), 1),
                active=sum(active),
                per_direction={direction: dict(correct=sum(ok for row, ok in
                    zip(rows, correct) if row['direction'] == direction),
                    total=sum(row['direction'] == direction for row in rows))
                    for direction in DIRS},
                per_polarity={polarity: dict(correct=sum(ok for row, ok in
                    zip(rows, correct) if row['polarity'] == polarity),
                    total=sum(row['polarity'] == polarity for row in rows))
                    for polarity in ('dark', 'bright')})


@torch.no_grad()
def make_graph():
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
    original = payload['state']['network']['magnitudes'].clamp(
        0, metadata['learning']['maximum_weight'])
    net.set_weights(original)
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
    table = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    columns, _ = infer_columns(metadata, retina, table)
    return dict(net=net, retina=retina, metadata=metadata, types=types,
                columns=columns, zero=zero, settled=settled,
                settled_tick=net.step_index, original=original,
                source_sha=source_sha, graph_sha=graph.identity(),
                annotations_sha=checksum(ANNOTATIONS))


@torch.no_grad()
def graph_features(graph, events, regions):
    net = graph['net']
    for name, value in graph['settled'].items():
        getattr(net, name).copy_(value)
    net.step_index = graph['settled_tick']
    types = graph['types']
    nodes = np.flatnonzero(np.isin(types, SUBTYPES))
    counts = torch.zeros(len(nodes), dtype=torch.int32)
    for frame, observed in enumerate(events):
        sensory = graph['retina'].project(observed)*graph['metadata'][
            'config']['sensory_gain']
        for tick in range(8):
            activity = net.step(sensory if tick == 0 else graph['zero'])
            if frame in ACTIVE:
                counts += activity.spikes[0, nodes].int()
    result = []
    local_types = types[nodes]
    local_columns = graph['columns'][nodes]
    for region in regions:
        bins = graph['retina'].pixel_bins[region.flatten()].unique().numpy()
        local = np.isin(local_columns, bins)
        result.append(torch.tensor([
            float(counts[torch.from_numpy(local & (local_types == name))].sum())
            /max(int((local & (local_types == name)).sum()), 1)
            for name in SUBTYPES]))
    return result


@torch.no_grad()
def latent_features(model, maps, regions):
    model.reset_state()
    observed = [torch.zeros(model.encoder.units) for _ in regions]
    hidden = [torch.zeros(model.encoder.units) for _ in regions]
    primitive = [torch.zeros(16) for _ in regions]
    for frame, (_, code, surface) in enumerate(sensory_sequence(maps)):
        model.step((code, surface))
        if frame not in ACTIVE:
            continue
        for index, region in enumerate(regions):
            observed[index] += model.observed[:, region].sum(1)
            hidden[index] += model.state[:, region].sum(1)
            primitive[index] += code[:, region].sum(1)
    return observed, hidden, primitive


@torch.no_grad()
def capture(model, graph, case):
    images, regions = case_frames(case)
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(case['polarity'] == 'dark')
    sparse_events = [camera.observe(image) for image in images]
    maps = [events_map(event) for event in sparse_events]
    observed, hidden, primitive = latent_features(model, maps, regions)
    measured = graph_features(graph, sparse_events, regions)
    return [dict(split=case['split'], polarity=case['polarity'],
                 direction=item['direction'], speed=item['speed'],
                 shape=item['shape'], observed=observed[index].tolist(),
                 hidden=hidden[index].tolist(),
                 primitive=primitive[index].tolist(),
                 t4t5=measured[index].tolist(),
                 hidden_t4t5=torch.cat((normalized(hidden[index]),
                                       normalized(measured[index]))).tolist())
            for index, item in enumerate(case['objects'])]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    model = train_state(saved['models']['learned_split'].code.motion,
                        [scene_events(seed) for seed in range(64)])
    trained_seconds = time.perf_counter()-started
    print(f'frozen local latent trained in {trained_seconds:.1f}s', flush=True)
    torch.set_num_threads(4)
    graph = make_graph()
    print(f'measured-graph T4/T5 initialized in '
          f'{time.perf_counter()-started:.1f}s', flush=True)
    cases = make_cases()
    rows = []
    for index, case in enumerate(cases):
        rows.extend(capture(model, graph, case))
        if (index+1) % 16 == 0:
            print(f'captured {index+1}/{len(cases)} scenes', flush=True)
    calibration = [row for row in rows if row['split'] == 'calibration']
    keys = ('primitive', 'observed', 'hidden', 't4t5', 'hidden_t4t5')
    centroids = {key: fit_centroids(calibration, key) for key in keys}
    splits = {name: {key: score([row for row in rows
                                 if row['split'] == name], centroids[key], key)
                     for key in keys}
              for name in ('calibration', 'position', 'speed', 'shape',
                           'separated', 'crossing')}
    control = []
    labels = [row['direction'] for row in calibration]
    for seed in range(16):
        shuffled = labels.copy()
        random.Random(1000+seed).shuffle(shuffled)
        fake = [{**row, 'direction': label}
                for row, label in zip(calibration, shuffled)]
        centroid = fit_centroids(fake, 'hidden')
        control.append({name: score([row for row in rows
                                     if row['split'] == name], centroid,
                                    'hidden')['accuracy']
                        for name in ('position', 'speed', 'separated')})
    if not torch.equal(graph['net'].magnitudes, graph['original']):
        raise AssertionError('measured-graph weights changed')
    if checksum(SOURCE) != graph['source_sha']:
        raise AssertionError('source checkpoint changed')
    result = dict(protocol='2026-09-25-decisive-representation-audit',
                  source_sha256=graph['source_sha'],
                  graph_sha256=graph['graph_sha'],
                  annotations_sha256=graph['annotations_sha'],
                  calibration_scenes=16, total_scenes=len(cases),
                  rows=rows, scores=splits,
                  shuffled_hidden_control={name: sum(row[name] for row in
                    control)/len(control)
                    for name in ('position', 'speed', 'separated')},
                  trained_seconds=trained_seconds,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {key: round(row['accuracy'], 3)
                             for key, row in group.items()}
                      for name, group in splits.items()}), flush=True)


if __name__ == '__main__':
    main()

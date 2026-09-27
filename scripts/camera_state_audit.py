"""Identity, continuous image noise and known sensor-outage state audit."""

import json
import time
from functools import partial
from pathlib import Path

import torch

from associative_patch_state import AssociativePatchState
from event_only_observer_teacher import fit_event_only_observer
from fly_connectome.sensor import EventCamera
from generic_motion_probe import local_motion_map
from generic_native_cadence import scene_events
from local_observation_model import LocalObservationModel
from run_associative_patch_state import train_memory
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence
from visual_history_benchmark import SHAPES


OUT = Path('docs/experiments/2026-09-26-camera-state-results.json')


def crossing_scene(axis, offset, background):
    if axis == 'horizontal':
        starts, directions = ((32+offset, 10), (32+offset, 54)), ((0, 1), (0, -1))
    elif axis == 'vertical':
        starts, directions = ((10, 32+offset), (54, 32+offset)), ((1, 0), (-1, 0))
    else:
        starts, directions = ((10, 10+offset), (54, 54+offset)), ((1, 1), (-1, -1))
    camera = EventCamera(1, 64, 64)
    camera.previous.fill_(background)
    images, events, truth = [], [], []
    for frame in range(42):
        image = torch.full((64, 64), background, dtype=torch.bool)
        centers = []
        for start, direction, shape in zip(starts, directions, ('square', 'plus')):
            y, x = (start[i]+frame*direction[i] for i in range(2))
            centers.append(torch.tensor([y, x], dtype=torch.float32))
            for dy, dx in SHAPES[shape]:
                if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                    image[y+dy, x+dx] = not background
        images.append(image.float())
        arrival = camera.observe(image[None])
        event = torch.zeros(2, 64*64)
        event[arrival.on.long(), arrival.pixels] = 1.
        events.append(event.reshape(2, 64, 64))
        truth.append(centers)
    return images, events, truth


@torch.no_grad()
def score_scene(base, memory, scene, condition, seed, *, state_type=AssociativePatchState):
    images, events, truth = scene
    noisy = corrupt_events(events, seed=seed)
    observer = LocalObservationModel(height=64, width=64)
    observer.weights.copy_(base.weights)
    observer.bias.copy_(base.bias)
    state = state_type(height=64, width=64, memory=memory)
    generator = torch.Generator().manual_seed(seed+1000)
    trace = torch.zeros(2, 64, 64)
    identities = []
    outage_hits = []
    for frame, (visible, event) in enumerate(zip(images, noisy)):
        available = not (condition == 'outage' and 16 <= frame < 21)
        if not available:
            event = torch.zeros_like(event)
        filtered, _ = observer.step(event)
        if available:
            if condition == 'image_noise':
                visible = (visible+.1*torch.randn(visible.shape, generator=generator)).clamp(0., 1.)
            observer.contrast.copy_(visible-visible.median())
        motion = local_motion_map(trace, filtered).clamp(-1., 1.)
        trace.mul_(.875).add_(filtered)
        contrast = observer.contrast
        sensory = torch.cat((filtered, observer.fast, observer.slow,
                              contrast.clamp(min=0.)[None],
                              (-contrast).clamp(min=0.)[None], motion, trace))
        field = state.step(sensory, observation_available=available)
        if frame == 6:
            claimed = set()
            for target in truth[frame]:
                candidates = [(float((e['position']-target).norm()), e['id'])
                              for e in state.entities if e['id'] not in claimed]
                chosen = min(candidates) if candidates else (float('inf'), None)
                identities.append(chosen[1] if chosen[0] <= 2 else None)
                claimed.add(chosen[1])
        if not available:
            outage_hits.extend(float(field[round(float(p[0])), round(float(p[1]))]) >= .5
                               for p in truth[frame])
    by_id = {e['id']: e for e in state.entities}
    errors, retained = [], []
    for identity, target in zip(identities, truth[-1]):
        entity = by_id.get(identity)
        error = float((entity['position']-target).norm()) if entity is not None else 64.
        errors.append(error)
        retained.append(entity is not None and entity['strength'] >= .5 and error <= 4.)
    extra = sum(e['strength'] >= .5 and
                min(float((e['position']-p).norm()) for p in truth[-1]) > 4.
                for e in state.entities)
    return dict(identity_retained=retained, center_errors=errors,
                extra_moving_hypotheses=extra, outage_hits=outage_hits)


@torch.no_grad()
def main(*, state_type=AssociativePatchState, out=OUT):
    torch.set_num_threads(1)
    start = time.perf_counter()
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    memory = train_memory(base, partial(anchored_sequence, period=1))
    results = dict(sensor='current visible frames plus events',
                   state=state_type.__name__, conditions={})
    for condition in ('clean', 'image_noise', 'outage'):
        rows = []
        for axis in ('horizontal', 'vertical', 'diagonal'):
            for offset in (-4, 0, 4):
                for background in (False, True):
                    row = score_scene(base, memory, crossing_scene(axis, offset, background),
                                      condition, 70000+len(rows), state_type=state_type)
                    rows.append(dict(axis=axis, offset=offset, background=background, **row))
        hits = [x for row in rows for x in row['identity_retained']]
        errors = [x for row in rows for x in row['center_errors']]
        outage = [x for row in rows for x in row['outage_hits']]
        result = dict(scenes=len(rows), identities=len(hits), retained=sum(hits),
                      mean_center_error=sum(errors)/len(errors),
                      mean_extra_moving_hypotheses=sum(row['extra_moving_hypotheses'] for row in rows)/len(rows),
                      outage_hits=sum(outage), outage_targets=len(outage), rows=rows)
        results['conditions'][condition] = result
        results['seconds'] = round(time.perf_counter()-start, 2)
        out.write_text(json.dumps(results, indent=2)+'\n')
        print(condition, json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()

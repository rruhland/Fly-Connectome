"""Self-supervised forecast capacity from actual observed visual histories."""

import copy
import json
import math
import time
from pathlib import Path

import torch

from associative_patch_state import AssociativePatchState
from event_only_observer_teacher import fit_event_only_observer
from fly_connectome.sensor import EventCamera
from generic_native_cadence import scene_events
from local_motion_dynamics import LocalMotionDynamics
from local_observation_model import LocalObservationModel
from run_zero_shot_visual_transfer import corrupt_events
from sensor_budget import anchored_sequence
from visual_history_benchmark import SHAPES


OUT = Path('docs/experiments/2026-09-26-local-dynamics-results.json')
KINDS = ('constant', 'period2', 'period3', 'period4', 'arc_left', 'arc_right')


def visual_episode(kind, phase, shape, *, angle=0., scale=1., background=False):
    camera = EventCamera(1, 64, 64)
    camera.previous.fill_(background)
    visible, events, truth = [], [], []
    direction = torch.tensor([math.sin(angle), math.cos(angle)])
    traveled = 0.
    cycles = dict(constant=(2,), period2=(1, 3), period3=(1, 2, 3), period4=(1, 1, 3, 2))
    for frame in range(20):
        if kind in cycles:
            center = torch.tensor([32., 32.])-26*direction+traveled*direction
            traveled += scale*cycles[kind][(frame+phase) % len(cycles[kind])]
        else:
            theta = angle+(frame+phase)*(.2 if kind == 'arc_left' else -.2)
            center = torch.tensor([32+10*scale*math.sin(theta), 32+10*scale*math.cos(theta)])
        center = center.round()
        image = torch.full((64, 64), background, dtype=torch.bool)
        for dy, dx in SHAPES[shape]:
            y, x = int(center[0])+dy, int(center[1])+dx
            if 0 <= y < 64 and 0 <= x < 64:
                image[y, x] = not background
        arrival = camera.observe(image[None])
        event = torch.zeros(2, 64*64)
        event[arrival.on.long(), arrival.pixels] = 1.
        events.append(event.reshape(2, 64, 64))
        visible.append(image)
        truth.append(center+torch.tensor(SHAPES[shape], dtype=torch.float32).mean(0))
    return dict(events=events, visible=visible), truth


@torch.no_grad()
def observed_positions(base, case, *, state_type=AssociativePatchState):
    observer = LocalObservationModel(height=64, width=64)
    observer.weights.copy_(base.weights)
    observer.bias.copy_(base.bias)
    # No missing observations here: the contextual gap correction is unused.
    state = state_type(height=64, width=64)
    positions = []
    for frame, sensory in enumerate(anchored_sequence(observer, case, period=1)):
        state.step(sensory)
        candidates = [e for e in state.entities if e['strength'] >= .5 and
                      state.tracker.slots[e['id']]['last_seen'] == frame]
        entity = min(candidates, key=lambda e: e['id']) if candidates else None
        positions.append(None if entity is None else entity['position'])
    return positions


def fixed_displacement(history, horizon, mode):
    if mode == 'persistence':
        return torch.zeros(2)
    if mode == 'velocity':
        return horizon*history[-1]
    if mode == 'mean_velocity':
        return horizon*history.mean(0)
    if mode == 'acceleration':
        return horizon*history[-1]+horizon*(horizon+1)/2*(history[-1]-history[-2])
    return sum((history[-2+(step % 2)] for step in range(horizon)), torch.zeros(2))


@torch.no_grad()
def forecast_positions(models, positions, *, online=False):
    models = copy.deepcopy(models)
    prior_targets = []
    generator = torch.Generator().manual_seed(992)
    predictions = {}
    names = (*models, 'persistence', 'velocity', 'acceleration', 'lag_two')
    for t in range(len(positions)):
        if online and t >= 5 and all(p is not None for p in positions[t-5:t+1]):
            observed = torch.stack(positions[t-5:t+1])
            differences = observed[1:]-observed[:-1]
            models['learned'].observe(differences[:4], differences[4])
            if prior_targets:
                index = int(torch.randint(len(prior_targets), (), generator=generator))
                models['shuffled'].observe(differences[:4], prior_targets[index])
            prior_targets.append(differences[4].clone())
        if t < 6:
            continue
        available = all(p is not None for p in positions[t-4:t+1])
        if available:
            prior = torch.stack(positions[t-4:t+1])
            history = prior[1:]-prior[:-1]
        for horizon in (1, 4, 8):
            arm = {}
            for name in names:
                displacement = (models[name].rollout(history, horizon) if name in models
                    else fixed_displacement(history, horizon, name)) if available else None
                arm[name] = positions[t]+displacement if available else None
            predictions[t, horizon] = arm
    return predictions


@torch.no_grad()
def main(*, state_type=AssociativePatchState, out=OUT, diverse=False, online=False):
    torch.set_num_threads(1)
    started = time.perf_counter()
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    examples = []
    generator = torch.Generator().manual_seed(810)
    for kind in KINDS:
        for phase in (0, 2):
            for shape in ('dot', 'square'):
                angle = float(torch.rand((), generator=generator))*2*math.pi if diverse else 0.
                scale = .8+.4*float(torch.rand((), generator=generator)) if diverse else 1.
                case, _ = visual_episode(kind, phase, shape, background=bool(phase),
                                         angle=angle, scale=scale)
                positions = observed_positions(base, case, state_type=state_type)
                for t in range(5, len(positions)):
                    if all(p is not None for p in positions[t-5:t+1]):
                        observed = torch.stack(positions[t-5:t+1])
                        differences = observed[1:]-observed[:-1]
                        examples.append((differences[:4], differences[4]))
    learned, shuffled = LocalMotionDynamics(), LocalMotionDynamics()
    permutation = torch.randperm(len(examples), generator=torch.Generator().manual_seed(991))
    for index, (history, target) in enumerate(examples):
        learned.observe(history, target)
        shuffled.observe(history, examples[int(permutation[index])][1])
    models = dict(learned=learned, shuffled=shuffled)
    names = (*models, 'persistence', 'velocity', 'acceleration', 'lag_two')
    results = dict(sensor='current visible frames plus events', state=state_type.__name__,
                   teacher='actually matched current positions only',
                   diverse=diverse, online=online,
                   training_pairs=len(examples), prototypes=len(learned.keys), families={})
    totals = {str(h): {name: [] for name in names} for h in (1, 4, 8)}
    missing, samples = 0, 0
    for kind in KINDS:
        errors = {str(h): {name: [] for name in names} for h in (1, 4, 8)}
        for phase in (3, 5):
            for angle in (0., math.pi/4, math.pi/2):
                case, truth = visual_episode(kind, phase, 'plus', angle=angle,
                                             scale=1.25, background=bool(phase % 2))
                positions = observed_positions(base, case, state_type=state_type)
                forecasts = forecast_positions(models, positions, online=online)
                for horizon in (1, 4, 8):
                    for t in range(6, len(truth)-horizon):
                        samples += 1
                        if forecasts[t, horizon]['learned'] is None:
                            missing += 1
                            for name in names:
                                errors[str(horizon)][name].append(64.)
                            continue
                        for name in names:
                            error = float((forecasts[t, horizon][name]-truth[t+horizon]).norm())
                            errors[str(horizon)][name].append(error)
        results['families'][kind] = {h: {name: sum(values)/len(values)
                                        for name, values in arm.items()} for h, arm in errors.items()}
        for h, arm in errors.items():
            for name, values in arm.items():
                totals[h][name].extend(values)
    results['mean_endpoint_error'] = {h: {name: sum(values)/len(values)
                                          for name, values in arm.items()} for h, arm in totals.items()}
    results.update(forecast_samples=samples, missing_tracking_samples=missing,
                   seconds=round(time.perf_counter()-started, 2))
    out.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

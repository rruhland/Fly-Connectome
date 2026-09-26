"""Shared, frozen sensory input and offline probes for visual representations."""

import torch

from generic_motion_probe import local_motion_map
from run_absolute_refresh import contrast_frames
from run_decisive_representation_audit import (ACTIVE, case_frames,
                                               object_mask)
from run_zero_shot_visual_transfer import corrupt_events


class SensoryControl:
    def reset_state(self):
        pass

    def step(self, features, *, learn=False):
        return features.clone()


@torch.no_grad()
def observed_inputs(observer, events, *, absolute=None, missing=()):
    """Create causal input planes; missing frames conceal both sensors."""
    observer.reset_state()
    trace = torch.zeros_like(events[0])
    if absolute is None:
        absolute = contrast_frames(events)
    missing = set(missing)
    inputs = []
    for frame, raw in enumerate(events):
        raw = torch.zeros_like(raw) if frame in missing else raw
        intensity = (absolute[frame] if frame % 8 == 0 and
                     frame not in missing else None)
        credible, _ = observer.step(raw, absolute=intensity)
        motion = local_motion_map(trace, credible).clamp(-1., 1.)
        trace.mul_(.875).add_(credible)
        contrast = observer.contrast
        inputs.append(torch.cat((credible,
                                 contrast.clamp(min=0.)[None],
                                 (-contrast).clamp(min=0.)[None],
                                 motion)))
    return inputs


@torch.no_grad()
def capture(model, inputs, *, learn=False):
    model.reset_state()
    return [model.step(value, learn=learn).clone() for value in inputs]


@torch.no_grad()
def fit_occupancy_probe(examples):
    """Frozen full-frame linear capacity probe, fit on calibration only."""
    positive = negative = None
    positive_count = negative_count = 0
    for state, mask in examples:
        channels = state.shape[0]
        flat = state.reshape(channels, -1)
        sites = mask.flatten()
        if positive is None:
            positive = torch.zeros(channels)
            negative = torch.zeros(channels)
        positive += flat[:, sites].sum(1)
        negative += flat[:, ~sites].sum(1)
        positive_count += int(sites.sum())
        negative_count += int((~sites).sum())
    positive /= max(positive_count, 1)
    negative /= max(negative_count, 1)
    weight = positive-negative
    threshold = float(weight @ (.5*(positive+negative)))
    return weight, threshold


def occupancy_mask(state, probe):
    weight, threshold = probe
    if not bool(weight.abs().sum()):
        return torch.zeros(state.shape[1:], dtype=torch.bool)
    return (weight[:, None, None]*state).sum(0) > threshold


def _feature(states, region):
    value = sum(state[:, region].sum(1) for frame, state in
                enumerate(states) if frame in ACTIVE)
    return value/value.norm().clamp(min=1e-8)


def _instant_feature(state, region):
    value = state[:, region].sum(1)
    return value/value.norm().clamp(min=1e-8)


def _cosine(first, second):
    a, b = first.flatten(), second.flatten()
    return float((a @ b)/(a.norm()*b.norm()).clamp(min=1e-8))


def _components(mask):
    """Count spatial islands large enough to represent visible evidence."""
    grid = mask.tolist()
    height, width = mask.shape
    seen = set()
    count = 0
    for y in range(height):
        for x in range(width):
            if not grid[y][x] or (y, x) in seen:
                continue
            stack = [(y, x)]
            seen.add((y, x))
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy-1, cx), (cy+1, cx),
                               (cy, cx-1), (cy, cx+1)):
                    if (0 <= ny < height and 0 <= nx < width and
                            grid[ny][nx] and (ny, nx) not in seen):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            count += size >= 2
    return count


@torch.no_grad()
def score_cases(model, cases, inputs):
    captured = [capture(model, episode) for episode in inputs]
    calibration = []
    for case, states in zip(cases, captured):
        if case['split'] == 'calibration':
            for frame in ACTIVE:
                calibration.append((states[frame], object_mask(
                    case['objects'][0], frame)))
    probe = fit_occupancy_probe(calibration)
    centroids = {}
    directions = ('up', 'down', 'left', 'right')
    for direction in directions:
        vectors = []
        for case, states in zip(cases, captured):
            if case['split'] != 'calibration':
                continue
            if case['objects'][0]['direction'] == direction:
                _, regions = case_frames(case)
                vectors.append(_feature(states, regions[0]))
        center = torch.stack(vectors).mean(0)
        centroids[direction] = center/center.norm().clamp(min=1e-8)
    counts = {}
    separated = []
    crossing = []
    densities = []
    for case, states in zip(cases, captured):
        split = case['split']
        row = counts.setdefault(split, dict(tp=0, fp=0, fn=0,
                                            direction_correct=0,
                                            direction_total=0))
        for frame in ACTIVE:
            actual = torch.zeros_like(states[frame][0], dtype=torch.bool)
            for item in case['objects']:
                actual |= object_mask(item, frame)
            predicted = occupancy_mask(states[frame], probe)
            row['tp'] += int((predicted & actual).sum())
            row['fp'] += int((predicted & ~actual).sum())
            row['fn'] += int((~predicted & actual).sum())
            densities.append(float((states[frame].abs() > .05).float().mean()))
            if split == 'separated' and frame == 8:
                separated.append(_components(predicted) == 2)
        _, regions = case_frames(case)
        for item, region in zip(case['objects'], regions):
            feature = _feature(states, region)
            guess = max(directions, key=lambda direction: float(
                feature @ centroids[direction]))
            row['direction_correct'] += guess == item['direction']
            row['direction_total'] += 1
        if split == 'crossing':
            first, second = case['objects']
            early = [_instant_feature(states[6], object_mask(item, 6))
                     for item in (first, second)]
            late = [_instant_feature(states[12], object_mask(item, 12))
                    for item in (first, second)]
            same = sum(float(early[i] @ late[i]) for i in range(2))
            swap = sum(float(early[i] @ late[1-i]) for i in range(2))
            crossing.append(same > swap)
    for row in counts.values():
        row['occupancy_f1'] = 2*row['tp']/max(2*row['tp']+row['fp']+row['fn'], 1)
        row['direction_accuracy'] = row['direction_correct']/max(
            row['direction_total'], 1)
    return dict(splits=counts,
                separated_two_components=sum(separated)/max(len(separated), 1),
                crossing_identity=sum(crossing)/max(len(crossing), 1),
                active_fraction=sum(densities)/max(len(densities), 1))


@torch.no_grad()
def score_perturbations(model, clean_inputs, noisy_inputs,
                        missing_inputs):
    clean = [capture(model, episode) for episode in clean_inputs]
    noisy = [capture(model, episode) for episode in noisy_inputs]
    missing = [capture(model, episode) for episode in missing_inputs]
    clean_corrupt = []
    clean_unrelated = []
    gap = []
    recovery = []
    for index, states in enumerate(clean):
        other = clean[(index+1) % len(clean)]
        for frame in (20, 40, 60):
            clean_corrupt.append(_cosine(states[frame], noisy[index][frame]))
            clean_unrelated.append(_cosine(states[frame], other[frame]))
        gap.append(_cosine(states[32], missing[index][32]))
        recovery.append(_cosine(states[34], missing[index][34]))
    mean = lambda values: sum(values)/max(len(values), 1)
    return dict(noise_margin=mean(clean_corrupt)-mean(clean_unrelated),
                missing_gap_similarity=mean(gap),
                missing_recovery_similarity=mean(recovery),
                clean_corrupt_similarity=mean(clean_corrupt),
                clean_unrelated_similarity=mean(clean_unrelated))


@torch.no_grad()
def heldout_inputs(observer, episodes):
    clean = [observed_inputs(observer, events) for events in episodes]
    noisy = [observed_inputs(
        observer, corrupt_events(events, seed=2000+index),
        absolute=contrast_frames(events))
        for index, events in enumerate(episodes)]
    missing = [observed_inputs(observer, events, missing=(30, 31, 32))
               for events in episodes]
    return clean, noisy, missing

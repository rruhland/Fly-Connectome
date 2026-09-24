"""Opt-in unlabeled local event-triplet learning across shapes."""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fly_connectome.sensor import EventCamera
from generic_motion_probe import SHAPES, object_mask, item, events_map


OUT = Path('docs/experiments/2026-09-23-generic-local-transition-results.json')
OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1),
           (-2, 0), (0, 2), (2, 0), (0, -2))


def shift(value, dy, dx):
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    source_y = slice(max(0, -dy), height-max(0, dy))
    target_y = slice(max(0, dy), height-max(0, -dy))
    source_x = slice(max(0, -dx), width-max(0, dx))
    target_x = slice(max(0, dx), width-max(0, -dx))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


class LocalTripletLearner:
    """Shared retinotopic efficacy, updated by delayed local visual error."""

    def __init__(self, *, eta, local_competition=False):
        if not 0 < eta <= 1:
            raise ValueError('local learning rate must be in (0,1]')
        self.eta = eta
        self.local_competition = local_competition
        self.weights = torch.zeros((2, len(OFFSETS)))
        self.reset_state()
        self.confirmed_updates = 0

    def reset_state(self):
        self.previous = None
        self.pending_features = None
        self.pending_prediction = None

    @torch.no_grad()
    def step(self, current, *, learn=True):
        if current.shape != (2, 32, 64):
            raise ValueError('two-polarity 32x64 event map required')
        if learn and self.pending_features is not None:
            error = current-self.pending_prediction
            for index, (dy, dx) in enumerate(OFFSETS):
                aligned_error = shift(error, -dy, -dx)
                eligible = self.pending_features[:, index]
                count = eligible.sum((1, 2))
                numerator = (eligible*aligned_error).sum((1, 2))
                update = self.eta*numerator/count.clamp(min=1)
                self.weights[:, index].add_(update).clamp_(0, 1)
                self.confirmed_updates += int((count > 0).sum())
        features = torch.stack([
            shift(self.previous, dy, dx)*current
            for dy, dx in OFFSETS]) if self.previous is not None else torch.zeros(
                (len(OFFSETS), 2, 32, 64))
        if self.local_competition:
            support = F.avg_pool2d(features.reshape(16, 1, 32, 64),
                                   kernel_size=7, stride=1, padding=3).reshape(
                                       len(OFFSETS), 2, 32, 64)
            winner = support.argmax(dim=0, keepdim=True)
            features = features * (winner == torch.arange(len(OFFSETS))[
                :, None, None, None])
        prediction = torch.zeros_like(current)
        for index, (dy, dx) in enumerate(OFFSETS):
            prediction += shift(features[index], dy, dx)*self.weights[:, index, None, None]
        prediction.clamp_(0, 1)
        self.previous = current.clone()
        self.pending_features = features.permute(1, 0, 2, 3)
        self.pending_prediction = prediction.clone()
        return prediction


def event_sequence(objects, *, background):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    result = []
    for frame in range(18):
        foreground = torch.zeros((32, 64), dtype=torch.bool)
        if 2 <= frame < 15:
            for obj in objects:
                step = frame-8 if obj['moving'] else 0
                center = (obj['center'][0]+obj['velocity'][0]*step,
                          obj['center'][1]+obj['velocity'][1]*step)
                foreground |= object_mask(obj['shape'], center)
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        image[0, foreground] = not background
        result.append(events_map(camera.observe(image)))
    return result


def accumulate(scores, prediction, target):
    selected = prediction >= .5
    actual = target > 0
    scores['tp'] += int((selected & actual).sum())
    scores['fp'] += int((selected & ~actual).sum())
    scores['fn'] += int((~selected & actual).sum())
    candidate = actual | (prediction >= .05)
    scores['squared_error'] += float(((prediction-target).square()*candidate).sum())
    scores['candidate_samples'] += int(candidate.sum())


def empty_score():
    return dict(tp=0, fp=0, fn=0, squared_error=0., candidate_samples=0)


def finish(score):
    tp, fp, fn = (score[name] for name in ('tp', 'fp', 'fn'))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(**score, precision=precision, recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        candidate_mse=score['squared_error']/score['candidate_samples']
                      if score['candidate_samples'] else None)


def evaluate(learner, cases):
    names = ('learned', 'persistence', 'unit', 'zero')
    if learner.local_competition:
        names += ('unit_competitive',)
    scores = {name: empty_score() for name in names}
    groups = {}
    unit = LocalTripletLearner(eta=learner.eta)
    unit.weights.fill_(1)
    unit_competitive = None
    if learner.local_competition:
        unit_competitive = LocalTripletLearner(eta=learner.eta,
                                             local_competition=True)
        unit_competitive.weights.fill_(1)
    for label, sequence in cases:
        learner.reset_state()
        unit.reset_state()
        if unit_competitive is not None:
            unit_competitive.reset_state()
        group = groups.setdefault(label, {name: empty_score() for name in scores})
        for t, current in enumerate(sequence):
            prediction = learner.step(current, learn=False)
            unit_prediction = unit.step(current, learn=False)
            competitive_prediction = (unit_competitive.step(current, learn=False)
                                      if unit_competitive is not None else None)
            if 3 <= t < 14:
                target = sequence[t+1]
                candidates = dict(learned=prediction, persistence=current,
                                  unit=unit_prediction, zero=torch.zeros_like(current))
                if competitive_prediction is not None:
                    candidates['unit_competitive'] = competitive_prediction
                for name, value in candidates.items():
                    accumulate(scores[name], value, target)
                    accumulate(group[name], value, target)
    return dict(overall={name: finish(value) for name, value in scores.items()},
                groups={label: {name: finish(value) for name, value in entries.items()}
                        for label, entries in groups.items()})


@torch.no_grad()
def main():
    started = time.perf_counter()
    learner = LocalTripletLearner(eta=.3)
    training = []
    for direction in ('up', 'down', 'left', 'right'):
        centers = ((16, 20), (16, 44)) if direction in ('up', 'down') else (
            (10, 32), (22, 32))
        for center in centers:
            for speed in (1, 2):
                for background in (True, False):
                    training.append(event_sequence(
                        [item('square', center, direction, speed)],
                        background=background))
    rng = random.Random(0)
    for _ in range(3):
        rng.shuffle(training)
        for sequence in training:
            learner.reset_state()
            for current in sequence:
                learner.step(current)

    cases = []
    for shape in SHAPES:
        for direction in ('up', 'down', 'left', 'right'):
            center = (16, 32)
            for speed in (1, 2):
                for background in (True, False):
                    cases.append((f'{shape}:{direction}:s{speed}:'
                                  f'{"dark" if background else "bright"}',
                        event_sequence([item(shape, center, direction, speed)],
                                       background=background)))
    for background in (True, False):
        static = dict(shape='ell', center=(16, 32),
                      velocity=(0, 0), moving=False)
        cases.append((f'static:{"dark" if background else "bright"}',
                      event_sequence([static], background=background)))
    pairs = [item('square', (16, 20), 'up', 1),
             item('plus', (16, 44), 'down', 1)]
    cases.append(('two_objects', event_sequence(pairs, background=True)))

    result = dict(eta=learner.eta, training_episodes=3*len(training),
        confirmed_local_updates=learner.confirmed_updates,
        weights=learner.weights.tolist(), evaluation=evaluate(learner, cases),
        elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(training_episodes=result['training_episodes'],
        confirmed_local_updates=result['confirmed_local_updates'],
        weights=result['weights'],
        overall={name: dict(f1=row['f1'], candidate_mse=row['candidate_mse'])
                 for name, row in result['evaluation']['overall'].items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

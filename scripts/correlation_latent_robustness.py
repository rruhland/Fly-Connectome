"""Frozen evaluation of learned local visual state on varied generic scenes."""

import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from diverse_visual_experience import (OFFSETS, diverse_training_sequences,
                                       unseen_shape_sequences)
from fly_connectome.sensor import EventCamera
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from generic_motion_probe import events_map
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import evaluate as evaluate_local


OUT = Path('docs/experiments/2026-09-24-correlation-latent-robustness-results.json')


def scene_sequence(objects, *, background, noise=0., seed=0):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    generator = torch.Generator().manual_seed(seed)
    sequence = []
    for frame in range(18):
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        if 2 <= frame < 15:
            for obj in objects:
                if frame in obj.get('hidden', ()):
                    continue
                dy, dx = obj['before'] if frame < 8 else obj['after']
                cy = obj['center'][0]+dy*(frame-8)
                cx = obj['center'][1]+dx*(frame-8)
                for oy, ox in OFFSETS[obj['shape']]:
                    y, x = cy+oy, cx+ox
                    if 0 <= y < 32 and 0 <= x < 64:
                        image[0, y, x] = not background
        if noise:
            image ^= torch.rand(image.shape, generator=generator) < noise
        sequence.append(events_map(camera.observe(image)))
    return sequence


def make_robust_cases():
    cases = []
    for axis in ('vertical', 'horizontal'):
        for speed in (1, 2):
            if axis == 'vertical':
                first = dict(shape='diamond', center=(16, 20),
                             before=(-speed, 0), after=(-speed, 0))
                second = dict(shape='ring', center=(16, 44),
                              before=(speed, 0), after=(speed, 0))
                crossing = [dict(shape='diamond', center=(16, 32),
                                 before=(speed, 0), after=(speed, 0)),
                            dict(shape='zigzag', center=(16, 32),
                                 before=(-speed, 0), after=(-speed, 0))]
            else:
                first = dict(shape='diamond', center=(10, 32),
                             before=(0, -speed), after=(0, -speed))
                second = dict(shape='ring', center=(22, 32),
                              before=(0, speed), after=(0, speed))
                crossing = [dict(shape='diamond', center=(16, 32),
                                 before=(0, speed), after=(0, speed)),
                            dict(shape='zigzag', center=(16, 32),
                                 before=(0, -speed), after=(0, -speed))]
            for background in (True, False):
                meta = dict(axis=axis, speed=speed, background=background)
                cases.append(('independent', meta, scene_sequence(
                    [first, second], background=background)))
                cases.append(('crossing', meta, scene_sequence(
                    crossing, background=background)))
    for shape in ('diamond', 'ring'):
        for speed in (1, 2):
            obj = dict(shape=shape, center=(16, 32),
                       before=(0, speed), after=(0, speed),
                       hidden=(7, 8, 9))
            for background in (True, False):
                meta = dict(shape=shape, speed=speed, background=background)
                cases.append(('occlusion', meta, scene_sequence(
                    [obj], background=background)))
    directions = {'up': (-1, 0), 'down': (1, 0),
                  'left': (0, -1), 'right': (0, 1)}
    for direction, (dy, dx) in directions.items():
        for before, after in ((1, 2), (2, 1)):
            obj = dict(shape='diamond', center=(16, 32),
                       before=(dy*before, dx*before),
                       after=(dy*after, dx*after))
            for background in (True, False):
                meta = dict(direction=direction, speed=after,
                            background=background, before=before)
                cases.append(('speed_change', meta, scene_sequence(
                    [obj], background=background)))
    for shape in ('diamond', 'ring'):
        for speed in (1, 2):
            obj = dict(shape=shape, center=(16, 32),
                       before=(0, speed), after=(0, speed))
            for background in (True, False):
                meta = dict(shape=shape, speed=speed, background=background)
                cases.append(('noise', meta, scene_sequence(
                    [obj], background=background, noise=.002, seed=17)))
    return cases


@torch.no_grad()
def train_models():
    passes = [[(events, correlation_sequence(events))
               for *_, events in episodes]
              for episodes in diverse_training_sequences()]
    learned = TransitionPopulation(channels=16, seed=0)
    random_code = TransitionPopulation(channels=16, seed=0)
    rng = random.Random(0)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, codes in order:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)
    rng = random.Random(1)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, codes in order:
            # Match the original run's shuffled-control draws so later
            # episode orders and both trained models are identical.
            shuffled_future = codes[1:].copy()
            rng.shuffle(shuffled_future)
            learned.reset_state()
            random_code.reset_state()
            for code in codes:
                learned.step(code, learn_prediction=True)
                random_code.step(code, learn_prediction=True)
    return dict(learned=learned, random_dictionary=random_code)


@torch.no_grad()
def selected_transition_scores(models, cases):
    scores = {family: {name: empty_score() for name in (*models, 'fixed')}
              for family in ('occlusion', 'speed_change')}
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, _, events in cases:
        if family not in scores:
            continue
        codes = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        fixed.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            predictions = {name: primitive_to_events(model.step(code))
                           for name, model in models.items()}
            fixed_prediction = fixed.step(event, learn=False)
            selected = (family == 'occlusion' and t in (6, 9)) or (
                family == 'speed_change' and t == 8)
            if selected:
                for name, prediction in predictions.items():
                    accumulate(scores[family][name], prediction, events[t+1])
                accumulate(scores[family]['fixed'], fixed_prediction,
                           events[t+1])
    return {family: {name: finish(value) for name, value in methods.items()}
            for family, methods in scores.items()}


@torch.no_grad()
def family_scores(robust_scores, cases):
    families = {family for family, _, _ in cases}
    scores = {family: {name: empty_score()
                       for name in ('learned', 'random_dictionary', 'fixed')}
              for family in families}
    for label, methods in robust_scores['by_shape_polarity'].items():
        family = label.split(':', 1)[0]
        for name, row in methods.items():
            for key in scores[family][name]:
                scores[family][name][key] += row['event'][key]
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, _, events in cases:
        fixed.reset_state()
        for t, event in enumerate(events):
            prediction = fixed.step(event, learn=False)
            if 3 <= t < 14:
                accumulate(scores[family]['fixed'], prediction, events[t+1])
    return {family: {name: finish(value) for name, value in methods.items()}
            for family, methods in scores.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    models = train_models()
    decoder = torch.zeros((16, 2))
    decoder[:8, 0] = 1
    decoder[8:, 1] = 1
    original = [(shape, direction, speed, background, events,
                 correlation_sequence(events))
                for shape, direction, speed, background, events
                in unseen_shape_sequences()]
    original_scores = evaluate_local(models, original, decoder)
    raw_robust = make_robust_cases()
    robust = [(family, 'mixed', meta['speed'], meta['background'], events,
               correlation_sequence(events))
              for family, meta, events in raw_robust]
    robust_scores = evaluate_local(models, robust, decoder)
    grouped = family_scores(robust_scores, raw_robust)
    transitions = selected_transition_scores(models, raw_robust)
    result = dict(original=original_scores, robust=robust_scores,
                  families=grouped, transitions=transitions,
                  cases=len(raw_robust),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        original={name: row['event']['f1']
                  for name, row in original_scores['overall'].items()},
        robust={name: {model: value['f1'] for model, value in methods.items()}
                for name, methods in grouped.items()},
        fixed=robust_scores['fixed_correlation_event']['f1'],
        transition={family: {name: row['f1'] for name, row in methods.items()}
                    for family, methods in transitions.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()

"""Matched opt-in test of shape-diverse experience for local visual learning."""

import json
import random
import time
from pathlib import Path

import torch

from fly_connectome.sensor import EventCamera
from frozen_code_capacity import evaluate as evaluate_capacity, fit_decoder
from generic_motion_probe import SHAPES, events_map
from learned_latent_probe import DIRECTIONS, LocalVisualLatent
from learned_transition_units import TransitionPopulation
from run_learned_transition_units import encode, evaluate as evaluate_local, probe


OUT = Path('docs/experiments/2026-09-24-diverse-visual-experience-results.json')
TRAIN_SHAPES = ('square', 'plus', 'ell', 'bar')
TEST_SHAPES = {
    'diamond': [(y, x) for y in range(-2, 3) for x in range(-2, 3)
                if abs(y)+abs(x) <= 2],
    'ring': [(y, x) for y in range(-2, 3) for x in range(-2, 3)
             if abs(y) == 2 or abs(x) == 2],
    'zigzag': [(-2, -2), (-2, -1), (-1, -1), (-1, 0), (0, 0),
               (0, 1), (1, 1), (1, 2), (2, 2)],
}
OFFSETS = {**SHAPES, **TEST_SHAPES}
VELOCITY = {'up': (-1, 0), 'down': (1, 0),
            'left': (0, -1), 'right': (0, 1)}


def event_sequence_for(shape, center, direction, speed, background):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    dy, dx = VELOCITY[direction]
    result = []
    for frame in range(18):
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        if 2 <= frame < 15:
            cy = center[0]+dy*speed*(frame-8)
            cx = center[1]+dx*speed*(frame-8)
            for oy, ox in OFFSETS[shape]:
                y, x = cy+oy, cx+ox
                if 0 <= y < 32 and 0 <= x < 64:
                    image[0, y, x] = not background
        result.append(events_map(camera.observe(image)))
    return result


def square_training_sequences():
    episodes = []
    for direction in DIRECTIONS:
        centers = ((16, 20), (16, 44)) if direction in ('up', 'down') else (
            (10, 32), (22, 32))
        for center in centers:
            for speed in (1, 2):
                for background in (True, False):
                    episodes.append(('square', direction, speed, background,
                                     event_sequence_for('square', center,
                                                        direction, speed,
                                                        background)))
    return [episodes.copy() for _ in range(3)]


def diverse_training_sequences(epochs=3):
    passes = []
    for epoch in range(epochs):
        episodes = []
        for shape_index, shape in enumerate(TRAIN_SHAPES):
            for direction_index, direction in enumerate(DIRECTIONS):
                centers = ((16, 20), (16, 44)) if direction in ('up', 'down') else (
                    (10, 32), (22, 32))
                for speed in (1, 2):
                    center = centers[(shape_index+direction_index+speed+epoch) % 2]
                    contrast_epoch = epoch if epoch < 3 else epoch+epoch//3
                    background = (shape_index+direction_index+contrast_epoch) % 2 == 0
                    episodes.append((shape, direction, speed, background,
                                     event_sequence_for(shape, center, direction,
                                                        speed, background)))
        passes.append(episodes)
    return passes


def unseen_shape_sequences():
    return [(shape, direction, speed, background,
             event_sequence_for(shape, (16, 32), direction, speed, background))
            for shape in TEST_SHAPES
            for direction in DIRECTIONS
            for speed in (1, 2)
            for background in (True, False)]


@torch.no_grad()
def train_arm(passes, raw_heldout):
    sensory = LocalVisualLatent(seed=0, homeostasis=True)
    rng = random.Random(0)
    for episodes in passes:
        order = episodes.copy()
        rng.shuffle(order)
        for _, _, _, _, events in order:
            sensory.reset_state()
            for event in events:
                sensory.step(event)
    training_passes = [[(shape, direction, speed, background, events,
                         encode(sensory, events))
                        for shape, direction, speed, background, events in episodes]
                       for episodes in passes]
    training = [case for episodes in training_passes for case in episodes]
    heldout = [(shape, direction, speed, background, events,
                encode(sensory, events))
               for shape, direction, speed, background, events in raw_heldout]
    learned = TransitionPopulation(seed=0)
    random_code = TransitionPopulation(seed=0)
    rng = random.Random(0)
    for episodes in training_passes:
        order = episodes.copy()
        rng.shuffle(order)
        for *_, codes in order:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_dictionary=True)
    rng = random.Random(1)
    for episodes in training_passes:
        order = episodes.copy()
        rng.shuffle(order)
        for *_, codes in order:
            learned.reset_state()
            for code in codes:
                learned.step(code, learn_prediction=True)
    decoder = sensory.sensory[:, [12, 37]]
    local = evaluate_local({'learned': learned}, heldout, decoder)
    local_train = evaluate_local({'learned': learned}, training, decoder)
    decoder_training = [(direction, events, codes)
                        for _, direction, _, _, events, codes in training]
    learned_weights, learned_fit = fit_decoder(learned, decoder_training)
    random_weights, random_fit = fit_decoder(random_code, decoder_training)
    capacity_models = {'learned': (learned, learned_weights),
                       'random_dictionary': (random_code, random_weights)}
    capacity = evaluate_capacity(capacity_models, heldout)
    capacity_train = evaluate_capacity(capacity_models, training)
    return dict(training_episodes=sum(map(len, passes)),
                code_probe=probe(learned, decoder_training, heldout),
                random_code_probe=probe(random_code, decoder_training, heldout),
                local_prediction=local,
                local_training=local_train['overall']['learned'],
                capacity=capacity,
                capacity_training=capacity_train['overall'],
                fit={'learned': learned_fit, 'random_dictionary': random_fit})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    heldout = unseen_shape_sequences()
    square = train_arm(square_training_sequences(), heldout)
    diverse = train_arm(diverse_training_sequences(), heldout)
    result = dict(square=square, diverse=diverse,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({
        name: dict(code_probe=arm['code_probe']['accuracy'],
                   local_event_f1=arm['local_prediction']['overall']['learned']['event']['f1'],
                   local_train_f1=arm['local_training']['event']['f1'],
                   capacity_event_f1=arm['capacity']['overall']['learned']['f1'],
                   capacity_train_f1=arm['capacity_training']['learned']['f1'],
                   random_capacity_f1=arm['capacity']['overall']['random_dictionary']['f1'])
        for name, arm in (('square', square), ('diverse', diverse))}), flush=True)


if __name__ == '__main__':
    main()

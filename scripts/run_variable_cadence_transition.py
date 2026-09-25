"""Local recurrent learning from generic fractional- and whole-pixel motion."""

import copy
import json
import math
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import OFFSETS, TRAIN_SHAPES
from fly_connectome.sensor import EventCamera
from gap_timing_transfer import MODEL_OUT
from generic_motion_probe import events_map
from history_gated_code import DIRECTIONS
from local_hidden_transition import LocalHiddenTransition
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_fast)
from run_local_visibility_likelihood import clean_training_cases
from run_native_trace_support import trace_correlation_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-variable-cadence-transition-results.json')
SHAPES = {**OFFSETS, 'dot': ((0, 0),)}
ARMS = ('fast_frozen', 'variable_adapted', 'variable_scratch', 'persistence')


def continuous_events(shape, center, direction, speed, background):
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(background)
    result = []
    dy, dx = direction
    for frame in range(18):
        image = torch.full((1, 32, 64), background, dtype=torch.bool)
        if 2 <= frame < 15:
            cy = math.floor(center[0]+dy*speed*(frame-8)+.5)
            cx = math.floor(center[1]+dx*speed*(frame-8)+.5)
            for oy, ox in SHAPES[shape]:
                y, x = cy+oy, cx+ox
                if 0 <= y < 32 and 0 <= x < 64:
                    image[0, y, x] = not background
        result.append(events_map(camera.observe(image)))
    return result


def training_cases():
    cases = []
    for shape_index, shape in enumerate((*TRAIN_SHAPES, 'dot')):
        for direction_index, velocity in enumerate(DIRECTIONS.values()):
            for speed_index, speed in enumerate((.25, .5, 1, 2)):
                for background in (False, True):
                    center = ((12, 24), (20, 40))[
                        (shape_index+direction_index+speed_index
                         +int(background)) % 2]
                    cases.append(continuous_events(
                        shape, center, velocity, speed, background))
    return cases


@torch.no_grad()
def train_variable(fast, cases):
    adapted = copy.deepcopy(fast)
    scratch = LocalHiddenTransition(copy.deepcopy(fast.encoder))
    order = list(range(len(cases)))
    random.Random(3).shuffle(order)
    for index in order:
        codes = trace_correlation_sequence(cases[index])
        for model in (adapted, scratch):
            model.reset_state()
        for code in codes:
            adapted.step(code, learn=True)
            scratch.step(code, learn=True)
    return adapted, scratch


@torch.no_grad()
def evaluate_native(models, *, seeds=SEEDS, frames=120):
    scores = {name: empty_score() for name in ARMS}
    counts = dict(frames=0, code_active=0, latent_active=0)
    for seed in seeds:
        events = pong_events(seed, stride=1, frames=frames)
        codes = trace_correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        for t, code in enumerate(codes):
            predictions = {name: model.pending for name, model in
                           models.items()}
            predictions['persistence'] = models[
                'fast_frozen'].observed.clone()
            for model in models.values():
                model.step(code)
            if t < 4:
                continue
            target = models['fast_frozen'].observed
            counts['frames'] += 1
            counts['code_active'] += int(code.sum() > 0)
            counts['latent_active'] += int(target.sum() > 0)
            if target.sum() > 0:
                for name, prediction in predictions.items():
                    add_score(scores[name], prediction, target)
    return dict(counts=counts,
                active={name: finish(score) for name, score in scores.items()})


@torch.no_grad()
def evaluate_generic(models, cases):
    scores = {family: {name: empty_score() for name in ARMS}
              for family in ('independent', 'crossing')}
    for family, _, events in cases:
        if family not in scores:
            continue
        codes = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        for t, code in enumerate(codes[:14]):
            predictions = {name: model.pending for name, model in
                           models.items()}
            predictions['persistence'] = models[
                'fast_frozen'].observed.clone()
            for model in models.values():
                model.step(code)
            if 4 <= t <= 13:
                target = models['fast_frozen'].observed
                for name, prediction in predictions.items():
                    add_score(scores[family][name], prediction, target)
    return {family: {name: finish(score) for name, score in arms.items()}
            for family, arms in scores.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    fast = train_fast(saved['models']['learned_split'].code.motion,
                      [(kind, '', events) for kind, events in clean])['learned']
    cases = training_cases()
    adapted, scratch = train_variable(fast, cases)
    training_seconds = time.perf_counter()-started
    print(f'variable-cadence transition trained on {len(cases)} episodes '
          f'in {training_seconds:.1f}s', flush=True)
    models = dict(fast_frozen=fast, variable_adapted=adapted,
                  variable_scratch=scratch)
    result = dict(fast_training_episodes=len(clean),
                  variable_training_episodes=len(cases),
                  native=evaluate_native(models),
                  generic=evaluate_generic(models, make_robust_cases()),
                  weights={name: dict(sum=float(model.weights.sum()),
                                      change_from_fast=float((
                                          model.weights-fast.weights).abs().sum()))
                           for name, model in models.items()},
                  training_seconds=training_seconds,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        native={name: (round(score['f1'], 3),
                       round(score['active_ratio'], 3))
                for name, score in result['native']['active'].items()},
        generic={family: {name: round(score['f1'], 3)
                          for name, score in arms.items()}
                 for family, arms in result['generic'].items()},
        training_seconds=round(training_seconds, 1),
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

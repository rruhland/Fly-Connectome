"""Opt-in test of locally learned hidden visual motion continuity."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import scene_sequence
from diverse_visual_experience import unseen_shape_sequences
from gap_timing_transfer import MODEL_OUT, timing_cases
from history_gated_code import DIRECTIONS
from local_hidden_transition import LocalHiddenTransition
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-25-local-hidden-transition-results.json')
ARMS = ('learned', 'shuffled_credit', 'frozen', 'state_reset',
        'post_entry_reset')
GAP_ARMS = ARMS+('pre_gap_hold', 'first_blank_hold')


def add_score(score, prediction, target):
    predicted = prediction.sum(0) >= .5
    actual = target.sum(0) >= .5
    score['tp'] += int((predicted & actual).sum())
    score['fp'] += int((predicted & ~actual).sum())
    score['fn'] += int((~predicted & actual).sum())
    score['predicted_sites'] += int(predicted.sum())
    score['target_sites'] += int(actual.sum())
    score['frames'] += 1


def finish(score):
    tp, fp, fn = (score[key] for key in ('tp', 'fp', 'fn'))
    return {**score, 'f1': 2*tp/max(2*tp+fp+fn, 1),
            'active_ratio': score['predicted_sites']/max(score['target_sites'], 1)}


def empty_score():
    return dict(tp=0, fp=0, fn=0, predicted_sites=0,
                target_sites=0, frames=0)


@torch.no_grad()
def train(encoder, cases):
    models = {name: LocalHiddenTransition(copy.deepcopy(encoder))
              for name in ARMS if name not in ('state_reset',
                                               'post_entry_reset')}
    encoded = [correlation_sequence(events) for _, _, events in cases]
    order = list(range(len(encoded)))
    random.Random(3).shuffle(order)
    credit_encoder = copy.deepcopy(encoder)
    for n, index in enumerate(order):
        paired = encoded[order[(n+1) % len(order)]]
        credit_encoder.reset_state()
        for model in models.values():
            model.reset_state()
        for code, unrelated in zip(encoded[index], paired):
            credit_encoder.step(unrelated)
            models['learned'].step(code, learn=True)
            models['shuffled_credit'].step(
                code, learn=True, credit_target=credit_encoder.latent)
    models['state_reset'] = copy.deepcopy(models['learned'])
    models['post_entry_reset'] = copy.deepcopy(models['learned'])
    return models


def visible_cases():
    cases = [events for *_, events in unseen_shape_sequences()]
    for center in ((12, 24), (20, 40)):
        for shape, direction, speed, background, _ in unseen_shape_sequences():
            dy, dx = DIRECTIONS[direction]
            obj = dict(shape=shape, center=center,
                       before=(dy*speed, dx*speed),
                       after=(dy*speed, dx*speed))
            cases.append(scene_sequence([obj], background=background))
    return cases


def gap_cases():
    cases = []
    for window, shape, direction, speed, background, hidden, events in timing_cases():
        dy, dx = DIRECTIONS[direction]
        for center in ((16, 32), (12, 24), (20, 40)):
            if center != (16, 32):
                obj = dict(shape=shape, center=center,
                           before=(dy*speed, dx*speed),
                           after=(dy*speed, dx*speed), hidden=hidden)
                chosen = scene_sequence([obj], background=background)
            else:
                chosen = events
            cases.append((window, shape, direction, speed, background,
                          hidden, center, chosen))
    return cases


@torch.no_grad()
def evaluate_visible(models, cases):
    scores = {name: empty_score() for name in ARMS
              if name not in ('state_reset', 'post_entry_reset')}
    for events in cases:
        codes = correlation_sequence(events)
        for name, model in models.items():
            if name in ('state_reset', 'post_entry_reset'):
                continue
            model.reset_state()
            for t, code in enumerate(codes):
                preceding = model.pending
                model.step(code)
                if 4 <= t <= 13:
                    add_score(scores[name], preceding, model.observed)
    return {name: finish(score) for name, score in scores.items()}


def spatial_center(state):
    sites = (state.sum(0) >= .5).nonzero(as_tuple=False)
    if len(sites) == 0:
        return None
    return sites.float().mean(0)


@torch.no_grad()
def evaluate_gaps(models, cases):
    scores = {window: {str(i+1): {name: empty_score() for name in GAP_ARMS}
                       for i in range(len(hidden))}
              for window, _, _, _, _, hidden, _, _ in cases}
    motion = {window: {name: dict(correct=0, valid=0, cases=0,
                                  axis_error_sum=0.) for name in ARMS}
              for window, *_ in cases}
    reference = copy.deepcopy(models['learned'].encoder)
    for window, shape, direction, speed, background, hidden, center, events in cases:
        dy, dx = DIRECTIONS[direction]
        obj = dict(shape=shape, center=center,
                   before=(dy*speed, dx*speed),
                   after=(dy*speed, dx*speed))
        unoccluded = scene_sequence([obj], background=background)
        codes = correlation_sequence(events)
        counterfactual_codes = correlation_sequence(unoccluded)
        reference.reset_state()
        for model in models.values():
            model.reset_state()
        first_centers = {}
        for t in range(hidden[-1]+1):
            reference.step(counterfactual_codes[t])
            if t == hidden[0]:
                pre_gap_state = models['learned'].state.clone()
            if t == hidden[0]:
                models['state_reset'].clear_hidden()
            if t == hidden[0]+1:
                models['post_entry_reset'].clear_hidden()
            for model in models.values():
                model.step(codes[t])
            if t in hidden:
                if t == hidden[0]:
                    first_blank_state = models['learned'].state.clone()
                    first_centers = {name: spatial_center(model.state)
                                     for name, model in models.items()}
                frame = str(t-hidden[0]+1)
                for name, model in models.items():
                    add_score(scores[window][frame][name], model.state,
                              reference.latent)
                for name, state in (('pre_gap_hold', pre_gap_state),
                                    ('first_blank_hold', first_blank_state)):
                    add_score(scores[window][frame][name], state,
                              reference.latent)
                if t == hidden[-1]:
                    axis = 0 if dy else 1
                    expected = (dy if dy else dx)*speed*(len(hidden)-1)
                    for name, model in models.items():
                        row = motion[window][name]
                        row['cases'] += 1
                        last = spatial_center(model.state)
                        if first_centers[name] is None or last is None:
                            continue
                        displacement = float(last[axis]-first_centers[name][axis])
                        row['valid'] += 1
                        row['correct'] += int(displacement*expected > .25)
                        row['axis_error_sum'] += abs(displacement-expected)
    return dict(occupancy={window: {
        frame: {name: finish(score) for name, score in by_name.items()}
        for frame, by_name in by_frame.items()}
        for window, by_frame in scores.items()},
        motion={window: {name: {**row,
             'direction_accuracy': row['correct']/row['cases'],
             'mean_axis_error': (row['axis_error_sum']/row['valid']
                                 if row['valid'] else None)}
             for name, row in arms.items()}
            for window, arms in motion.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    encoder = saved['models']['learned_split'].code.motion
    training = [(kind, shape, events)
                for kind, shape, events in make_training_cases()
                if kind == 'clean']
    models = train(encoder, training)
    train_seconds = time.perf_counter()-started
    print(f'trained {len(training)} episodes in {train_seconds:.1f}s', flush=True)
    visible = visible_cases()
    gaps = gap_cases()
    result = dict(training_episodes=len(training),
                  visible_cases=len(visible), gap_cases=len(gaps),
                  visible=evaluate_visible(models, visible),
                  gaps=evaluate_gaps(models, gaps),
                  transition_weight_sum={name: float(model.weights.sum())
                                         for name, model in models.items()},
                  training_seconds=train_seconds,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        visible={name: round(row['f1'], 3)
                 for name, row in result['visible'].items()},
        last_blank={window: {name: round(row['f1'], 3)
                             for name, row in by_frame[str(len(by_frame))].items()}
                    for window, by_frame in result['gaps']['occupancy'].items()},
        motion={window: round(arms['learned']['direction_accuracy'], 3)
                for window, arms in result['gaps']['motion'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

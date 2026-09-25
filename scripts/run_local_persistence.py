"""Bounded intrinsic local hidden-state memory alongside learned motion."""

import copy
import json
import math
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases, scene_sequence
from gap_timing_transfer import MODEL_OUT
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_transition)
from run_local_visibility_likelihood import clean_training_cases
from run_native_trace_support import trace_correlation_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-local-persistence-results.json')
NAMES = ('fast', 'local_hold', 'persistence')


@torch.no_grad()
def score_visible(models, cases, coder, *, active_only, score_until):
    scores = {name: empty_score() for name in NAMES}
    for events in cases:
        codes = coder(events)
        for model in models.values():
            model.reset_state()
        for t, code in enumerate(codes[:score_until]):
            predictions = {name: model.pending for name, model in
                           models.items()}
            predictions['persistence'] = models['fast'].observed.clone()
            for model in models.values():
                model.step(code)
            if t < 4 or (active_only and models['fast'].observed.sum() == 0):
                continue
            target = models['fast'].observed
            for name, prediction in predictions.items():
                add_score(scores[name], prediction, target)
    return {name: finish(row) for name, row in scores.items()}


@torch.no_grad()
def score_gaps(models, cases):
    scores = {str(i): {name: empty_score()
                       for name in ('fast', 'local_hold', 'pre_gap_hold')}
              for i in range(1, 4)}
    reference = copy.deepcopy(models['fast'].encoder)
    for family, meta, events in cases:
        if family != 'occlusion':
            continue
        speed = meta['speed']
        obj = dict(shape=meta['shape'], center=(16, 32),
                   before=(0, speed), after=(0, speed))
        visible = scene_sequence([obj], background=meta['background'])
        codes = correlation_sequence(events)
        reference_codes = correlation_sequence(visible)
        reference.reset_state()
        for model in models.values():
            model.reset_state()
        for t in range(10):
            reference.step(reference_codes[t])
            for model in models.values():
                model.step(codes[t])
            if t == 6:
                held = models['fast'].state.clone()
            if t in (7, 8, 9):
                frame = str(t-6)
                for name, model in models.items():
                    add_score(scores[frame][name], model.state,
                              reference.latent)
                add_score(scores[frame]['pre_gap_hold'], held,
                          reference.latent)
    return {frame: {name: finish(row) for name, row in arms.items()}
            for frame, arms in scores.items()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    fast = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])['learned']
    local_hold = copy.deepcopy(fast)
    local_hold.persistence = math.exp(-1/4)
    models = dict(fast=fast, local_hold=local_hold)
    robust = make_robust_cases()
    native_cases = [pong_events(seed, stride=1, frames=120)
                    for seed in SEEDS]
    result = dict(training_episodes=len(clean), seeds=list(SEEDS),
                  native=score_visible(models, native_cases,
                                       trace_correlation_sequence,
                                       active_only=True, score_until=120),
                  generic={family: score_visible(
                      models, [events for label, _, events in robust
                               if label == family], correlation_sequence,
                      active_only=False, score_until=14)
                           for family in ('independent', 'crossing')},
                  occlusion_hidden=score_gaps(models, robust),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        native={name: (round(row['f1'], 3),
                       round(row['active_ratio'], 3))
                for name, row in result['native'].items()},
        generic={family: {name: (round(row['f1'], 3),
                                round(row['active_ratio'], 3))
                          for name, row in arms.items()}
                 for family, arms in result['generic'].items()},
        gap={frame: {name: (round(row['f1'], 3),
                            round(row['active_ratio'], 3))
                    for name, row in arms.items()}
             for frame, arms in result['occlusion_hidden'].items()},
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

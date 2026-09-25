"""Frozen hidden-state forecast indexed by local sensory-code events."""

import json
import time
from pathlib import Path

import torch

from gap_timing_transfer import MODEL_OUT
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_transition)
from run_local_visibility_likelihood import clean_training_cases
from run_native_trace_support import trace_correlation_sequence
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-event-clock-transfer-results.json')


def advance(model, code, *, clock):
    return model.step(code, advance=(clock == 'frame' or bool(code.sum() > 0)))


@torch.no_grad()
def evaluate(model, *, seeds=SEEDS, frames=120):
    scores = {}
    intervals = {}
    for clock in ('frame', 'event'):
        row = empty_score()
        predicted_sites = 0
        for seed in seeds:
            events = pong_events(seed, stride=1, frames=frames)
            codes = trace_correlation_sequence(events)
            model.reset_state()
            last_t = None
            for t, code in enumerate(codes):
                active = bool(code.sum() > 0)
                prediction = (model.pending if model.pending is not None
                              else torch.zeros_like(model.observed))
                advance(model, code, clock=clock)
                if active:
                    if t >= 4:
                        add_score(row, prediction, model.observed)
                        predicted_sites += int((prediction.sum(0) >= .5).sum())
                        if clock == 'event' and last_t is not None:
                            gap = str(t-last_t)
                            intervals[gap] = intervals.get(gap, 0)+1
                    last_t = t
        scores[clock] = dict(learned=finish(row),
                             predicted_sites=predicted_sites)
        if clock == 'frame':
            # The persistence control is scored on the same event times.
            persistence_score = empty_score()
            for seed in seeds:
                events = pong_events(seed, stride=1, frames=frames)
                codes = trace_correlation_sequence(events)
                model.reset_state()
                previous_event = torch.zeros_like(model.observed)
                for t, code in enumerate(codes):
                    advance(model, code, clock='frame')
                    if code.sum() > 0:
                        if t >= 4:
                            add_score(persistence_score, previous_event,
                                      model.observed)
                        previous_event = model.observed.clone()
            scores['persistence'] = finish(persistence_score)
    assert scores['frame']['learned']['target_sites'] == (
        scores['event']['learned']['target_sites'])
    return dict(scores=scores, physical_frame_intervals=intervals)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    trained = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])
    result = dict(training_episodes=len(clean), seeds=list(SEEDS),
                  frames_per_seed=120,
                  evaluation=evaluate(trained['learned']),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    scores = result['evaluation']['scores']
    print(json.dumps(dict(
        event_f1=round(scores['event']['learned']['f1'], 3),
        frame_f1=round(scores['frame']['learned']['f1'], 3),
        persistence_f1=round(scores['persistence']['f1'], 3),
        target_sites=scores['event']['learned']['target_sites'],
        intervals=result['evaluation']['physical_frame_intervals'],
        elapsed_seconds=round(result['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()

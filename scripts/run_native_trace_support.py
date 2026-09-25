"""Frozen visual model with longer local event coincidence at native timing."""

import json
import math
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from gap_timing_transfer import MODEL_OUT
from generic_local_transition import OFFSETS, shift
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_transition)
from run_local_visibility_likelihood import clean_training_cases
from run_pong_camera_transfer import SEEDS, pong_events


OUT = Path('docs/experiments/2026-09-25-native-trace-support-results.json')
DECAY = math.exp(-1/4)


def trace_correlation_sequence(events):
    trace = torch.zeros_like(events[0])
    codes = []
    for current in events:
        channels = torch.stack([shift(trace, dy, dx)*current
                                for dy, dx in OFFSETS])
        codes.append(channels.permute(1, 0, 2, 3).reshape(16, 32, 64))
        trace = trace*DECAY+current
    return codes


@torch.no_grad()
def evaluate(models, *, seeds=SEEDS, frames=120):
    scores = {}
    for arm, coder in (('adjacent', correlation_sequence),
                       ('trace_4', trace_correlation_sequence)):
        all_scores = {name: empty_score()
                      for name in ('learned', 'persistence')}
        active_scores = {name: empty_score()
                         for name in ('learned', 'persistence')}
        counts = dict(frames=0, camera_active=0, correlation_active=0,
                      latent_active=0, event_pixels=0,
                      quiet_frames=0, quiet_hidden_sites=0)
        for seed in seeds:
            events = pong_events(seed, stride=1, frames=frames)
            codes = coder(events)
            for model in models.values():
                model.reset_state()
            for t, (event, code) in enumerate(zip(events, codes)):
                prediction = models['learned'].pending
                persistence = models['learned'].observed.clone()
                for model in models.values():
                    model.step(code)
                if t < 4:
                    continue
                target = models['learned'].observed
                counts['frames'] += 1
                counts['camera_active'] += int(event.sum() > 0)
                counts['correlation_active'] += int(code.sum() > 0)
                counts['latent_active'] += int(target.sum() > 0)
                counts['event_pixels'] += int(event.sum())
                if event.sum() == 0:
                    counts['quiet_frames'] += 1
                    counts['quiet_hidden_sites'] += int((
                        models['learned'].state.sum(0) >= .5).sum())
                for name, proposed in (('learned', prediction),
                                       ('persistence', persistence)):
                    add_score(all_scores[name], proposed, target)
                    if target.sum() > 0:
                        add_score(active_scores[name], proposed, target)
        scores[arm] = dict(
            counts={**counts,
                    'correlation_active_fraction': counts[
                        'correlation_active']/counts['frames'],
                    'latent_active_fraction': counts['latent_active']/counts['frames'],
                    'hidden_sites_per_quiet_frame': counts[
                        'quiet_hidden_sites']/max(counts['quiet_frames'], 1)},
            all={name: finish(row) for name, row in all_scores.items()},
            active={name: finish(row) for name, row in active_scores.items()})
    return scores


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
                  scores=evaluate({'learned': trained['learned']}),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({arm: dict(
        correlation_fraction=round(rows['counts'][
            'correlation_active_fraction'], 3),
        active_f1={name: round(score['f1'], 3)
                   for name, score in rows['active'].items()},
        target_sites=rows['active']['learned']['target_sites'],
        quiet_hidden=round(rows['counts']['hidden_sites_per_quiet_frame'], 3))
        for arm, rows in result['scores'].items()}), flush=True)


if __name__ == '__main__':
    main()

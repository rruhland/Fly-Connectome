"""Zero-shot latent motion transfer to native and coarser Pong camera streams."""

import json
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence
from fly_connectome.pong import Pong
from fly_connectome.sensor import EventCamera
from gap_timing_transfer import MODEL_OUT
from generic_motion_probe import events_map
from run_local_hidden_transition import (add_score, empty_score, finish,
                                         train as train_transition)
from run_local_visibility_likelihood import clean_training_cases


OUT = Path('docs/experiments/2026-09-25-pong-camera-transfer-results.json')
ARMS = ('learned', 'shuffled_credit', 'frozen', 'persistence')
SEEDS = (1101, 1102, 1103, 1104)


@torch.no_grad()
def pong_events(seed, *, stride, frames):
    pong = Pong([seed])
    camera = EventCamera(1, 32, 64)
    sequence = []
    for _ in range(frames):
        sequence.append(events_map(camera.observe(pong.render())))
        for _ in range(stride):
            pong.step(torch.zeros(1))
    return sequence


@torch.no_grad()
def evaluate(models, *, seeds=SEEDS, frames=120):
    results = {}
    for stride in (1, 4):
        all_scores = {name: empty_score() for name in ARMS}
        active_scores = {name: empty_score() for name in ARMS}
        counts = dict(frames=0, camera_active=0, correlation_active=0,
                      latent_active=0, event_pixels=0,
                      quiet_frames=0, quiet_hidden_sites=0)
        for seed in seeds:
            events = pong_events(seed, stride=stride, frames=frames)
            codes = correlation_sequence(events)
            for model in models.values():
                model.reset_state()
            for t, (event, code) in enumerate(zip(events, codes)):
                predictions = {name: model.pending for name, model in
                               models.items()}
                predictions['persistence'] = models[
                    'learned'].observed.clone()
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
                for name, prediction in predictions.items():
                    add_score(all_scores[name], prediction, target)
                    if target.sum() > 0:
                        add_score(active_scores[name], prediction, target)
        results[str(stride)] = dict(
            counts={**counts,
                    'camera_active_fraction': counts['camera_active']/counts['frames'],
                    'correlation_active_fraction': counts[
                        'correlation_active']/counts['frames'],
                    'latent_active_fraction': counts['latent_active']/counts['frames'],
                    'event_pixels_per_frame': counts['event_pixels']/counts['frames'],
                    'hidden_sites_per_quiet_frame': counts[
                        'quiet_hidden_sites']/max(counts['quiet_frames'], 1)},
            all={name: finish(row) for name, row in all_scores.items()},
            active={name: finish(row) for name, row in active_scores.items()})
    return results


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    saved = torch.load(MODEL_OUT, map_location='cpu', weights_only=False)
    clean = clean_training_cases()
    trained = train_transition(
        saved['models']['learned_split'].code.motion,
        [(kind, '', events) for kind, events in clean])
    models = {name: trained[name] for name in ARMS[:3]}
    result = dict(training_episodes=len(clean), seeds=list(SEEDS),
                  frames_per_seed=120, scores=evaluate(models),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({stride: dict(
        counts=rows['counts'],
        active_f1={name: round(score['f1'], 3)
                   for name, score in rows['active'].items()})
        for stride, rows in result['scores'].items()}), flush=True)


if __name__ == '__main__':
    main()

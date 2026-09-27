"""Train, checkpoint and test the joint streaming candidate without promotion."""

import copy
import json
import math
import time
from pathlib import Path

import torch

from appearance_relation import appearance_case
from event_only_observer_teacher import fit_event_only_observer
from generic_native_cadence import scene_events
from local_metric_association import LocalMetricAssociation
from local_observation_model import LocalObservationModel
from run_associative_patch_state import rotate_case
from run_joint_context_acceptance import fit_joint_memory
from run_local_motion_dynamics import KINDS, visual_episode
from run_visual_context_controls import context_cases
from run_zero_shot_visual_transfer import corrupt_events
from streaming_visual_state import LearnedVisualCandidate, StreamingVisualState
from visual_history_scoring import hidden_rank


def observer_at(base, height, width):
    observer = LocalObservationModel(height=height, width=width)
    observer.weights.copy_(base.weights)
    observer.bias.copy_(base.bias)
    return observer


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    base = fit_event_only_observer([corrupt_events(scene_events(seed), seed=10000+seed)
                                   for seed in range(64)])
    memory = fit_joint_memory(base, memory_type=LocalMetricAssociation)
    dynamic_model = LearnedVisualCandidate(height=64, width=64, observer=observer_at(base, 64, 64))
    for kind in KINDS:
        for phase in (0, 2):
            for shape in ('dot', 'square'):
                case, _ = visual_episode(kind, phase, shape, background=bool(phase))
                dynamic_model.reset_state()
                for event, visible in zip(case['events'], case['visible']):
                    dynamic_model.step(event, visible, learn=True)
    result = dict(sensor='current visible grayscale frames plus events', context={}, dynamics={},
                   production_promoted=False)
    for name, transform in (('spatial', lambda c: c), ('appearance', appearance_case)):
        result['context'][name] = {}
        for turns in range(4):
            height, width = (64, 32) if turns % 2 else (32, 64)
            model = LearnedVisualCandidate(height=height, width=width,
                observer=observer_at(base, height, width), memory=memory,
                dynamics=dynamic_model.dynamics)
            hits = []
            for original in context_cases():
                case = rotate_case(transform(original), turns)
                model.reset_state()
                for t in range(case['decision']+1):
                    state = model.step(case['events'][t], case['visible'][t])
                hits.append(hidden_rank(state['support_field'], case['hidden'][t], k=32))
            result['context'][name][str(turns*90)] = dict(hits=sum(hits), total=len(hits))
    model = LearnedVisualCandidate(height=64, width=64, observer=observer_at(base, 64, 64),
                                    memory=memory, dynamics=dynamic_model.dynamics)
    model.calibration = copy.deepcopy(dynamic_model.calibration)
    runtime, frames = 0., 0
    for kind in KINDS:
        errors = {h: [] for h in (1, 4, 8)}
        for phase in (3, 5):
            for angle in (0., math.pi/4, math.pi/2):
                case, truth = visual_episode(kind, phase, 'plus', angle=angle, scale=1.25,
                                             background=bool(phase % 2))
                model.reset_state()
                for t, (event, visible) in enumerate(zip(case['events'], case['visible'])):
                    before = time.perf_counter()
                    state = model.step(event, visible)
                    runtime += time.perf_counter()-before
                    frames += 1
                    if t < 6:
                        continue
                    for h in errors:
                        if t+h >= len(truth):
                            continue
                        forecasts = [f for f in state['forecasts'] if f['horizon_samples'] == h]
                        forecast = min(forecasts, key=lambda f: f['id']) if forecasts else None
                        errors[h].append(64. if forecast is None else
                            float((forecast['position']-truth[t+h]).norm()))
        result['dynamics'][kind] = {h: sum(v)/len(v) for h, v in errors.items()}
    result['forecast_fps'] = frames/runtime
    result['checkpoint'] = 'checkpoints/m1a5/streaming-candidate.pt'
    checkpoint = Path(result['checkpoint'])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.reset_state()
    model.save(checkpoint)
    loaded = StreamingVisualState.load(checkpoint)
    assert type(loaded) is LearnedVisualCandidate
    assert all(torch.equal(loaded.dynamics[h].values, model.dynamics[h].values) for h in model.dynamics)
    assert all(loaded.calibration[h].summary() == model.calibration[h].summary() for h in model.calibration)
    result['calibration_samples'] = {h: len(c.outcomes) for h, c in loaded.calibration.items()}
    result['checkpoint_verified'] = True
    result['seconds'] = time.perf_counter()-start
    Path('docs/experiments/2026-09-26-integrated-visual-candidate-results.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

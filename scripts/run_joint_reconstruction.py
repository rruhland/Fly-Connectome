"""Bounded comparison of learned joint event and sparse-image reconstruction."""

import copy
import json
import time
from functools import partial
from pathlib import Path

import torch

from evidence_state import EvidenceContextMemory
from generic_motion_probe import local_motion_map
from generic_native_cadence import scene_events
from joint_reconstruction import SparseFrameObserver, TemporalPatchObserver
from run_local_context_memory import score, score_disappearance, train
from run_visual_context_controls import context_cases
from run_visual_history_controls import cases, two_mover_cases
from run_zero_shot_visual_transfer import corrupt_events


OUT = Path('docs/experiments/2026-09-26-joint-reconstruction-results.json')


@torch.no_grad()
def reconstruction_sequence(observer, case, *, hybrid=False, image_noise=0.):
    observer.reset_state()
    trace = torch.zeros_like(case['events'][0])
    generator = torch.Generator().manual_seed(58000)
    states = []
    for frame, event in enumerate(case['events']):
        absolute = None
        if isinstance(observer, SparseFrameObserver) and frame % 8 == 0:
            visible = case['visible'][frame].float()
            flips = torch.rand(visible.shape, generator=generator) < image_noise
            visible = torch.where(flips, 1-visible, visible)
            absolute = visible-visible.median()
        filtered, _ = observer.step(event, absolute=absolute)
        motion = local_motion_map(trace, filtered).clamp(-1., 1.)
        trace.mul_(.875).add_(filtered)
        contrast = observer.contrast
        states.append(torch.cat((filtered, observer.fast.clone(),
                                 observer.slow.clone(),
                                 contrast.clamp(min=0.)[None],
                                 (-contrast).clamp(min=0.)[None],
                                 motion, trace.clone())))
    return states


@torch.no_grad()
def fit_observation(model):
    frozen = None
    for seed in range(64):
        events, images = scene_events(seed, return_contrast=True,
                                      motion_stride=(1, 4, 8)[seed % 3])
        noisy = corrupt_events(events, seed=10000+seed)
        model.reset_state()
        for frame, event in enumerate(noisy):
            absolute = None
            if isinstance(model, SparseFrameObserver) and frame % 8 == 0:
                absolute = images[frame]-images[frame].median()
            model.step(event, absolute=absolute, learn=True)
        # Data-initialized rather than zero-dictionary frozen control.
        if seed == 0:
            frozen = copy.deepcopy(model)
    return model, frozen


@torch.no_grad()
def evaluate(model, observer, *, image_noise=0.):
    sequence = partial(reconstruction_sequence, image_noise=image_noise)
    kwargs = dict(hybrid=False, sequence=sequence)
    context = context_cases()
    return dict(effects=model.effects.tolist(), counts=model.counts.tolist(),
                clean=score(model, observer, context, **kwargs),
                known_noise=score(model, observer, context, noise=(.1, .001),
                                  noise_seed=50000, **kwargs),
                heavy_noise=score(model, observer, context, noise=(.2, .002),
                                  noise_seed=50000, **kwargs),
                constant_motion=score(model, observer, cases(heldout=True), **kwargs),
                two_movers=score(model, observer, two_mover_cases(), multi=True, **kwargs),
                disappearance=score_disappearance(model, observer, context, **kwargs))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    results = dict(noise_seed_base=50000, arms={})
    for name, model in (('joint_events', TemporalPatchObserver()),
                        ('sparse_frames', SparseFrameObserver())):
        learned, frozen = fit_observation(model)
        arm = {}
        variants = [('learned', learned, False), ('frozen', frozen, False),
                    ('shuffled_context', learned, True)]
        if name == 'sparse_frames':
            variants.append(('noisy_images', learned, False))
        for label, observer, shuffled in variants:
            state = EvidenceContextMemory(representation='surface')
            train(state, observer, shuffled=shuffled,
                  sequence=reconstruction_sequence)
            arm[label] = evaluate(state, observer,
                                  image_noise=.01 if label == 'noisy_images' else 0.)
            results['arms'][name] = arm
            results['seconds'] = round(time.perf_counter()-started, 2)
            OUT.write_text(json.dumps(results, indent=2)+'\n')
            print(name, label, json.dumps(arm[label]), flush=True)


if __name__ == '__main__':
    main()

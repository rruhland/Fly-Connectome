"""Broader causal visual experience with fixed local learning and memory budget."""

import copy
import json
import math
import time
from pathlib import Path

import torch

from run_fresh_candidate_acceptance import main as accept
from run_local_motion_dynamics import KINDS, visual_episode
from run_zero_shot_visual_transfer import corrupt_events
from streaming_visual_state import LearnedVisualCandidate, StreamingVisualState


@torch.no_grad()
def train(learner):
    generator = torch.Generator().manual_seed(141001)
    start, frames = time.perf_counter(), 0
    for trial in range(32):
        for index, kind in enumerate(KINDS):
            seed = 142000+trial*len(KINDS)+index
            angle = float(torch.rand((), generator=generator))*2*math.pi
            scale = .6+float(torch.rand((), generator=generator))
            phase = int(torch.randint(8, (), generator=generator))
            case, _ = visual_episode(kind, phase, 'dot' if trial % 2 else 'square',
                angle=angle, scale=scale, background=bool(trial % 2))
            events = corrupt_events(case['events'], seed=seed, dropout=.15, false_rate=.0005)
            noise = torch.Generator().manual_seed(seed+1000)
            learner.reset_state()
            for t, (event, binary) in enumerate(zip(events, case['visible'])):
                image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                         .02*torch.randn(binary.shape, generator=noise)).clamp(0., 1.)
                learner.step(event, image, learn=True)
                frames += 1
    seconds = time.perf_counter()-start
    return dict(training_streams=192, frames=frames, seconds=seconds,
        training_fps=frames/seconds, prototypes={h: len(m.keys) for h, m in learner.dynamics.items()})


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    learner = LearnedVisualCandidate(height=64, width=64, observer=copy.deepcopy(source.observer))
    summary = train(learner)
    source.dynamics = learner.dynamics
    source.calibration = copy.deepcopy(learner.calibration)
    source.reset_state()
    checkpoint = 'checkpoints/m1a5/broad-experience-candidate.pt'
    source.save(checkpoint)
    summary['checkpoint'] = checkpoint
    Path('docs/experiments/2026-09-26-broad-online-training-results.json').write_text(
        json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    accept(checkpoint=checkpoint,
        out=Path('docs/experiments/2026-09-26-broad-online-acceptance-results.json'),
        phases=(11, 13), angles=(math.pi/8, 3*math.pi/8, 5*math.pi/8), scales=(.85, 1.35))


if __name__ == '__main__':
    main()

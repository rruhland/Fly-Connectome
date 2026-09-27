"""One causal competence-learning experiment over fixed generic alternatives."""

import copy
import json
import math
from pathlib import Path

import torch

from contextual_motion_dynamics import ContextualMotionDynamics
from forecast_competence import ForecastCompetence
from local_metric_association import LocalMetricAssociation
from run_fresh_candidate_acceptance import main as accept
from run_local_motion_dynamics import KINDS, visual_episode
from run_zero_shot_visual_transfer import corrupt_events
from statistical_visual_state import StatisticalVisualCandidate
from streaming_visual_state import StreamingVisualState


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    data = torch.load('checkpoints/m1a5/StatisticalVisualCandidate-ContextualMotionDynamics.pt', weights_only=True)
    dynamics = {}
    for h, values in data['dynamics'].items():
        base = ContextualMotionDynamics()
        for name, value in values.items():
            setattr(base, name, value)
        dynamics[h] = ForecastCompetence(base, h)
    learner = StatisticalVisualCandidate(height=64, width=64, observer=copy.deepcopy(source.observer),
                                         dynamics=dynamics)
    generator = torch.Generator().manual_seed(171000)
    for trial in range(64):
        kind = KINDS[trial % len(KINDS)]
        angle = float(torch.rand((), generator=generator))*2*math.pi
        scale = .6+float(torch.rand((), generator=generator))
        phase = int(torch.randint(8, (), generator=generator))
        case, _ = visual_episode(kind, phase, 'dot' if trial % 2 else 'square',
                                 angle=angle, scale=scale, background=bool(trial % 2))
        events = corrupt_events(case['events'], seed=172000+trial, dropout=.15, false_rate=.0005)
        noise = torch.Generator().manual_seed(173000+trial)
        learner.reset_state()
        for t, (event, binary) in enumerate(zip(events, case['visible'])):
            image = (.25+.05*math.sin(t/9)+.35*binary.float()+
                     .02*torch.randn(binary.shape, generator=noise)).clamp(0., 1.)
            learner.step(event, image, learn=True)
    torch.save(dict(dynamics={h: dict(base=vars(m.base), risk=dict(keys=m.risk.keys,
                    values=m.risk.values, metric=m.risk.metric)) for h, m in dynamics.items()},
                    calibration={h: dict(outcomes=list(c.outcomes), residuals=list(c.residuals))
                                 for h, c in learner.calibration.items()}),
               'checkpoints/m1a5/forecast-competence-experiment.pt')
    for arm in ('learned', 'shuffled', 'no_gate'):
        models = copy.deepcopy(dynamics)
        for h, model in models.items():
            if arm == 'shuffled':
                permutation = torch.randperm(len(model.risk.values), generator=torch.Generator().manual_seed(174000+h))
                model.risk.values = model.risk.values[permutation]
            elif arm == 'no_gate':
                model.risk = LocalMetricAssociation(dimensions=8, outputs=4)
        candidate = StatisticalVisualCandidate(height=64, width=64, observer=copy.deepcopy(source.observer),
            memory=copy.deepcopy(source.state.memory), dynamics=models)
        candidate.calibration = copy.deepcopy(learner.calibration)
        print('evaluating', arm, flush=True)
        accept(candidate=candidate,
            out=Path(f'docs/experiments/2026-09-26-competence-{arm}-results.json'),
            phases=(23, 29), angles=(math.pi/10, 3*math.pi/10, 7*math.pi/10), scales=(.95, 1.45))


if __name__ == '__main__':
    main()

"""One registered 2x2 observation/dynamics architecture comparison."""

import copy
import json
import math
from pathlib import Path

import torch

from contextual_motion_dynamics import ContextualMotionDynamics
from local_generative_dynamics import LocalGenerativeDynamics
from run_broad_online_dynamics import train
from run_fresh_candidate_acceptance import main as accept
from statistical_visual_state import StatisticalVisualCandidate
from streaming_visual_state import LearnedVisualCandidate, StreamingVisualState


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    summaries = {}
    for state_type in (LearnedVisualCandidate, StatisticalVisualCandidate):
        for dynamics_type in (ContextualMotionDynamics, LocalGenerativeDynamics):
            name = state_type.__name__+'-'+dynamics_type.__name__
            learner = state_type(height=64, width=64, observer=copy.deepcopy(source.observer),
                                  dynamics={h: dynamics_type() for h in (1, 4, 8)})
            summaries[name] = train(learner)
            candidate = state_type(height=64, width=64, observer=copy.deepcopy(source.observer),
                                    memory=copy.deepcopy(source.state.memory), dynamics=learner.dynamics)
            candidate.calibration = copy.deepcopy(learner.calibration)
            torch.save(dict(state_type=state_type.__name__, dynamics_type=dynamics_type.__name__,
                dynamics={h: vars(m) for h, m in learner.dynamics.items()},
                calibration={h: dict(outcomes=list(c.outcomes), residuals=list(c.residuals))
                             for h, c in learner.calibration.items()}),
                Path('checkpoints/m1a5')/(name+'.pt'))
            Path('docs/experiments/2026-09-26-generative-factorial-training-results.json').write_text(
                json.dumps(summaries, indent=2)+'\n')
            print(name, summaries[name], flush=True)
            accept(candidate=candidate,
                out=Path(f'docs/experiments/2026-09-26-factorial-{name}-results.json'),
                phases=(17, 19), angles=(math.pi/12, 5*math.pi/12, 3*math.pi/4), scales=(.9, 1.4))


if __name__ == '__main__':
    main()

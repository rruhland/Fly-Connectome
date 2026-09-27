"""One entity-local online recurrence versus its frozen prior and fixed controls."""

import copy
import math
from functools import partial
from pathlib import Path

import torch

from online_entity_dynamics import OnlineEntityDynamics
from run_fresh_candidate_acceptance import main as accept
from statistical_visual_state import StatisticalVisualCandidate
from streaming_visual_state import StreamingVisualState


def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    for learn in (True, False):
        candidate = StatisticalVisualCandidate(height=64, width=64,
            observer=copy.deepcopy(source.observer), memory=copy.deepcopy(source.state.memory),
            dynamics=copy.deepcopy(source.dynamics))
        accept(candidate=candidate, entity_dynamics=partial(OnlineEntityDynamics, learn=learn),
            out=Path(f'docs/experiments/2026-09-26-online-entity-{"learned" if learn else "frozen"}-results.json'),
            phases=(23, 29), angles=(math.pi/10, 3*math.pi/10, 7*math.pi/10), scales=(.95, 1.45))


if __name__ == '__main__':
    main()

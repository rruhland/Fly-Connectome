"""Experience controls have equal visual exposure and different tempo."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_multicadence_experience import training_sequences


def test_repeated_and_multicadence_training_have_equal_episode_budget():
    repeated, diverse = training_sequences(seeds=(3, 7), strides=(1, 4),
                                           frames=12)
    assert len(repeated) == len(diverse) == 4
    assert all(len(events) == 12 for events in repeated+diverse)
    assert torch.equal(repeated[0][0], diverse[0][0])
    assert any(not torch.equal(a, b) for a, b in zip(
        repeated[1], diverse[1]))

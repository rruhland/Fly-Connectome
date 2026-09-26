"""Contrast uncertainty follows causal signed transitions without mass loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contrast_belief import ContrastBelief


def test_certain_polarity_pair_returns_to_baseline():
    belief = ContrastBelief(height=1, width=1)
    belief.step(torch.tensor([[[0.]], [[1.]]]))
    assert belief.mean.item() == 1.
    belief.step(torch.tensor([[[1.]], [[0.]]]))
    assert belief.probabilities[:, 0, 0].tolist() == [0., 1., 0.]


def test_uncertain_event_retains_alternatives_and_empty_input_preserves_them():
    belief = ContrastBelief(height=1, width=1)
    belief.step(torch.tensor([[[0.]], [[.25]]]))
    assert belief.probabilities[:, 0, 0].tolist() == [0., .75, .25]
    belief.step(torch.zeros(2, 1, 1))
    assert belief.probabilities[:, 0, 0].tolist() == [0., .75, .25]


def test_conflicting_evidence_is_symmetric_and_mass_is_conserved():
    belief = ContrastBelief(height=4, width=4)
    belief.step(torch.full((2, 4, 4), .6))
    assert torch.allclose(belief.mean, torch.zeros(4, 4))
    generator = torch.Generator().manual_seed(20)
    for _ in range(50):
        belief.step(torch.rand(2, 4, 4, generator=generator))
    assert (belief.probabilities >= 0).all()
    assert torch.allclose(belief.probabilities.sum(0), torch.ones(4, 4))

"""Behavioral checks for the evaluation-only local linear capacity probe."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_code_capacity import scatter_local_decoder


def test_local_decoder_scatter_uses_existing_source_neighborhood():
    weights = torch.zeros((2, 2*25))
    weights[0, 2*5+3] = .7
    output, reachable = scatter_local_decoder(weights, [(16, 15, 0)], units=2)
    assert output[0, 16, 16] == torch.tensor(.7)
    assert output.sum() == torch.tensor(.7)
    assert reachable[16, 16]
    assert not reachable[16, 18]

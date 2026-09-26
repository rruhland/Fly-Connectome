"""Fixed post-inference spatial probe preserves relative motion geometry."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_spatial_state_audit import spatial_feature


def test_relative_spatial_feature_translates_with_evaluation_center():
    first = torch.zeros((1, 4, 4))
    second = torch.zeros_like(first)
    first[0, 1, 2] = 1.
    second[0, 2, 3] = 1.
    assert torch.equal(spatial_feature(first, (4, 4)),
                       spatial_feature(second, (8, 8)))
    assert spatial_feature(first, (4, 4)).reshape(1, 3, 3)[0, 1, 2] == 1


def test_fast_state_uses_same_four_pixel_evaluation_grid():
    fast = torch.zeros((1, 16, 16))
    fast[0, 4, 8] = 1.
    slow = torch.zeros((1, 4, 4))
    slow[0, 1, 2] = 1.
    assert torch.equal(spatial_feature(fast, (4, 4), is_fast=True),
                       spatial_feature(slow, (4, 4)))

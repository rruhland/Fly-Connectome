"""Evaluation boundaries for the two-timescale M1A experiment."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_two_timescale_assemblies import component_count


def test_component_count_separates_distant_assembly_regions():
    active = torch.zeros((5, 6), dtype=torch.bool)
    active[1, 1:3] = True
    active[3, 4] = True
    assert component_count(active) == 2
    assert component_count(torch.zeros_like(active)) == 0

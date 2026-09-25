"""Frozen capacity features retain local offset and unit identity."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from blank_trace_capacity import local_matrix, target_coverage


def test_local_trace_matrix_preserves_unit_offset_and_amplitude():
    trace = torch.zeros((24, 32, 64))
    trace[2, 16, 32] = .5
    matrix = local_matrix([trace], radius=2)
    assert matrix.shape == (2048, 24*25)
    assert matrix[16*64+33, 2*25+2*5+3] == .5
    assert matrix[16*64+32, 2*25+2*5+2] == .5
    assert matrix[16*64+38].nnz == 0


def test_trace_reachability_expands_with_local_field():
    trace = torch.zeros((24, 32, 64))
    trace[0, 16, 32] = 1
    target = torch.zeros((2, 32, 64))
    target[0, 16, 38] = 1
    coverage = target_coverage([trace], [target])
    assert coverage[2]['fraction'] == 0
    assert coverage[4]['fraction'] == 0
    assert coverage[8]['fraction'] == 1

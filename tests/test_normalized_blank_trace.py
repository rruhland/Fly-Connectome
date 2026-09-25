"""Site-local trace normalization preserves support and relative activity."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from normalized_blank_trace import normalize_trace_sites


def test_normalization_is_local_and_invariant_to_site_amplitude():
    trace = torch.zeros((24, 32, 64))
    trace[2, 10, 11] = .2
    trace[3, 10, 11] = .6
    trace[2, 12, 14] = .1
    trace[3, 12, 14] = .3
    normalized = normalize_trace_sites(trace)
    assert torch.allclose(normalized[:, 10, 11],
                          normalized[:, 12, 14])
    assert torch.allclose(normalized[:, 10, 11].sum(), torch.tensor(1.))
    assert normalized[:, 0, 0].sum() == 0
    assert torch.equal(normalized > 0, trace > 0)

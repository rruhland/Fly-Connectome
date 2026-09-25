"""Local competition and continuity in the opt-in visual-state probe."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sparse_coherent_state import SparseCoherentState


def site(unit, y, x, amplitude=1.):
    state = torch.zeros((2, 8, 8))
    state[unit, y, x] = amplitude
    return state


def test_nearby_sites_compete_but_separated_sites_survive():
    state = site(0, 2, 2, .8) + site(1, 2, 3) + site(0, 6, 6)
    probe = SparseCoherentState(decay=0.)
    output = probe.step(state)
    assert output[1, 2, 3] == 1
    assert output[0, 2, 2] == 0
    assert output[0, 6, 6] == 1


def test_quiet_state_retains_then_expires_without_input():
    probe = SparseCoherentState(decay=.8)
    probe.step(site(0, 3, 3))
    empty = torch.zeros((2, 8, 8))
    assert probe.step(empty)[0, 3, 3] == .8
    assert probe.step(empty)[0, 3, 3] > .5
    assert probe.step(empty)[0, 3, 3] > .5
    assert probe.step(empty)[0, 3, 3] == 0


def test_new_evidence_replaces_weaker_retained_neighbor_and_reset_clears():
    probe = SparseCoherentState(decay=.9)
    probe.step(site(0, 3, 3))
    output = probe.step(site(1, 3, 4))
    assert output[1, 3, 4] == 1
    assert output[0, 3, 3] == 0
    probe.reset_state()
    assert not probe.step(torch.zeros((2, 8, 8))).any()


def test_spatial_shift_preserves_winners_away_from_edges():
    state = site(0, 2, 2, .8) + site(1, 2, 3) + site(0, 5, 5)
    a = SparseCoherentState(decay=0.).step(state)
    b = SparseCoherentState(decay=0.).step(torch.roll(state, (1, 1), (1, 2)))
    torch.testing.assert_close(torch.roll(a, (1, 1), (1, 2)), b)

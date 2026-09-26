"""Learned local co-occurrence binds signed visual fragments."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_affinity_binding import fit_diagonal_affinity
from persistent_entity_files import PersistentEntityFiles


def pair_event(y, x, *, height=12, width=20):
    event = torch.zeros((2, height, width))
    event[1, y, x] = 1.
    event[1, y+1, x+1] = 1.
    return event


def test_diagonal_proposal_needs_a_matching_signed_link():
    event = pair_event(5, 5)
    four = PersistentEntityFiles(height=12, width=20)
    four.step(event)
    assert len(four.live_slots) == 2

    linked = PersistentEntityFiles(
        height=12, width=20, diagonal_links={(1., 1): True})
    linked.step(event)
    assert len(linked.live_slots) == 1

    opposite = PersistentEntityFiles(
        height=12, width=20, diagonal_links={(-1., 1): True})
    opposite.step(event)
    assert len(opposite.live_slots) == 2


def test_aligned_local_error_learns_a_transferable_diagonal_link():
    streams = [[pair_event(2+(index*3) % 8, 2+(index*7) % 15)]
               for index in range(80)]
    links = fit_diagonal_affinity(streams)
    assert links['aligned'][(1., 1)]
    assert not links['shuffled'][(1., 1)]

    heldout = PersistentEntityFiles(
        height=12, width=20, diagonal_links=links['aligned'])
    heldout.step(pair_event(4, 9))
    assert len(heldout.live_slots) == 1


def test_opposite_sign_diagonal_events_do_not_bind():
    event = pair_event(5, 5)
    event[1, 6, 6] = 0.
    event[0, 6, 6] = 1.
    linked = PersistentEntityFiles(
        height=12, width=20, diagonal_links={(1., 1): True,
                                              (-1., 1): True})
    linked.step(event)
    assert len(linked.live_slots) == 2

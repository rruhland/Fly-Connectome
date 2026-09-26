"""Sparse local assemblies bind candidate evidence across event gaps."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_motion_assemblies import (LocalTransitionAffinity,
                                     MotionAssemblies, local_candidates)


def candidate(y, x, unit=0):
    feature = torch.zeros(4)
    feature[unit] = 1
    return (y, x, feature)


def test_adjacent_event_candidates_compete_but_distant_ones_survive():
    event = torch.zeros((2, 8, 8))
    observed = torch.zeros((2, 8, 8))
    event[0, 2, 2] = 1
    event[0, 2, 3] = 1
    event[1, 6, 6] = 1
    proposals = local_candidates(event, observed)
    assert len(proposals) == 2
    assert {(y, x) for y, x, _ in proposals} == {(2, 2), (6, 6)}


def test_local_hebbian_affinity_learns_repeated_displacement():
    learned = LocalTransitionAffinity(features=4, radius=2)
    for _ in range(12):
        learned.observe([candidate(3, 3)], [candidate(3, 4)])
    learned.finalize()
    source = candidate(3, 3)
    assert learned.score(source, candidate(3, 4)) > learned.score(
        source, candidate(3, 2))


def test_two_assemblies_persist_through_quiet_frames_without_merging():
    tracker = MotionAssemblies(radius=2, max_gap=4)
    tracker.step(0, [candidate(2, 2), candidate(6, 6)])
    tracker.step(1, [])
    tracker.step(2, [])
    tracker.step(3, [candidate(2, 3), candidate(6, 5)])
    histories = [row['history'] for row in tracker.tokens]
    assert len(histories) == 2
    assert sorted(len(history) for history in histories) == [2, 2]
    assert {(history[-1][1], history[-1][2]) for history in histories} == {
        (2, 3), (6, 5)}


def test_old_assembly_does_not_claim_new_distant_event():
    tracker = MotionAssemblies(radius=2, max_gap=2)
    tracker.step(0, [candidate(2, 2)])
    tracker.step(3, [candidate(2, 3)])
    assert len(tracker.tokens) == 2

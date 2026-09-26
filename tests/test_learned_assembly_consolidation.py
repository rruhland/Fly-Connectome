"""Local displacement learning and autonomous competing-path selection."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from learned_assembly_consolidation import (LocalDisplacementExpectation,
                                            consolidate_paths)


def candidate(y, x, polarity=0):
    feature = torch.zeros(4)
    feature[polarity] = .5
    feature[2] = .5
    return (y, x, feature)


def path(points, score):
    return dict(history=[(t, y, x) for t, (y, x) in enumerate(points)],
                score=score)


def test_hebbian_expectation_favors_experienced_displacement():
    learned = LocalDisplacementExpectation(features=4, radius=2)
    for _ in range(10):
        learned.observe([candidate(3, 3)], [candidate(3, 4)])
    learned.finalize()
    assert learned.score(candidate(3, 3), candidate(3, 4)) > 0
    assert learned.score(candidate(3, 3), candidate(3, 2)) < 0


def test_co_moving_local_paths_consolidate_but_opponents_survive():
    right = path([(4, 4), (4, 5), (4, 6)], 5)
    parallel = path([(4, 6), (4, 7), (4, 8)], 4)
    left = path([(4, 8), (4, 7), (4, 6)], 3)
    output = consolidate_paths([right, parallel, left])
    assert right in output
    assert parallel not in output
    assert left in output


def test_distant_parallel_paths_remain_distinct():
    first = path([(4, 4), (4, 5), (4, 6)], 5)
    second = path([(15, 15), (15, 16), (15, 17)], 4)
    assert len(consolidate_paths([first, second])) == 2

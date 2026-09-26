"""Learned and shuffled local readouts receive identical future events."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_rank_reliability import LocalRankReliability
from run_local_rank_reliability import score_learned


def test_learned_readouts_use_the_same_target_event():
    event = torch.zeros((2, 1, 4))
    fast = event.clone()
    files = event.clone()
    event[1, 0, 3] = 1.
    fast[1, 0, 0] = 1.
    files[1, 0, 3] = 1.
    aligned = LocalRankReliability()
    aligned.observe(fast, files, event)
    rows = score_learned([[(event, fast, files)]],
                         {'aligned': aligned, 'shuffled':
                          LocalRankReliability()})
    assert rows['aligned']['events'] == rows['shuffled']['events'] == 1
    assert rows['aligned']['targets'] == rows['shuffled']['targets'] == 1
    assert rows['aligned']['top_8_hits'] == 1
    assert rows['aligned']['positive_sites'] == 1

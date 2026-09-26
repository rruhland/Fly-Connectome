"""Zero-shot transfer uses aligned and unrelated visual streams fairly."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_observation_model_transfer import make_streams


def test_transfer_streams_keep_targets_and_shuffle_intensity_by_scene():
    first = [torch.zeros((2, 8, 10)) for _ in range(2)]
    second = [torch.zeros((2, 8, 10)) for _ in range(2)]
    first[0][1, 4, 4] = 1.
    second[0][1, 5, 7] = 1.
    streams = make_streams([first, second])
    assert streams[0][0] is first
    assert streams[0][1] is first
    assert streams[0][2] is second
    assert streams[0][3][0][4, 4] == 1
    assert streams[0][4][0][5, 7] == 1

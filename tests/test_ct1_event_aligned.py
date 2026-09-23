import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_ct1_event_aligned import aligned


def test_aligned_uses_each_cells_own_event_tick():
    values = torch.arange(20)[:, None]*10+torch.arange(3)[None, :]
    anchors = torch.tensor([5, 7, 9])
    cells = torch.tensor([0, 2])
    result = aligned(values, anchors, cells, torch.tensor([-1, 0, 1]))
    assert torch.equal(result, torch.tensor([[40, 50, 60],
                                             [82, 92, 102]]))

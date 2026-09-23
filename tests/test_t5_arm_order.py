import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_t5_arm_order import lag_order


def test_order_score_changes_sign_when_arm_order_reverses():
    early = torch.tensor([[0.], [1.], [0.], [0.]])
    late = torch.tensor([[0.], [0.], [1.], [0.]])
    assert torch.equal(lag_order(early, late, 1)[:, 0],
                       torch.tensor([0., 0., 1., 0.]))
    assert torch.equal(lag_order(late, early, 1)[:, 0],
                       torch.tensor([0., 0., -1., 0.]))

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from generic_motion_probe import local_motion_map


def test_local_motion_stays_at_the_current_event_location():
    past = torch.zeros(2, 5, 5)
    current = torch.zeros_like(past)
    past[0, 2, 1] = 1
    current[0, 2, 2] = 1
    motion = local_motion_map(past, current)
    assert motion.shape == (2, 5, 5)
    assert motion[0, 2, 2] > 0
    assert motion[0].sum() > 0 and motion[1].sum() == 0
    assert torch.count_nonzero(motion[:, :2]) == 0

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from t5_local_order_current import local_order_current


def test_local_order_current_uses_ordered_arms_and_local_blank_floor():
    delayed_tm4 = torch.tensor([2., 0., 2.])
    delayed_tm9 = torch.tensor([0., 2., 0.])
    now_tm4 = torch.tensor([0., 2., 0.])
    now_tm9 = torch.tensor([3., 0., 3.])
    floor = torch.tensor([1., 0., 7.])
    current = local_order_current(delayed_tm4, delayed_tm9,
                                  now_tm4, now_tm9, floor,
                                  gain=.25, cap=1.)
    assert torch.equal(current, torch.tensor([1., 0., 0.]))


def test_local_order_current_rejects_nonphysical_gain_or_cap():
    zero = torch.zeros(1)
    for gain, cap in ((0., 1.), (1., 0.), (1., -1.)):
        try:
            local_order_current(zero, zero, zero, zero, zero,
                                gain=gain, cap=cap)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid local current parameters accepted')

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from m1a_local_wall_model import reflect_feature, update_gain


def test_local_reflection_delta_moves_prediction_toward_visual_target():
    y = torch.tensor([.03])
    displacement = torch.tensor([-.03])
    linear, feature = reflect_feature(y, displacement, horizon=8)
    assert feature.item() > 0
    target = linear + feature
    gain = update_gain(0., feature, target-linear, eta=.5)
    assert 0 < gain <= 1
    assert abs((linear+gain*feature-target).item()) < abs((linear-target).item())

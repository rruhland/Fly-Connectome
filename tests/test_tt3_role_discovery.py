from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'experimental'))

from scripts.experimental import tt3_visual
from scripts.experimental.tt3_role_discovery import DiscoveryBinder, FEATURES


def _camera(tick, side):
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    y_shift = 7 if side else 0
    marks = ((16, 14, (220, 80, 120)), (34 + tick, 30, (90, 220, 120)), (77, 49, (120, 80, 220)))
    for x, y, color in marks:
        image[y+y_shift:y+y_shift+4, x:x+4] = color
    return image


def test_discovery_binder_updates_roles_at_reward_without_legacy_attribution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError('the discovery binder used the supplied reward witness')

    monkeypatch.setattr(tt3_visual, '_reward_witness', forbidden)
    coefficients = [0.] * (len(FEATURES) + 1)
    coefficients[FEATURES.index('recent_travel') + 1] = 1.
    clock = dict(center=[0.] * len(FEATURES), scale=[1.] * len(FEATURES), coefficients=coefficients)
    binder = DiscoveryBinder(tt3_visual.RoleModel.untrained(), clock, learn=True)
    for tick in range(20):
        binder.update((_camera(tick, False), _camera(tick, True)), np.zeros(3))
    binder.observe_reward(0.)
    assert not binder.reward_attributions
    binder.observe_reward(1.)
    assert len(binder.reward_attributions) == 1
    assert binder.reward_attributions[0]['contrasts'] > 0
    assert binder.model.receiving_reward_examples == binder.reward_attributions[0]['contrasts']

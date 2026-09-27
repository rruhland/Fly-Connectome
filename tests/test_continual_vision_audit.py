import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continual_vision_audit import CausalDynamicsStream
from fly_connectome.vision import load_legacy_default as load_default


def test_cached_observed_credit_matches_production_learning():
    torch.set_num_threads(1)
    live, cached = load_default(), load_default()
    runner = CausalDynamicsStream(cached.dynamics)
    previous = torch.zeros(64, 64)
    for t in range(25):
        frame = torch.zeros(64, 64)
        frame[25:28, 5+t:8+t] = 1.
        event = torch.stack(((previous-frame).clamp(min=0), (frame-previous).clamp(min=0)))
        previous = frame
        result = live.step(event, None if 12 <= t < 15 else frame, learn=True)
        positions = {e['id']: e['position'] for e in result['entities'] if e['observed']}
        runner.step(positions, available=not (12 <= t < 15), learn=True)
    for h, model in live.dynamics.items():
        other = cached.dynamics[h]
        assert model.seen == other.seen
        assert torch.equal(model.keys, other.keys)
        assert torch.equal(model.values, other.values)
        assert torch.equal(torch.stack(list(model.calibration.ranks)), torch.stack(list(other.calibration.ranks)))

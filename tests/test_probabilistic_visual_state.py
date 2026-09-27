import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probabilistic_visual_state import ProbabilisticVisualState


def sample(x):
    image = torch.zeros(16, 32)
    image[8, x] = 1.
    events = torch.zeros(2, 16, 32)
    events[1, 8, x] = 1.
    return events, image


def test_checkpoint_replay_and_continued_learning(tmp_path):
    torch.set_num_threads(1)
    model = ProbabilisticVisualState(height=16, width=32)
    for x in range(3, 24):
        model.step(*sample(x), learn=True)
    path = tmp_path/'belief.pt'
    model.save(path)
    loaded = ProbabilisticVisualState.load(path)
    model.reset_state()
    for x in range(3, 24):
        a, b = model.step(*sample(x), learn=True), loaded.step(*sample(x), learn=True)
        assert len(a['forecasts']) == len(b['forecasts'])
        for left, right in zip(a['forecasts'], b['forecasts']):
            for key in ('position', 'mixture_centers', 'mixture_weights', 'marginal_interval_90'):
                assert torch.equal(left[key], right[key])
    for h in model.dynamics:
        assert torch.equal(model.dynamics[h].values, loaded.dynamics[h].values)
        assert torch.equal(model.dynamics[h].generator.get_state(), loaded.dynamics[h].generator.get_state())


def test_outage_caches_beliefs_without_self_training():
    model = ProbabilisticVisualState(height=16, width=32)
    for x in range(3, 15):
        state = model.step(*sample(x), learn=True)
    issued = next(f for f in state['forecasts'] if f['horizon_samples'] == 8)
    counts = {h: m.seen for h, m in model.dynamics.items()}
    ranks = {h: len(m.calibration.ranks) for h, m in model.dynamics.items()}
    state = model.step(torch.zeros(2, 16, 32), None, learn=True)
    cached = next(f for f in state['forecasts'] if f['horizon_samples'] == 7)
    assert torch.equal(cached['mixture_centers'], issued['mixture_centers'])
    assert cached['evidence_age'] == 1
    assert counts == {h: m.seen for h, m in model.dynamics.items()}
    assert ranks == {h: len(m.calibration.ranks) for h, m in model.dynamics.items()}

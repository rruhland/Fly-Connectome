import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt10_strategy import StrategyMemory, optical_exchange


def test_optical_continuation_requires_an_exchanged_ball():
    positions = [(20, -.1, -1.25), (21, -.05, -1.12), (36, .25, .57), (37, .28, .67),
                 (41, .31, 1.17), (43, .31, .87), (61, .29, -.85), (63, .27, -.60),
                 (70, .25, .27)]
    trace = [dict(tick=tick, ball=[x, y, .29]) for tick, x, y in positions]
    estimate = optical_exchange(trace, 18.)
    assert estimate['outbound_slope'] == pytest.approx(.05 / .13)
    assert estimate['reply'] == 43
    assert estimate['follow_up']
    assert not optical_exchange(trace[:-1], 18.)['follow_up']
    assert optical_exchange(trace[:5], 18.)['reply'] is None


def test_strategy_context_can_choose_between_equally_rewarded_sweeps():
    low = np.array([1., -1., 0.])
    high = np.array([1., 1., 0.])
    memory = StrategyMemory()
    assert np.argmax(memory.values(low)) == 0
    for _ in range(12):
        memory.observe(low, 1, 1., dict(follow_up=True))
        memory.observe(high, 1, -1., dict(follow_up=False))
        memory.observe(low, 2, -1., dict(follow_up=False))
        memory.observe(high, 2, 1., dict(follow_up=True))
    memory.observe(low, 0, 0., dict(follow_up=True))
    restored = StrategyMemory(**memory.state())
    assert len(restored.samples) == 48
    assert np.argmax(restored.values(low)) == 1
    assert np.argmax(restored.values(high)) == 2
    assert restored.values(low, contextual=False)[1] == restored.values(high, contextual=False)[2]

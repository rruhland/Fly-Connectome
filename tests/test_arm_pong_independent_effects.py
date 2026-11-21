import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from arm_pong_rewire import IndependentEffects


def hold(binding, button, sign, start):
    paddle = .5
    binding.reset_trace()
    for tick in range(start, start + 56):
        binding.advance(button)
        paddle += sign * binding.speed[button] / 120
        if tick % 4 == 0:
            binding.observe(paddle, tick)


def test_both_effects_require_individual_evidence_and_can_change():
    binding = IndependentEffects(1, lock_after_acquisition=False)
    hold(binding, 1, 1, 0)
    assert binding.signs == [0, 1]
    assert binding.down_button is None
    assert binding.resolve(-1) == 0
    hold(binding, 0, -1, 56)
    assert binding.down_button == 1
    hold(binding, 1, -1, 112)
    assert binding.down_button is None
    assert binding.resolve(1) == 0
    hold(binding, 0, 1, 168)
    assert binding.down_button == 0


def test_frozen_comparison_keeps_both_initial_effects():
    binding = IndependentEffects(1, lock_after_acquisition=True)
    hold(binding, 0, 1, 0)
    hold(binding, 1, -1, 56)
    hold(binding, 0, -1, 112)
    assert binding.signs == [1, -1]
    assert binding.resolve(0) is None

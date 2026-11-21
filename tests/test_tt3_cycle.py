import os

import numpy as np
import pytest

from scripts.experimental.tt3_arena import TT3Arena
from scripts.experimental.tt3_visual import _components, _reference_components


@pytest.mark.skipif(
    not os.environ.get("TT3_COMPONENT_LIBRARY"),
    reason="optional compiled camera backend",
)
def test_compiled_components_preserve_positions_appearance_area_and_reference_order():
    rng = np.random.default_rng(10531)
    palette = rng.integers(0, 255, (6, 3), dtype=np.uint8)
    frames = [palette[rng.integers(0, 6, (31, 41))], np.zeros((2, 3, 3), np.uint8)]
    arena = TT3Arena(5321, distractor=True)
    for _ in range(12):
        images, _, _, _ = arena.step(rng.uniform(-1, 1, 3))
        frames.extend(images)
    for image in frames:

        def describe(components):
            return [(c.color, c.area, tuple(c.position)) for c in components]

        assert describe(_components(image)) == describe(_reference_components(image))


def test_context_outcomes_round_trip_and_keep_both_motor_histories(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "scripts" / "experimental")
    )
    from tt3_cycle import ConditionedNumeric

    state = dict(
        coefficients=[[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        totals=[2.0, 1.0],
        counts=[2, 2],
    )
    learned = ConditionedNumeric(state)
    learned.identify(np.eye(3))
    for _ in range(8):
        learned.observe(0, -1)
    restored = ConditionedNumeric.restore(learned.state())
    rng = np.random.default_rng(11)
    assert restored.choose_for_response(np.array([[1.0], [0.0]]), rng) == 0
    assert restored.choose_for_response(np.eye(3), rng) == 1
    assert sum(restored.shared.counts) == 12
    assert restored.state() == learned.state()

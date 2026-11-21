"""A change of venue must preserve the receiving task's physics and payouts."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experimental.tt4_arena import TT4Arena


def test_hall_visuals_preserve_physical_trajectory():
    studio = TT4Arena(seed=85)
    hall = TT4Arena(seed=85, venue="hall")
    rng = np.random.default_rng(472)
    try:
        studio.reset(seed=429, render=False)
        hall.reset(seed=429, render=False)
        assert hall.model.ngeom > studio.model.ngeom
        for _ in range(studio.max_steps):
            action = rng.uniform(-1, 1, 4)
            _, score, term, trunc, info = studio.step(action, render=False)
            _, hall_score, hall_term, hall_trunc, hall_info = hall.step(action, render=False)
            np.testing.assert_allclose(studio.ball, hall.ball, atol=1e-10, rtol=0)
            np.testing.assert_allclose(studio.joints, hall.joints, atol=1e-10, rtol=0)
            assert (score, term, trunc, info) == (hall_score, hall_term, hall_trunc, hall_info)
            if term or trunc:
                break
        assert term or trunc
    finally:
        studio.close()
        hall.close()

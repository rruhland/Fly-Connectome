"""Keep deployed hall checkpoints compatible as the native optics change."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
from scripts.experimental.tt5_checkpoint import load, save
from scripts.experimental.tt5_vision import HallTracker


def test_optional_blade_mode_and_random_state_survive_a_point_boundary(tmp_path):
    original = Path(__file__).resolve().parents[1] / "artifacts/tt5/first-hall-player.npz"
    player = load(original)
    tracker = player.tracker_factory()
    assert not tracker.motion_supported_blade
    player.tracker_factory = lambda: HallTracker(tracker.stereo, tracker.reference, tracker.apertures,
                                                 motion_supported_blade=True)
    path = tmp_path / "player.npz"
    save(path, player)
    restored = load(path)
    assert restored.tracker_factory().motion_supported_blade
    assert restored.prior.choose() == player.prior.choose()
    np.testing.assert_array_equal(restored.adapter.gains, player.adapter.gains)

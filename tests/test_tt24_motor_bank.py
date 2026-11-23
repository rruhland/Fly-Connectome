import numpy as np
import pytest

from scripts.experimental.tt24_motor_bank import MotorBank, STEP


@pytest.mark.parametrize("measured", [False, True])
def test_discover_reload_and_recognize_motor_interface(tmp_path, measured):
    mapping = np.roll(np.eye(5), 1, axis=0) * np.array([1., -.9, 1.05, -.8, .93])
    desired = np.array([.15, -.12, -.02, .2, .05])
    position = np.zeros(5)

    def step(action):
        position[:] += STEP * (mapping @ action)
        return position.copy()

    bank = MotorBank()
    receipt = bank.connect(position.copy(), step, 3126, measured_settling=measured)
    assert receipt["known"] is None
    np.testing.assert_allclose(mapping @ bank.apply(desired), desired, atol=1e-12)
    checkpoint = tmp_path / "bank.npz"
    bank.save(checkpoint)
    restored = MotorBank.load(checkpoint)
    recall = restored.connect(position.copy(), step, 752, measured_settling=measured)
    assert recall["known"] == 1
    assert recall["ticks"] < receipt["ticks"]
    assert len(restored.maps) == 2
    np.testing.assert_array_equal(restored.apply(desired), bank.apply(desired))

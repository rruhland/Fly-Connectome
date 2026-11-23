import numpy as np

from scripts.experimental.tt24_rolling_motor import RollingMotorLearner


def test_mid_block_reload_preserves_rolling_history_and_retires_oldest(tmp_path):
    model = RollingMotorLearner(capacity=4, block_ticks=16)
    joints = np.zeros(5)
    model.begin(joints)
    # The device is parked. No actuator map may be inferred from these readings.
    for _ in range(51):
        model.observe(joints, joints)
    model.save(tmp_path / 'motor.npz')
    resumed = RollingMotorLearner.load(tmp_path / 'motor.npz')
    for learner in (model, resumed):
        for _ in range(37):
            learner.observe(joints, joints)
        assert not learner.end()
        assert learner.completed == 5
        assert len(learner.blocks) == 4
        assert np.array_equal(learner.map, np.eye(5))
    for a, b in zip(model.blocks, resumed.blocks, strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

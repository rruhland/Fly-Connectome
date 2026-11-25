import numpy as np

from scripts.experimental import tt86_motor_mixture as motor


def example():
    rng = np.random.default_rng(1356)
    # Two possible movements in exactly the same body and camera context.
    x = np.zeros((320, motor.D))
    u = np.zeros((320, motor.core.HORIZON, 6))
    u[:160, :, 0] = 0.18
    u[160:, :, 0] = -0.18
    return x, u + rng.normal(scale=0.0005, size=u.shape), np.ones(len(u), dtype=bool)


def test_modes_preserve_opposing_actions_in_the_same_visual_state(monkeypatch):
    monkeypatch.setattr(motor, "K", 2)
    x, u, good = example()
    policy = motor.Mixture.fit(x, u, good)
    _, _, _, weights, moves, _ = policy.infer(x[0])
    centers = moves.mean(1)[:, 0]
    assert centers.min() < -0.17 and centers.max() > 0.17
    assert np.isclose(weights.sum(), 1)


def test_actual_miss_reweights_executed_mode_and_reward_updates_local_fit(monkeypatch):
    monkeypatch.setattr(motor, "K", 2)
    x, u, good = example()
    policy = motor.Mixture.fit(x, u, good)
    before = policy.offset.copy()
    old = policy.counts.copy()
    policy.learn(x[0], u[0], False)
    assert np.isclose(np.sum(policy.counts - old), 1)
    assert np.array_equal(before, policy.offset)
    policy.learn(x[0], u[0] + 0.02, True)
    assert not np.array_equal(before, policy.offset)
    assert np.isclose(np.sum(policy.counts - old), 2)

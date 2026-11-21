import numpy as np

from scripts.experimental.tt6_shot_memory import ShotMemory, response_signature


def test_response_signature_survives_wiring_changes():
    rng = np.random.default_rng(99381)
    jacobian, settling = rng.normal(size=(4, 5)), rng.uniform(size=(5, 14))
    order = [4, 0, 2, 1, 3]
    signs = [-1, 1, -1, -1, 1]
    np.testing.assert_allclose(response_signature(jacobian, settling),
                               response_signature(jacobian[:, order] * signs, settling[order]))


def test_memories_prefer_different_options_and_survive_round_trip():
    first, second = np.array([.0]), np.array([1.])
    memory = ShotMemory(.2)
    memory.observe(first, 0, 1.)
    memory.observe(first, 1, -1.)
    memory.observe(second, 0, -1.)
    memory.observe(second, 1, 1.)
    assert np.argmax(memory.estimate(first)) == 0
    assert np.argmax(memory.estimate(second)) == 1
    restored = ShotMemory.from_state(memory.state())
    restored.observe(first, 0, -1.)
    restored.observe(first, 1, 1.)
    restored.observe(first, 1, 1.)
    assert np.argmax(restored.estimate(first)) == 1
    assert np.argmax(memory.estimate(first)) == 0

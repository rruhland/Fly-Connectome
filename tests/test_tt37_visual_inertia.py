import numpy as np

from scripts.experimental.tt37_visual_inertia import (
    DT,
    VisualInertia,
    future_velocity,
    training,
)


def test_velocity_labels_use_only_images_and_do_not_span_points():
    time = np.arange(8) * DT
    acc = np.array([0.0, -0.2, -9.81])
    pos = (
        np.array([0.1, -0.3, 1.2])
        + time[:, None] * np.array([0.0, -2.0, 1.2])
        + 0.5 * time[:, None] ** 2 * acc
    )
    states = np.c_[pos, np.tile([0.0, -2.0, 1.2], (len(time), 1))]
    data = {
        "sensor": np.c_[time, pos, states, np.zeros((len(time), 2))],
        "groups": np.zeros((len(time), 3)),
    }
    assert np.linalg.norm(future_velocity(pos[:4]) - [0.0, -2.0, 1.2]) < 0.06
    assert len(training(data)[0]) == 2
    data["groups"][4:] = 1
    assert len(training(data)[0]) == 0


def test_missing_images_reset_recent_history(tmp_path):
    path = tmp_path / "optical.npz"
    np.savez(path, features=np.zeros((16, 13)), updates=np.zeros((16, 3)), gated=True)
    filt = VisualInertia(path)
    state = np.array([0.1, -0.4, 1.2, 0.0, -2.0, 1.0])
    for t in np.arange(3) * DT:
        filt.update(t, state[:3], state)
    assert len(filt.samples) == 3
    assert filt.update(3 * DT, None, state) is state
    assert not filt.samples

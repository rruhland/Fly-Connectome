import numpy as np

from scripts.experimental.tt50_forecast_learning import CameraCorrection
from scripts.experimental.tt51_flight_risk import Kernel, sensor_future
from scripts.experimental.tt50_group_forecast import columns


def test_measured_outcome_adapts_only_its_gate():
    m = np.zeros(90)
    s = np.ones(90)
    c = np.zeros((90, 5))
    b = np.zeros(5)
    model = CameraCorrection({0.0: (m, s, c, b), 0.45: (m, s, c, b)})
    x = np.zeros(90)
    actual = np.array([0.17, 1.02, -0.2, -3.9, 0.51])
    assert not model.read(0.0, x).any()
    for _ in range(12):
        model.observe(0.0, x, actual)
    assert np.linalg.norm(model.read(0.0, x) - actual) < np.linalg.norm(actual)
    assert not model.read(0.45, x).any()
    assert model.count == {0.0: 12, 0.45: 0}


def test_distributional_fit_retains_camera_and_live_vector_interface():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(60, 90))
    y = np.c_[
        0.1 * x[:, 0],
        1 + 0.015 * x[:, 1],
        0.03 * x[:, 2],
        -4 + 0.05 * x[:, 3],
        0.7 + 0.05 * x[:, 4],
    ]
    labels = sensor_future(y)
    assert labels.shape == (60, 9)
    model = Kernel.fit(x, labels, np.ones(len(x)))
    assert np.isfinite(model.predict(x[0])).all()
    assert np.array_equal(model.data[0], columns("position_face"))
    assert np.array_equal(
        model.predict(x[:2]), Kernel(tuple(model.data)).predict(x[:2])
    )

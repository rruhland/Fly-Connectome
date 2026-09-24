import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from m1a_motion_decode import fit_centroids, predict_direction


def test_calibration_centroids_decode_unseen_direction_samples():
    samples = [('up', np.array([3., 0., 1., 0.])),
               ('up', np.array([3.1, 0., 1., 0.])),
               ('down', np.array([0., 3., 0., 1.])),
               ('down', np.array([0., 3.1, 0., 1.])),
               ('left', np.array([1., 0., 3., 0.])),
               ('left', np.array([1., 0., 3.1, 0.])),
               ('right', np.array([0., 1., 0., 3.])),
               ('right', np.array([0., 1., 0., 3.1]))]
    model = fit_centroids(samples)
    for label, features in samples:
        assert predict_direction(model, features) == label

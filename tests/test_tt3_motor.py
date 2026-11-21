"""Check that the visual body fitter learns a pose map and its action derivative."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from tt3_motor import COMMAND_SCALE, VisualBodyModel


def test_cubic_body_map_fits_visual_poses_and_differentiates_in_command_units():
    def visual_position(coordinates):
        u, v, w = coordinates
        return np.array((.1 + .16 * u - .025 * v**2 + .06 * u * w,
                         -.14 + .12 * v + .028 * u * w,
                         -.1 + .09 * w - .025 * u**2))

    model = VisualBodyModel()
    rng = np.random.default_rng(105)
    for coordinates in rng.uniform(-1., 1., (400, 3)):
        model.add(coordinates, visual_position(coordinates))
    sample = np.array((.23, -.41, .17))
    pose = model.pose(sample)
    np.testing.assert_allclose(pose, visual_position(sample), atol=.0025)
    steps = np.eye(3) * 1e-5
    finite_difference = (model.pose(sample + steps) - pose).T / 1e-5 / COMMAND_SCALE
    np.testing.assert_allclose(model.derivative(sample), finite_difference, atol=2e-7)
    saved = model.snapshot()
    model.add(sample, visual_position(sample) + .12)
    model.restore(saved)
    np.testing.assert_allclose(model.pose(sample), pose)

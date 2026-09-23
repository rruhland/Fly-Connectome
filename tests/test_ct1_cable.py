import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_ct1_cable import cable_nearest


def test_cable_nearest_follows_branch_path_instead_of_3d_shortcut():
    # Nodes 1 and 4 are close in 3D but separated by a long U-shaped cable.
    points = np.array([[0., 0., 0.], [0., 10., 0.],
                       [10., 10., 0.], [10., 0., 0.],
                       [1., 0., 0.]])
    distance, source, roots = cable_nearest(
        points, np.array([-1, 1, 2, 3, 4]),
        np.array([1, 2, 3, 4, 5]),
        input_nodes=np.array([0, 3]),
        input_attachment=np.array([0., 0.]))
    assert roots == 1
    assert source[4] == 1
    assert np.isclose(distance[4], 9.)

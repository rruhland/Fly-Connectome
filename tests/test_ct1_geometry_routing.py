import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from ct1_geometry_routing import route_output_contacts


def test_routing_uses_nearest_eligible_input_and_preserves_contact_mass():
    posts, columns, contacts = route_output_contacts(
        input_body=np.array([11, 12, 13]),
        input_xyz=np.array([[0, 0, 0], [100, 0, 0], [2, 0, 0]]),
        output_body=np.array([21, 21, 22]),
        output_xyz=np.array([[1, 0, 0], [3, 0, 0], [99, 0, 0]]),
        source_columns={11: 4, 12: 9}, target_nodes={21: 1, 22: 2})
    assert list(zip(posts.tolist(), columns.tolist(), contacts.tolist())) == [
        (1, 4, 2), (2, 9, 1)]

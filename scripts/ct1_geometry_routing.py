"""Assign measured CT1 output contacts to nearest eligible CT1 input sites."""

import numpy as np
from scipy.spatial import cKDTree


def route_output_contacts(input_body, input_xyz, output_body, output_xyz,
                          source_columns, target_nodes):
    """Return target nodes, source columns, and contact counts; no new contacts."""
    eligible = np.isin(input_body, list(source_columns))
    if not eligible.any() or len(output_body) != len(output_xyz):
        raise ValueError('eligible CT1 input and measured output sites required')
    nearest = cKDTree(input_xyz[eligible]).query(output_xyz, k=1)[1]
    selected_input = input_body[eligible][nearest]
    columns = np.asarray([source_columns[int(body)] for body in selected_input],
                         dtype=np.int64)
    targets = np.asarray([target_nodes[int(body)] for body in output_body],
                         dtype=np.int64)
    if (columns < 0).any() or (targets < 0).any():
        raise ValueError('all routed contacts need local columns and target nodes')
    pairs, counts = np.unique(np.column_stack((targets, columns)),
                              axis=0, return_counts=True)
    if counts.sum() != len(output_body):
        raise AssertionError('routing changed measured contact count')
    return pairs[:, 0], pairs[:, 1], counts

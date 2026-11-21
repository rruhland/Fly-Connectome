import numpy as np

from scripts.experimental.tt17_compound import permutation_matrix


def test_a_recoded_query_and_fitted_coupling_use_opposite_coordinate_changes():
    order, signs = [2,0,1], [-1,1,-1]
    p = permutation_matrix(order, signs)
    native_sequence = np.array([[1.,-1.,0.],[.6,.3,-.1]])
    changed = native_sequence @ p
    np.testing.assert_allclose(changed @ p.T,native_sequence)
    residual = np.array([[.61,.19,0.],[0.,.79,.12],[.24,0.,.69]])
    represented_in_public_ports = p.T @ residual @ p
    np.testing.assert_allclose((changed @ represented_in_public_ports.T) @ p.T,
                               native_sequence @ residual.T)

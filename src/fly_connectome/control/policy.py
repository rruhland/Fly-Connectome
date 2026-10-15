"""Read-only linear action values over the generic visual transport."""

import numpy as np


class FrozenLinearPolicy:
    """Evaluation owns a defensive weight copy; ties choose first legal action.

    Action indices are local to an explicitly supplied actuator. No learning,
    exploration, feature normalization, or RNG consumption occurs here.
    """

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all():
            raise ValueError('finite nonempty action-by-feature weights required')
        self._weights = weights.copy()
        self._weights.flags.writeable = False

    @property
    def weights(self):
        return self._weights.copy()

    def q_values(self, indices, values):
        indices, values = np.asarray(indices), np.asarray(values, dtype=np.float64)
        if (indices.ndim != 1 or values.shape != indices.shape
                or indices.dtype.kind not in 'iu' or not np.isfinite(values).all()
                or (indices < 0).any() or (indices >= self._weights.shape[1]).any()
                or len(np.unique(indices)) != len(indices)):
            raise ValueError('unique in-range integer indices and finite values required')
        return self._weights[:, indices] @ values

    def choose(self, indices, values, legal_actions=None):
        q = self.q_values(indices, values)
        legal = np.arange(len(q)) if legal_actions is None else np.asarray(legal_actions)
        if (legal.ndim != 1 or not len(legal) or legal.dtype.kind not in 'iu'
                or (legal < 0).any() or (legal >= len(q)).any()
                or len(np.unique(legal)) != len(legal)):
            raise ValueError('nonempty unique legal action indices required')
        return int(legal[np.argmax(q[legal])]), q

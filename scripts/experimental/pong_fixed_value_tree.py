"""Fixed randomized averaging partitions and supported empirical Bellman values.

Only existing capped raw visual coordinates enter this stateless controller.
No outcome-dependent partition fitting, gradients, expert targets or planning.
"""
import time
import numpy as np
import scipy.sparse as sp
from pong_fixed_pairwise import FixedPairwise, FrozenPairwiseReadout

SCHEMA = 'fixed-supported-value-forest-v1'


def frozen(a, dtype=None):
    a = np.array(a, dtype=dtype, copy=True)
    assert np.isfinite(a).all()
    a.flags.writeable = False
    return a


def visual_features(X, original, B):
    X = sp.csr_matrix(X)
    return np.column_stack((X @ B.T, X[:, 4384].toarray().ravel(), original.basis.batch(X)))


def build_forest(F, rng):
    assert F.shape == (4096, 711) and np.isfinite(F).all()
    trees = []; leaf_count = 0
    for _ in range(32):
        nodes = []
        def grow(rows, depth):
            nonlocal leaf_count
            node = len(nodes); nodes.append([-1, 0., -1, -1, -1])
            if depth < 10 and len(rows) >= 32:
                for coordinate in rng.choice(711, 32, replace=False):
                    values = F[rows, coordinate]; low, high = values.min(), values.max()
                    if high > low:
                        threshold = rng.uniform(low, high)
                        left = rows[values <= threshold]; right = rows[values > threshold]
                        if len(left) >= 16 and len(right) >= 16:
                            l = grow(left, depth+1); r = grow(right, depth+1)
                            nodes[node] = [int(coordinate), threshold, l, r, -1]
                        break  # first positive range; never retry threshold/coordinate
            if nodes[node][0] == -1:
                nodes[node][4] = leaf_count; leaf_count += 1
            return node
        grow(np.arange(len(F)), 0); trees.append(nodes)
    size = max(map(len, trees)); table = np.full((32, size, 5), -1., dtype=float)
    for i, nodes in enumerate(trees): table[i, :len(nodes)] = nodes
    return table, leaf_count


def leaves_batch(F, table):
    out = np.empty((len(F), 32), dtype=np.int32)
    for t in range(32):
        nodes = np.zeros(len(F), dtype=np.int32)
        for _ in range(11):
            coord = table[t, nodes, 0].astype(np.int32)
            active = np.flatnonzero(coord >= 0)
            if not len(active): break
            current = nodes[active]
            right = F[active, coord[active]] > table[t, current, 1]
            nodes[active] = np.where(right, table[t, current, 3], table[t, current, 2]).astype(np.int32)
        assert np.all(table[t, nodes, 0] == -1)
        out[:, t] = table[t, nodes, 4].astype(np.int32)
    return out


def supported_values(leaves, values, eligible):
    mask = eligible[leaves]
    counts = mask.sum(axis=1)
    q = np.divide(np.sum(values[leaves]*mask, axis=1), counts,
                  out=np.zeros((len(leaves), 3)), where=counts > 0)
    supported = counts >= 24
    return q, supported, counts


def fit_values(current, successor, actions, rewards, discounts, weights, leaf_count, deadline):
    cells = current*3 + actions[:, None]
    count = np.bincount(cells.ravel(), minlength=leaf_count*3).reshape(-1, 3)
    weight = np.bincount(cells.ravel(), weights=np.repeat(weights, 32), minlength=leaf_count*3).reshape(-1, 3)
    square = np.bincount(cells.ravel(), weights=np.repeat(weights**2, 32), minlength=leaf_count*3).reshape(-1, 3)
    ess = np.divide(weight**2, square, out=np.zeros_like(weight), where=square > 0)
    eligible = (count >= 4) & (ess >= 4)
    values = np.zeros((leaf_count, 3)); history = []
    _, support, _ = supported_values(successor, values, eligible)
    cutoff = ~support.any(axis=1) & (discounts > 0)
    for sweep in range(256):
        if time.monotonic() >= deadline: raise TimeoutError('fixed 256-sweep fit incomplete')
        q, support, _ = supported_values(successor, values, eligible)
        continuation = np.max(np.where(support, q, 0.), axis=1)
        target = rewards + discounts*continuation
        numerator = np.bincount(cells.ravel(), weights=np.repeat(weights*target, 32), minlength=leaf_count*3).reshape(-1, 3)
        updated = np.divide(numerator, weight, out=np.zeros_like(weight), where=weight > 0)
        assert np.isfinite(updated).all() and (updated >= 0).all()
        if sweep in (0, 63, 127, 255): history.append(dict(sweep=sweep+1, max_update=float(np.max(np.abs(updated-values))), max_value=float(updated.max())))
        values = updated
    return values, count, weight, ess, eligible, dict(history=history,
        coverage_cutoff_backup_rows=int(cutoff.sum()), natural_terminal_rows=int(np.count_nonzero(discounts == 0)),
        coverage_cutoff_weight=float(weights[cutoff].sum()), completed_sweeps=256)


class FixedValueTreeReadout:
    schema = SCHEMA
    def __init__(self, original, B, table, values, count, weight, ess, eligible):
        self.original = original; self.base = original.base; self.basis = original.basis
        self.B = frozen(B); self.table = frozen(table); self.values = frozen(values)
        self.count = frozen(count, np.int64); self.weight = frozen(weight); self.ess = frozen(ess)
        self.eligible = frozen(eligible, bool)
        assert self.B.shape == (198, 4389) and self.table.shape[0] == 32
        assert self.values.shape == self.count.shape == self.weight.shape == self.ess.shape == self.eligible.shape
        assert self.values.shape[1] == 3 and (self.values >= 0).all()
        assert np.array_equal(self.eligible, (self.count >= 4) & (self.ess >= 4))

    def features(self, indices, values):
        where = np.flatnonzero(np.asarray(indices) == 4384)
        present = float(values[where[0]]) if len(where) else 0.
        return np.r_[self.B[:, indices] @ values, present, self.basis.features(indices, values)]

    def inspect(self, indices, values):
        F = self.features(indices, values)
        nodes = np.zeros(32, dtype=np.int32); tree = np.arange(32)
        for _ in range(11):
            coordinate = self.table[tree, nodes, 0].astype(np.int32)
            active = np.flatnonzero(coordinate >= 0)
            if not len(active): break
            current = nodes[active]
            right = F[coordinate[active]] > self.table[active, current, 1]
            nodes[active] = np.where(right, self.table[active, current, 3], self.table[active, current, 2]).astype(np.int32)
        leaves = self.table[tree, nodes, 4].astype(np.int32)
        q, supported, counts = supported_values(leaves[None, :], self.values, self.eligible)
        return q[0], supported[0], counts[0], leaves

    def choose(self, indices, values, legal_actions=None):
        q, supported, _, _ = self.inspect(indices, values)
        legal = np.arange(3) if legal_actions is None else np.asarray(legal_actions)
        if not supported[legal].any(): return self.original.choose(indices, values, legal_actions)
        q = np.where(supported, q, -1.)
        return int(legal[np.argmax(q[legal])]), q

    def frozen_parameters(self):
        return (SCHEMA, self.original.frozen_parameters(), self.B, self.table, self.values,
                self.count, self.weight, self.ess, self.eligible)


def save_readout(path, model):
    b = model.original.basis
    np.savez_compressed(path, kind=SCHEMA, base=model.base.weights, hashes=b.hashes,
        signs=b.signs, scale=b.scale, mean=b.mean, rms=b.rms, residual=model.original.residual,
        tree_B=model.B, tree_table=model.table, tree_values=model.values, tree_count=model.count,
        tree_weight=model.weight, tree_ess=model.ess, tree_eligible=model.eligible)


def from_archive(f):
    original = FrozenPairwiseReadout(f['base'], FixedPairwise(f['hashes'], f['signs'], f['scale'], f['mean'], f['rms']), f['residual'])
    return FixedValueTreeReadout(original, *[f['tree_'+k] for k in ('B', 'table', 'values', 'count', 'weight', 'ess', 'eligible')])

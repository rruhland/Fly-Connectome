"""Exact single-thread training queries, avoiding duplicate tree traversals.

Support is only actual-option presence in a majority of leaves. Independent
start counts stay a separate initial/final diagnostic, never a support input.
"""
import numpy as np
from pong_smdp_adaptive_forest import (OPTION_COUNT, TREES, MIN_SUPPORTED_TREES,
    option_features)


def values_from_leaves(model, leaves):
    """Match sklearn's n_jobs=1 tree-order float64 additions exactly."""
    leaves = np.asarray(leaves)
    values = np.zeros(len(leaves), dtype=np.float64)
    for tree, estimator in enumerate(model.estimators_):
        values += estimator.tree_.value[leaves[:, tree], 0, 0]
    values /= len(model.estimators_)
    return values


def option_presence_from_leaves(model, leaves, options):
    leaves = np.asarray(leaves, dtype=np.int64)
    options = np.asarray(options, dtype=np.int64)
    size = max(t.tree_.node_count for t in model.estimators_)
    presence = np.zeros((TREES, size, OPTION_COUNT), dtype=bool)
    for tree in range(TREES):
        cells = leaves[:, tree] * OPTION_COUNT + options
        presence[tree] = np.bincount(cells, minlength=size * OPTION_COUNT).reshape(size, OPTION_COUNT) > 0
    return presence


def independent_counts_from_leaves(model, leaves, options, start_ids):
    """Original true distinct-start diagnostic, used only initial/final."""
    size = max(t.tree_.node_count for t in model.estimators_)
    counts = np.zeros((TREES, size, OPTION_COUNT), np.int32)
    for tree in range(TREES):
        triples = np.unique(np.column_stack((leaves[:, tree], options, start_ids)), axis=0)
        cells = triples[:, 0] * OPTION_COUNT + triples[:, 1]
        counts[tree] = np.bincount(cells, minlength=size * OPTION_COUNT).reshape(size, OPTION_COUNT)
    return counts


def training_leaf_cache(model, features, options, selected=None):
    """One C apply supplies observed predictions and queried-option presence."""
    leaves = model.apply(features)
    selected = np.ones(len(leaves), bool) if selected is None else np.asarray(selected, bool)
    presence = option_presence_from_leaves(model, leaves[selected], np.asarray(options)[selected])
    return leaves, presence, values_from_leaves(model, leaves)


def fast_q_values(model, presence, states, valid, chunk=4096):
    states = np.asarray(states, dtype=np.float32)
    q = np.zeros((len(states), OPTION_COUNT))
    supported = np.zeros_like(q, dtype=bool)
    trees = np.arange(TREES)[None, :]
    for begin in range(0, len(states), chunk):
        end = min(begin + chunk, len(states))
        options = np.tile(np.arange(OPTION_COUNT), end - begin)
        F = option_features(np.repeat(states[begin:end], OPTION_COUNT, axis=0), options)
        leaves = model.apply(F)
        q[begin:end] = values_from_leaves(model, leaves).reshape(-1, OPTION_COUNT)
        counts = presence[trees, leaves, options[:, None]].sum(axis=1)
        supported[begin:end] = (counts >= MIN_SUPPORTED_TREES).reshape(-1, OPTION_COUNT)
    supported &= np.asarray(valid, bool)[:, None]
    return q, supported


def readonly_c_forest(exported):
    """Rehydrate saved parameters for query-only parity; never fit or alter them.

    Node sample/impurity fields are unavailable in the portable export and are
    irrelevant to finite-input apply/predict. They are zero placeholders. This
    reconstruction is for verification only and is never a training initializer.
    """
    from sklearn.tree import _tree
    from sklearn.tree import DecisionTreeRegressor
    estimators = []
    for tree in range(TREES):
        stack = [(0, 0)]; largest = 0; depth = 0
        while stack:
            node, level = stack.pop(); largest = max(largest, node); depth = max(depth, level)
            if exported.left[tree, node] >= 0:
                stack.extend([(int(exported.left[tree, node]), level + 1),
                    (int(exported.right[tree, node]), level + 1)])
        count = largest + 1
        compiled = _tree.Tree(38, np.ones(1, dtype=np.intp), 1)
        dtype = compiled.__getstate__()['nodes'].dtype
        nodes = np.zeros(count, dtype=dtype)
        nodes['left_child'] = exported.left[tree, :count]
        nodes['right_child'] = exported.right[tree, :count]
        nodes['feature'] = exported.feature[tree, :count]
        nodes['threshold'] = exported.threshold[tree, :count]
        values = exported.value[tree, :count].reshape(count, 1, 1).copy()
        compiled.__setstate__(dict(max_depth=depth, node_count=count, nodes=nodes, values=values))
        estimator = DecisionTreeRegressor()
        estimator.tree_ = compiled; estimator.n_features_in_ = 38; estimator.n_outputs_ = 1
        estimators.append(estimator)
    class QueryOnlyForest:
        estimators_ = estimators
        def apply(self, features):
            features = np.asarray(features, np.float32)
            return np.column_stack([t.apply(features, check_input=False) for t in self.estimators_])
        def predict(self, features):
            features = np.asarray(features, np.float32)
            total = np.zeros(len(features), np.float64)
            for estimator in self.estimators_:
                total += estimator.predict(features, check_input=False)
            return total / TREES
    return QueryOnlyForest()

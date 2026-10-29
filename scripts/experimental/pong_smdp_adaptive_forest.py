"""Observed finite-duration options and reward-adaptive fitted values, no gradients."""
from pathlib import Path
import numpy as np

SCHEMA = 'anonymous-smdp-adaptive-forest-v1'
CAPACITY = 4
HOLD = 16
STATE_DIM = 34
OPTION_COUNT = 4
TREES = 32
MIN_SUPPORTED_TREES = 17
HALF_LIFE_SECONDS = 3.2


def state_features(scene, phase, past_hit, scales):
    rows = np.asarray(scene, dtype=np.float64).reshape(-1, 6)
    if not np.isfinite(rows).all() or not np.isin(rows[:, 4:], [0, 1]).all():
        raise ValueError('finite anonymous six-field visual rows required')
    if not len(rows) or len(rows) > CAPACITY:
        return None
    if not 0 <= int(phase) < 5:
        raise ValueError('public camera phase must be in [0,5)')
    rows = rows[np.lexsort(rows[:, [5, 4, 3, 2, 0, 1]].T)]
    padded = np.zeros((CAPACITY, 6), dtype=np.float64)
    padded[:len(rows)] = rows
    padded[:, :4] /= np.asarray(scales) * 4
    presence = np.zeros(CAPACITY); presence[:len(rows)] = 1
    return np.r_[padded.ravel(), presence, bool(past_hit), np.eye(5)[int(phase)]].astype(np.float32)


def option_features(states, options):
    states = np.asarray(states, dtype=np.float32).reshape(-1, STATE_DIM)
    options = np.asarray(options, dtype=np.int64).reshape(-1)
    if len(states) != len(options) or not np.isin(options, range(OPTION_COUNT)).all():
        raise ValueError('one declared categorical option per state required')
    return np.column_stack((states, np.eye(OPTION_COUNT, dtype=np.float32)[options]))


def discount(elapsed):
    elapsed = np.asarray(elapsed, dtype=np.float64)
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError('positive actual elapsed durations required')
    return np.exp2(-elapsed / HALF_LIFE_SECONDS)


def observed_windows(actions, hits, natural, baseline_actions, successor_available):
    """Only observed constant commands or actual baseline one-camera actions.

    Returns (source, option, last_camera, kind), with kind 1=own hit, 2=natural
    end, 0=complete nonterminal window. No hit is turned into environment end.
    """
    actions = np.asarray(actions, dtype=np.int8)
    hits = np.asarray(hits, dtype=bool)
    natural = np.asarray(natural, dtype=bool)
    baseline_actions = np.asarray(baseline_actions, dtype=np.int8)
    successor_available = np.asarray(successor_available, dtype=bool)
    n = len(actions)
    if any(a.shape != (n,) for a in (hits, natural, baseline_actions, successor_available)):
        raise ValueError('one aligned record per actual transition required')
    run = np.ones(n, dtype=np.int32)
    event = np.full(n, n, dtype=np.int32)
    nearest = n
    for i in range(n - 1, -1, -1):
        if i + 1 < n and actions[i] == actions[i + 1]:
            run[i] += run[i + 1]
        if hits[i] or natural[i]:
            nearest = i
        event[i] = nearest
    rows = []
    rejected = dict(changing_action_before_boundary=0, censored_short_macro=0,
        missing_successor_macro=0, nonbaseline_one_camera=0, missing_successor_baseline=0)
    for i in range(n):
        kind = 1 if hits[i] else 2 if natural[i] else 0
        if actions[i] == baseline_actions[i]:
            if kind or successor_available[i]:
                rows.append((i, 3, i, kind))
            else:
                rejected['missing_successor_baseline'] += 1
        else:
            rejected['nonbaseline_one_camera'] += 1
        end = min(i + HOLD - 1, int(event[i]), n - 1)
        length = end - i + 1
        if run[i] < length:
            rejected['changing_action_before_boundary'] += 1
            continue
        kind = 1 if hits[end] else 2 if natural[end] else 0
        if length < HOLD and not kind:
            rejected['censored_short_macro'] += 1
            continue
        if not kind and not successor_available[end]:
            rejected['missing_successor_macro'] += 1
            continue
        rows.append((i, int(actions[i]), end, kind))
    return rows, rejected


def observed_mc_returns(hits, natural, elapsed):
    """Discount through the first future hit interval; unknown tails stay NaN."""
    hits, natural = np.asarray(hits, bool), np.asarray(natural, bool)
    d = discount(elapsed)
    values = np.full(len(hits), np.nan)
    carry = np.nan
    for i in range(len(hits) - 1, -1, -1):
        if hits[i]:
            carry = d[i]
        elif natural[i]:
            carry = 0.
        elif np.isfinite(carry):
            carry *= d[i]
        values[i] = carry
    return values


def balanced_weights(options, start_ids, selected=None):
    """Equal option mass, then equal start mass, uniform windows within each."""
    options, start_ids = np.asarray(options), np.asarray(start_ids)
    selected = np.ones(len(options), dtype=bool) if selected is None else np.asarray(selected, bool)
    weights = np.zeros(len(options))
    present_options = np.unique(options[selected])
    if not len(present_options):
        return weights
    for option in present_options:
        mask = selected & (options == option)
        starts = np.unique(start_ids[mask])
        for start in starts:
            rows = mask & (start_ids == start)
            weights[rows] = 1 / (len(present_options) * len(starts) * rows.sum())
    assert np.isclose(weights.sum(), 1)
    return weights


class AdaptiveForest:
    """Portable sklearn-compatible scalar forest plus actual-option leaf support."""
    def __init__(self, scales, feature, threshold, left, right, value, independent):
        self.scales = np.asarray(scales, dtype=np.float64).copy()
        self.feature = np.asarray(feature, dtype=np.int32).copy()
        self.threshold = np.asarray(threshold, dtype=np.float64).copy()
        self.left = np.asarray(left, dtype=np.int32).copy()
        self.right = np.asarray(right, dtype=np.int32).copy()
        self.value = np.asarray(value, dtype=np.float64).copy()
        self.independent = np.asarray(independent, dtype=np.int32).copy()
        shape = self.feature.shape
        if (shape[0] != TREES or self.threshold.shape != shape or self.left.shape != shape
                or self.right.shape != shape or self.value.shape != shape
                or self.independent.shape != (*shape, OPTION_COUNT)):
            raise ValueError('32 trees with aligned values and option-specific support required')
        for a in (self.scales, self.feature, self.threshold, self.left, self.right, self.value, self.independent):
            if not np.isfinite(a).all():
                raise ValueError('finite exported forest required')
            a.flags.writeable = False

    @classmethod
    def export(cls, model, scales, features, options, start_ids, independent=None):
        size = max(t.tree_.node_count for t in model.estimators_)
        shape = (TREES, size)
        feature = np.full(shape, -2, dtype=np.int32)
        threshold = np.zeros(shape); left = np.full(shape, -1, dtype=np.int32)
        right = left.copy(); value = np.zeros(shape)
        if independent is None:
            independent = leaf_independence(model, features, options, start_ids)
        for tree, estimator in enumerate(model.estimators_):
            t = estimator.tree_; n = t.node_count
            feature[tree, :n] = t.feature; threshold[tree, :n] = t.threshold
            left[tree, :n] = t.children_left; right[tree, :n] = t.children_right
            value[tree, :n] = t.value[:, 0, 0]
        return cls(scales, feature, threshold, left, right, value, independent)

    def predict(self, features, options):
        F = np.asarray(features, dtype=np.float32).reshape(-1, STATE_DIM + OPTION_COUNT)
        options = np.asarray(options, dtype=np.int64)
        nodes = np.zeros((len(F), TREES), dtype=np.int32)
        trees = np.arange(TREES)[None, :]
        for _ in range(self.feature.shape[1] + 1):
            coordinate = self.feature[trees, nodes]
            active = coordinate >= 0
            if not active.any():
                break
            rows, columns = np.where(active)
            current = nodes[rows, columns]
            goes_right = F[rows, coordinate[rows, columns]] > self.threshold[columns, current]
            nodes[rows, columns] = np.where(goes_right, self.right[columns, current], self.left[columns, current])
        else:
            raise AssertionError('exported tree traversal did not reach leaves')
        assert (self.feature[trees, nodes] < 0).all()
        values = self.value[trees, nodes].cumsum(axis=1)[:, -1] / TREES
        support_count = (self.independent[trees, nodes, options[:, None]] >= 1).sum(axis=1)
        return values, support_count >= MIN_SUPPORTED_TREES, support_count

    def q_values(self, states, valid=None):
        states = np.asarray(states, dtype=np.float32).reshape(-1, STATE_DIM)
        options = np.tile(np.arange(OPTION_COUNT), len(states))
        expanded = np.repeat(states, OPTION_COUNT, axis=0)
        values, supported, count = self.predict(option_features(expanded, options), options)
        q = values.reshape(-1, OPTION_COUNT)
        supported = supported.reshape(-1, OPTION_COUNT)
        if valid is not None:
            supported &= np.asarray(valid, bool)[:, None]
        return q, supported, count.reshape(-1, OPTION_COUNT)

    def inspect(self, scene, phase, past_hit):
        state = state_features(scene, phase, past_hit, self.scales)
        if state is None:
            return dict(option=3, q=[0.] * OPTION_COUNT, supported=[False] * OPTION_COUNT,
                tree_support_counts=[0] * OPTION_COUNT, representation_supported=False)
        q, support, count = self.q_values(state[None])
        option = 3
        if past_hit and support[0, 3]:
            best = int(np.argmax(np.where(support[0], q[0], -np.inf)))
            if q[0, best] > q[0, 3]:
                option = best
        return dict(option=option, q=q[0].tolist(), supported=support[0].tolist(),
            tree_support_counts=count[0].tolist(), representation_supported=True)

    def save(self, path):
        np.savez_compressed(Path(path), kind=SCHEMA, scales=self.scales, feature=self.feature,
            threshold=self.threshold, left=self.left, right=self.right, value=self.value,
            independent=self.independent)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['kind']) != SCHEMA:
                raise ValueError('unknown SMDP forest schema')
            return cls(*[data[key] for key in ('scales', 'feature', 'threshold', 'left', 'right', 'value', 'independent')])

    def frozen_parameters(self):
        return SCHEMA, HOLD, STATE_DIM, TREES, MIN_SUPPORTED_TREES, self.scales, self.feature, self.threshold, self.left, self.right, self.value, self.independent


def leaf_independence(model, features, options, start_ids):
    leaves = model.apply(features)
    size = max(t.tree_.node_count for t in model.estimators_)
    counts = np.zeros((TREES, size, OPTION_COUNT), np.int32)
    for tree in range(TREES):
        triples = np.unique(np.column_stack((leaves[:, tree], options, start_ids)), axis=0)
        cells = triples[:, 0] * OPTION_COUNT + triples[:, 1]
        counts[tree] = np.bincount(cells, minlength=size * OPTION_COUNT).reshape(size, OPTION_COUNT)
    return counts


def sklearn_q_values(model, independent, states, valid, chunk=4096):
    """C tree predict/apply for fitting; portable NumPy traversal is runtime only."""
    states = np.asarray(states, np.float32)
    q = np.zeros((len(states), OPTION_COUNT))
    supported = np.zeros_like(q, bool)
    two_start_trees = np.zeros_like(q, np.int8)
    trees = np.arange(TREES)[None, :]
    for begin in range(0, len(states), chunk):
        end = min(begin + chunk, len(states))
        options = np.tile(np.arange(OPTION_COUNT), end - begin)
        F = option_features(np.repeat(states[begin:end], OPTION_COUNT, axis=0), options)
        q[begin:end] = model.predict(F).reshape(-1, OPTION_COUNT)
        leaves = model.apply(F)
        distinct = independent[trees, leaves, options[:, None]]
        supported[begin:end] = ((distinct >= 1).sum(1) >= MIN_SUPPORTED_TREES).reshape(-1, OPTION_COUNT)
        two_start_trees[begin:end] = (distinct >= 2).sum(1).reshape(-1, OPTION_COUNT)
    supported &= np.asarray(valid, bool)[:, None]
    return q, supported, two_start_trees


class SMDPController:
    def __init__(self, forest):
        self.forest = forest
        self.past_hit = False
        self.ordinal = 0
        self.remaining = 0
        self.held_action = None

    def choose(self, scene, baseline_action):
        before = self.remaining
        if not self.past_hit:
            option = 3; decision = dict(mode='baseline_before_hit')
        elif before:
            option = self.held_action; self.remaining -= 1
            decision = dict(mode='earned_hold')
        else:
            decision = self.forest.inspect(scene, self.ordinal % 5, self.past_hit)
            option = decision['option']
            if option != 3:
                self.remaining = HOLD - 1; self.held_action = option
            decision['mode'] = 'baseline_camera' if option == 3 else 'earned_start'
        action = int(baseline_action) if option == 3 else int(option)
        decision.update(option=int(option), action=action, baseline_action=int(baseline_action),
            past_hit=self.past_hit, ordinal=self.ordinal, phase=self.ordinal % 5,
            hold_before=before, hold_after=self.remaining)
        return action, decision

    def observe(self, hits, natural_terminal=False):
        self.ordinal += 1
        self.past_hit |= bool(hits)
        if hits or natural_terminal:
            self.remaining = 0; self.held_action = None

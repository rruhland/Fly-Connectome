"""Learned visual deltas and event masses; primitive receding-horizon control.

All searches use learned models only. No environment or geometry oracle is read.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
from sklearn.tree import _tree
from pong_smdp_adaptive_forest import AdaptiveForest, state_features, option_features
from pong_smdp_fast_training import readonly_c_forest, values_from_leaves

SCHEMA = 'anonymous-learned-delta-primitive-planner-v1'
HORIZON = 8
BEAM = 8
HISTORY = 16
WORLD_DIM = 117
EVENT_DIM = 147
TAIL_SHA = 'e58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduled_seconds(ordinal):
    t = np.asarray(ordinal, np.int64)
    return (((t + 1) * 12 // 5) - (t * 12 // 5)) / 120.


def inputs(states, histories, actions):
    states = np.asarray(states, np.float64).reshape(-1, 34)
    histories = np.asarray(histories, np.float64).reshape(-1, HISTORY * 3)
    actions = np.asarray(actions, np.int64).reshape(-1)
    onehot = np.eye(3)[actions]
    dt = scheduled_seconds(np.argmax(states[:, 29:], axis=1))[:, None]
    X = np.column_stack((states, onehot, histories, dt, dt * states[:, :28], dt * onehot))
    assert X.shape == (len(states), WORLD_DIM) and np.isfinite(X).all()
    rows = states[:, :24].reshape(-1, 4, 6)
    presence = states[:, 24:28]
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            active = presence[:, i] * presence[:, j]
            pairs.append(np.column_stack(((rows[:, j, :4] - rows[:, i, :4]) * active[:, None], active)))
    return X, np.column_stack((X, *pairs))


def decode_scene(predicted, states):
    """Generic binary decoding and horizontal content ordering, no identities."""
    predicted = np.asarray(predicted, np.float64).reshape(-1, 28)
    current = np.asarray(states).reshape(-1, 34)
    output = np.zeros((len(predicted), 34), np.float32)
    rows = predicted[:, :24].reshape(-1, 4, 6).copy()
    active = predicted[:, 24:28] >= .5
    valid = np.isfinite(predicted).all(axis=1) & active.any(axis=1)
    rows[:, :, 4:] = rows[:, :, 4:] >= .5
    rows[:, :, 2:4] *= rows[:, :, 5:6]
    order = np.lexsort((rows[:, :, 5], rows[:, :, 4], rows[:, :, 3], rows[:, :, 2],
        rows[:, :, 0], rows[:, :, 1], ~active), axis=1)
    rows = np.take_along_axis(rows, order[:, :, None], axis=1)
    active = np.take_along_axis(active, order, axis=1)
    rows[~active] = 0
    output[:, :24] = rows.reshape(-1, 24); output[:, 24:28] = active
    output[:, 28] = current[:, 28]
    output[:, 29:] = np.eye(5)[(np.argmax(current[:, 29:], axis=1) + 1) % 5]
    output[~valid] = 0
    return output, valid


class CompiledForest:
    def __init__(self, n_features, outputs, offsets, depths, feature, threshold, left, right, value):
        self.n_features, self.outputs = int(n_features), int(outputs)
        self.offsets = np.asarray(offsets, np.int64).copy(); self.depths = np.asarray(depths, np.int32).copy()
        self.feature = np.asarray(feature, np.int64).copy(); self.threshold = np.asarray(threshold, np.float64).copy()
        self.left = np.asarray(left, np.int64).copy(); self.right = np.asarray(right, np.int64).copy()
        self.value = np.asarray(value, np.float64).reshape(-1, self.outputs).copy()
        self.trees = []
        for i, (begin, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            count = end - begin
            tree = _tree.Tree(self.n_features, np.ones(self.outputs, np.intp), self.outputs)
            nodes = np.zeros(count, dtype=tree.__getstate__()['nodes'].dtype)
            nodes['left_child'] = self.left[begin:end]; nodes['right_child'] = self.right[begin:end]
            nodes['feature'] = self.feature[begin:end]; nodes['threshold'] = self.threshold[begin:end]
            tree.__setstate__(dict(max_depth=int(self.depths[i]), node_count=int(count), nodes=nodes,
                values=self.value[begin:end, :, None].copy()))
            self.trees.append(tree)
        for array in (self.offsets, self.depths, self.feature, self.threshold, self.left, self.right, self.value):
            array.flags.writeable = False

    @classmethod
    def export(cls, model):
        trees = [e.tree_ for e in model.estimators_]
        offsets = np.r_[0, np.cumsum([t.node_count for t in trees])]
        return cls(model.n_features_in_, model.n_outputs_, offsets, [t.max_depth for t in trees],
            np.concatenate([t.feature for t in trees]), np.concatenate([t.threshold for t in trees]),
            np.concatenate([t.children_left for t in trees]), np.concatenate([t.children_right for t in trees]),
            np.concatenate([t.value[:, :, 0] for t in trees]))

    def predict(self, features):
        F = np.asarray(features, np.float32).reshape(-1, self.n_features)
        result = np.zeros((len(F), self.outputs))
        for tree in self.trees:
            leaves = tree.apply(F)
            result += tree.value[leaves, :, 0]
        return result / len(self.trees)

    def arrays(self, prefix):
        return {prefix + key:np.asarray(getattr(self, key)) for key in
            ('n_features', 'outputs', 'offsets', 'depths', 'feature', 'threshold', 'left', 'right', 'value')}

    @classmethod
    def load(cls, data, prefix):
        return cls(*[data[prefix + key] for key in
            ('n_features', 'outputs', 'offsets', 'depths', 'feature', 'threshold', 'left', 'right', 'value')])

    def frozen_parameters(self):
        return (self.n_features, self.outputs, self.offsets, self.depths, self.feature, self.threshold,
            self.left, self.right, self.value, tuple((t.children_left, t.children_right,
                t.feature, t.threshold, t.value) for t in self.trees))


class DeltaPlannerModel:
    def __init__(self, scales, mean, scale, delta_scale, ridge, residual, events, tail_path):
        self.scales = np.asarray(scales).copy(); self.mean = np.asarray(mean).copy()
        self.scale = np.asarray(scale).copy(); self.delta_scale = np.asarray(delta_scale).copy()
        self.ridge = np.asarray(ridge).copy(); self.residual, self.events = residual, events
        self.tail_path = str(tail_path)
        if sha(self.tail_path) != TAIL_SHA:
            raise ValueError('frozen C1 approximate tail changed')
        self.tail = AdaptiveForest.load(self.tail_path)
        if not np.array_equal(self.scales, self.tail.scales):
            raise ValueError('world scene scaling differs from frozen C1 tail')
        self.compiled_tail = readonly_c_forest(self.tail)
        for a in (self.scales, self.mean, self.scale, self.delta_scale, self.ridge):
            if not np.isfinite(a).all():
                raise ValueError('finite learned model transforms required')
            a.flags.writeable = False

    def tail_values(self, states):
        options = np.full(len(states), 3)
        F = option_features(states, options)
        leaves = self.compiled_tail.apply(F)
        values = values_from_leaves(self.compiled_tail, leaves)
        supported = (self.tail.independent[np.arange(32)[None, :], leaves, 3] >= 1).sum(1) >= 17
        return values, supported

    def step(self, states, histories, actions):
        X, E = inputs(states, histories, actions)
        Z = (X - self.mean) / self.scale
        delta = (np.column_stack((Z, np.ones(len(Z)))) @ self.ridge + self.residual.predict(Z)) * self.delta_scale
        future, valid = decode_scene(np.asarray(states)[:, :28] + delta, states)
        masses = self.events.predict(E)
        ph = np.clip(masses[:, 0], 0, 1); pe = np.clip(masses[:, 1], 0, 1 - ph)
        hit_discounted = np.clip(masses[:, 2], 0, ph)
        survive_discounted = np.clip(masses[:, 3], 0, 1 - ph - pe)
        history = np.asarray(histories).reshape(-1, HISTORY, 3)
        next_history = np.concatenate((history[:, 1:], np.eye(3)[np.asarray(actions, int)][:, None]), axis=1)
        return future, next_history, hit_discounted, survive_discounted, valid, ph, pe

    def plan(self, state, history, baseline_action):
        state = np.asarray(state, np.float32).reshape(1, 34)
        history = np.asarray(history).reshape(1, HISTORY, 3)
        _, current_support = self.tail_values(state)
        diagnostic = dict(predicted_invalid=0, unsupported_tail=0, expanded_nodes=0,
            root_scores=[None, None, None], current_tail_supported=bool(current_support[0]))
        if not current_support[0]:
            return int(baseline_action), dict(diagnostic, mode='baseline_current_tail_unsupported')
        states = state; histories = history; root = np.array([-1], int)
        gain = np.zeros(1); remaining = np.ones(1)
        for depth in range(HORIZON):
            alive = remaining > 0
            old_terminal = np.flatnonzero(~alive)
            parents = np.repeat(np.flatnonzero(alive), 3)
            actions = np.tile(np.arange(3), int(alive.sum()))
            if len(parents):
                future, future_history, hit, survival, valid, _, _ = self.step(states[parents], histories[parents], actions)
                new_gain = gain[parents] + remaining[parents] * hit
                new_remaining = remaining[parents] * survival
                tail, tail_support = self.tail_values(future)
                diagnostic['expanded_nodes'] += len(parents)
                diagnostic['predicted_invalid'] += int((~valid).sum())
                diagnostic['unsupported_tail'] += int((~tail_support).sum())
                valid |= new_remaining == 0
                valid &= tail_support | (new_remaining == 0)
                new_root = actions if depth == 0 else root[parents]
                # Finished probability mass receives no later reward or tail.
                score = new_gain + new_remaining * tail
                candidate_state = np.concatenate((states[old_terminal], future[valid]))
                candidate_history = np.concatenate((histories[old_terminal], future_history[valid]))
                candidate_root = np.r_[root[old_terminal], new_root[valid]]
                candidate_gain = np.r_[gain[old_terminal], new_gain[valid]]
                candidate_remaining = np.r_[remaining[old_terminal], new_remaining[valid]]
                candidate_score = np.r_[gain[old_terminal], score[valid]]
            else:
                candidate_state, candidate_history, candidate_root = states, histories, root
                candidate_gain, candidate_remaining, candidate_score = gain, remaining, gain
            if not len(candidate_score):
                return int(baseline_action), dict(diagnostic, mode='baseline_no_valid_prediction')
            order = np.lexsort((np.arange(len(candidate_score)), -candidate_score))
            # Keep a candidate for each represented initial primitive command.
            mandatory = [int(order[np.flatnonzero(candidate_root[order] == a)[0]])
                for a in range(3) if np.any(candidate_root == a)]
            chosen = mandatory + [int(i) for i in order if int(i) not in mandatory][:max(0, BEAM - len(mandatory))]
            chosen = np.asarray(chosen, int)
            states, histories, root = candidate_state[chosen], candidate_history[chosen], candidate_root[chosen]
            gain, remaining = candidate_gain[chosen], candidate_remaining[chosen]
            final_score = candidate_score[chosen]
        root_scores = np.full(3, -np.inf)
        for action in range(3):
            if np.any(root == action):
                root_scores[action] = final_score[root == action].max()
        diagnostic['root_scores'] = [float(v) if np.isfinite(v) else None for v in root_scores]
        if not np.isfinite(root_scores[int(baseline_action)]):
            return int(baseline_action), dict(diagnostic, mode='baseline_no_reference_prediction')
        best = int(np.argmax(root_scores))
        action = best if root_scores[best] > root_scores[int(baseline_action)] else int(baseline_action)
        return action, dict(diagnostic, mode='planned_primitive' if action != baseline_action else 'baseline_predicted_best')

    def profile(self, states, histories, baseline_actions):
        timings = []; decisions = []
        for state, history, base in zip(states, histories, baseline_actions):
            began = time.perf_counter(); decisions.append(self.plan(state, history, int(base)))
            timings.append(time.perf_counter() - began)
        return dict(queries=len(timings), mean_seconds=float(np.mean(timings)), max_seconds=float(max(timings)),
            expanded_nodes=[d[1]['expanded_nodes'] for d in decisions])

    def save(self, path):
        np.savez_compressed(path, kind=SCHEMA, scales=self.scales, mean=self.mean, scale=self.scale,
            delta_scale=self.delta_scale, ridge=self.ridge, tail_path=self.tail_path, tail_sha=TAIL_SHA,
            **self.residual.arrays('delta_'), **self.events.arrays('event_'))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['kind']) != SCHEMA or str(data['tail_sha']) != TAIL_SHA:
                raise ValueError('unknown planner or frozen tail')
            return cls(*[data[k] for k in ('scales', 'mean', 'scale', 'delta_scale', 'ridge')],
                CompiledForest.load(data, 'delta_'), CompiledForest.load(data, 'event_'), str(data['tail_path']))

    def frozen_parameters(self):
        return (SCHEMA, HORIZON, BEAM, HISTORY, self.scales, self.mean, self.scale, self.delta_scale,
            self.ridge, self.residual.frozen_parameters(), self.events.frozen_parameters(), TAIL_SHA,
            self.tail.frozen_parameters(), tuple((e.tree_.children_left, e.tree_.children_right,
                e.tree_.feature, e.tree_.threshold, e.tree_.value) for e in self.compiled_tail.estimators_))


class DeltaPlannerController:
    def __init__(self, model):
        self.model = model; self.history = np.zeros((HISTORY, 3))
        self.ordinal = 0; self.past_hit = False; self.pending_action = None

    def choose(self, scene, baseline_action):
        state = state_features(scene, self.ordinal % 5, self.past_hit, self.model.scales)
        if not self.past_hit:
            action = int(baseline_action); decision = dict(mode='baseline_before_hit')
        elif state is None:
            action = int(baseline_action); decision = dict(mode='baseline_invalid_scene')
        else:
            action, decision = self.model.plan(state, self.history, baseline_action)
        self.pending_action = int(action)
        decision.update(action=int(action), baseline_action=int(baseline_action), ordinal=self.ordinal,
            phase=self.ordinal % 5, past_hit=self.past_hit)
        return int(action), decision

    def observe(self, hits, natural_terminal=False):
        if self.pending_action is None:
            raise ValueError('an actual chosen action must precede the observation update')
        self.history = np.concatenate((self.history[1:], np.eye(3)[self.pending_action][None]), axis=0)
        self.pending_action = None; self.ordinal += 1; self.past_hit |= bool(hits)

"""True-online linear Sarsa(lambda), with physical-time discounting.

Gamma on the TD target is the OUTGOING transition discount. Gamma on the
Dutch trace is the INCOMING discount saved from the preceding transition.
Lambda is per decision, not a claim about effective physical credit duration.
"""
import math
import numpy as np
from .policy import FrozenLinearPolicy


class TrueOnlineSarsa:
    def __init__(self, actions, dimension, *, alpha=.01, lam=.9,
                 gamma_per_second=.99, epsilon=.1, seed=0, weights=None):
        if (type(actions) is not int or type(dimension) is not int or min(actions, dimension) < 1
                or not math.isfinite(alpha) or alpha <= 0
                or not 0 <= lam <= 1 or not 0 < gamma_per_second <= 1
                or not 0 <= epsilon <= 1):
            raise ValueError('invalid learner configuration')
        self.weights = np.zeros((actions, dimension), dtype=np.float64)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != self.weights.shape or not np.isfinite(w).all():
                raise ValueError('finite matching initial weights required')
            self.weights[:] = w
        self.alpha, self.lam = alpha, lam
        self.gamma_per_second, self.epsilon = gamma_per_second, epsilon
        self.rng = np.random.default_rng(seed)
        self.transitions = 0
        self.reset_episode()

    def reset_episode(self):
        self.traces = np.zeros_like(self.weights)
        self.q_old = 0.
        self.incoming_discount = 0.
        self.pending_action = None
        self.pending_features = None

    def _features(self, indices, values):
        # Share the transport validator, but never normalize features adaptively.
        FrozenLinearPolicy(self.weights).q_values(indices, values)
        x = np.zeros(self.weights.shape[1], dtype=np.float64)
        x[np.asarray(indices, dtype=np.int64)] = values
        return x

    def _sample(self, x):
        with np.errstate(over='raise',invalid='raise'):
            q = self.weights @ x
        if not np.isfinite(q).all():raise FloatingPointError('nonfinite learner values')
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        ties = np.flatnonzero(q == q.max())
        return int(self.rng.choice(ties))

    def start(self, indices, values):
        if self.pending_action is not None:
            raise ValueError('episode already active; reuse pending action')
        x = self._features(indices, values)
        self.pending_action = self._sample(x)
        self.pending_features = x
        return self.pending_action

    def advance(self, reward, elapsed_seconds, indices=None, values=None, *,
                terminated=False, truncated=False):
        if self.pending_action is None:
            raise ValueError('start an episode before advancing')
        if (not math.isfinite(reward) or not math.isfinite(elapsed_seconds)
                or elapsed_seconds <= 0 or (terminated and truncated)):
            raise ValueError('finite reward/positive duration and exclusive end flags required')
        outgoing = 0. if terminated else self.gamma_per_second ** elapsed_seconds
        # Bootstrap from the actual final observation for truncation, never reset state.
        next_x = None if terminated else self._features(indices, values)
        next_action = None if terminated else self._sample(next_x)
        q_next = 0. if terminated else float(self.weights[next_action] @ next_x)
        x = np.zeros_like(self.weights)
        x[self.pending_action] = self.pending_features
        q = float(np.sum(self.weights*x))
        delta = reward + outgoing*q_next - q
        if not np.isfinite([q,q_next,delta]).all():
            raise FloatingPointError('nonfinite learner target')
        decay = self.incoming_discount*self.lam
        with np.errstate(over='raise',invalid='raise'):
            self.traces *= decay
            self.traces += (1-self.alpha*float(np.sum(self.traces*x)))*x
            if not np.isfinite(self.traces).all():raise FloatingPointError('nonfinite learner traces')
            proposed=self.weights + (self.alpha*(delta+q-self.q_old)*self.traces - self.alpha*(q-self.q_old)*x)
        if not np.isfinite(proposed).all():raise FloatingPointError('nonfinite learner weights')
        self.weights[:]=proposed
        self.transitions += 1
        info = dict(q=q, q_next=q_next, delta=delta, outgoing_discount=outgoing,
                    bootstrap_action=next_action)
        if terminated or truncated:
            self.reset_episode()
        else:
            self.q_old = q_next
            self.incoming_discount = outgoing
            self.pending_features, self.pending_action = next_x, next_action
        return self.pending_action, info

    def frozen(self):
        """Independent evaluator; does not mutate weights, traces or training RNG."""
        return FrozenLinearPolicy(self.weights)

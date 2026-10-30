"""Small reward-only selectors carried across actual Pong games.

No selector receives an environment seed, phase, entity, actuator mapping or
opportunity annotation. A selected anonymous structure owns one complete game.
Own-hit events are successes, negative public terminal scores are failures, and
all other endings censor the unfinished return. This is structural selection
over an inherited controller family, not pixel-to-action learning from scratch.
"""
import copy
import hashlib
import json
import math

import numpy as np

SCHEMA = 'carried-reward-only-structure-selector-v1'


def state_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class CarriedStructureLearner:
    """Two predeclared adaptive rules plus explicit no-retention controls.

    The discounted rule uses smoothed Bernoulli means and a simple optimism
    bonus. It is a fixed heuristic, without a claimed discounted-UCB guarantee.
    """
    def __init__(self, kind, arms=16, random_seed=2026103001, *, evidence_cap=8.,
                 discount=.95, exploration=.15):
        if kind not in ('capped_beta', 'discounted_ucb', 'reset_beta', 'frozen'):
            raise ValueError('unknown selector')
        if type(arms) is not int or arms < 2:
            raise ValueError('at least two anonymous structures required')
        if not (math.isfinite(evidence_cap) and evidence_cap > 0
                and 0 < discount <= 1 and math.isfinite(exploration) and exploration >= 0):
            raise ValueError('invalid fixed learning parameters')
        self.kind, self.arms = kind, arms
        self.evidence_cap, self.discount, self.exploration = map(float,
            (evidence_cap, discount, exploration))
        self.rng = np.random.default_rng(random_seed)
        self.priority = self.rng.permutation(arms).tolist()
        self.success = [0.] * arms
        self.failure = [0.] * arms
        self.lifetime_success = [0] * arms
        self.lifetime_failure = [0] * arms
        self.queue = list(self.priority)
        self.incumbent = self.priority[0]
        self.games = self.outcomes = self.sweeps = self.loss_streak = 0
        self.active_arm = None
        self.choice_mode = None
        self.game_hits = 0
        self.game_seconds = 0.
        self.updates_enabled = kind != 'frozen'

    def scores(self):
        means = [(s + 1.) / (s + f + 2.)
            for s, f in zip(self.success, self.failure)]
        if self.kind == 'discounted_ucb':
            return [m + self.exploration * math.sqrt(
                math.log(2 + self.outcomes) / (1 + s + f))
                for m, s, f in zip(means, self.success, self.failure)]
        return means

    def best_arm(self):
        scores = self.scores()
        return max(self.priority, key=lambda arm: scores[arm])

    def begin_game(self):
        """Choose before seeing this game's initial observation or seed."""
        if self.active_arm is not None:
            raise ValueError('finish the previous actual game first')
        if self.kind == 'reset_beta':
            # Only the experiment's draw generator advances across resets.
            self.success = [0.] * self.arms
            self.failure = [0.] * self.arms
            self.lifetime_success = [0] * self.arms
            self.lifetime_failure = [0] * self.arms
            self.outcomes = self.loss_streak = self.sweeps = 0
            self.priority = self.rng.permutation(self.arms).tolist()
            self.queue = list(self.priority)
            self.incumbent = self.priority[0]
        if not self.updates_enabled:
            arm, mode = self.incumbent, 'frozen_learned_incumbent'
        elif self.queue:
            arm, mode = self.queue.pop(0), 'initial_sweep' if self.sweeps == 0 else 'surprise_sweep'
        else:
            self.incumbent = self.best_arm()
            arm, mode = self.incumbent, 'exploit'
        self.active_arm, self.choice_mode = int(arm), mode
        self.game_hits = 0
        self.game_seconds = 0.
        return self.active_arm, mode

    def _outcome(self, success):
        arm = self.active_arm
        if self.kind == 'discounted_ucb':
            self.success = [s * self.discount for s in self.success]
            self.failure = [f * self.discount for f in self.failure]
        self.lifetime_success[arm] += int(success)
        self.lifetime_failure[arm] += int(not success)
        self.success[arm] += float(success)
        self.failure[arm] += float(not success)
        if self.kind != 'discounted_ucb':
            total = self.success[arm] + self.failure[arm]
            if total > self.evidence_cap:
                factor = self.evidence_cap / total
                self.success[arm] *= factor
                self.failure[arm] *= factor
        self.outcomes += 1

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds):
        """Consume only public feedback from the command just executed."""
        if self.active_arm is None:
            raise ValueError('an active actual game is required')
        if (type(own_hits) is not int or own_hits < 0
                or type(terminated) is not bool or type(truncated) is not bool
                or not math.isfinite(reward) or not math.isfinite(elapsed_seconds)
                or elapsed_seconds < 0):
            raise ValueError('invalid public transition feedback')
        self.game_hits += own_hits
        self.game_seconds += float(elapsed_seconds)
        own_loss = terminated and reward < 0
        if self.updates_enabled:
            for _ in range(own_hits):
                self._outcome(True)
            if own_loss:
                self._outcome(False)
        ended = terminated or truncated
        if ended:
            if self.updates_enabled and self.choice_mode == 'exploit':
                self.loss_streak = self.loss_streak + 1 if own_loss else 0
                if self.kind in ('capped_beta', 'reset_beta') and self.loss_streak >= 2:
                    # Re-examine every anonymous arm. No evidence is cleared and
                    # no arm correspondence is supplied by the intervention.
                    self.queue = self.rng.permutation(self.arms).tolist()
                    self.sweeps += 1
                    self.loss_streak = 0
            elif self.choice_mode in ('initial_sweep', 'surprise_sweep'):
                self.loss_streak = 0
            self.games += 1
            self.active_arm = None
            if self.updates_enabled and not self.queue:
                self.incumbent = self.best_arm()
        return dict(successes=own_hits if self.updates_enabled else 0,
            failures=int(own_loss and self.updates_enabled), ended=ended,
            unfinished_censored=bool(ended and not own_loss),
            own_loss=bool(own_loss), updates_enabled=self.updates_enabled)

    def frozen_branch(self):
        if self.active_arm is not None:
            raise ValueError('branch only at an actual game boundary')
        other = self.from_state(self.state())
        other.kind = 'frozen'
        other.incumbent = self.best_arm()
        other.updates_enabled = False
        return other

    def state(self):
        values = {key: copy.deepcopy(value) for key, value in vars(self).items() if key != 'rng'}
        values.update(schema=SCHEMA, rng_state=copy.deepcopy(self.rng.bit_generator.state))
        return values

    @classmethod
    def from_state(cls, state):
        values = copy.deepcopy(state)
        if values.pop('schema', None) != SCHEMA:
            raise ValueError('unknown carried learner checkpoint')
        rng_state = values.pop('rng_state')
        other = cls(values['kind'], values['arms'], evidence_cap=values['evidence_cap'],
            discount=values['discount'], exploration=values['exploration'])
        if set(values) != set(vars(other)) - {'rng'}:
            raise ValueError('checkpoint fields differ')
        for key, value in values.items():
            setattr(other, key, value)
        for key in ('success', 'failure', 'lifetime_success', 'lifetime_failure'):
            array = getattr(other, key)
            if len(array) != other.arms or any(not math.isfinite(x) or x < 0 for x in array):
                raise ValueError('invalid checkpoint evidence')
        if (sorted(other.priority) != list(range(other.arms))
                or any(type(arm) is not int or not 0 <= arm < other.arms for arm in other.queue)
                or not 0 <= other.incumbent < other.arms
                or (other.active_arm is not None and not 0 <= other.active_arm < other.arms)):
            raise ValueError('invalid checkpoint arm identities')
        other.rng.bit_generator.state = rng_state
        return other

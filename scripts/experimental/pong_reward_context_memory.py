"""One explicit scalar-own-reward context amendment to fixed option memory."""
from pong_option_memory import MemoryController, OptionMemoryEvaluator
from pong_fixed_rbf import FreshRBFEvaluator

SCHEMA = 'own-hit-history-conditioned-option-memory-v1'


class RewardContextController:
    """All stored sites were observed after an own-hit reward in the same scene."""
    def __init__(self, memory):
        self.controller = MemoryController(memory)
        self.has_received_own_hit = False

    def choose(self, scene, baseline_action):
        if not self.has_received_own_hit:
            assert self.controller.remaining == 0
            action = int(baseline_action)
            decision = dict(option=3, mode='baseline_reward_context', hold_before=0,
                hold_after=0, action=action, baseline_action=action,
                has_received_own_hit=False, stored_site_context=True)
            return action, decision
        action, decision = self.controller.choose(scene, baseline_action)
        decision.update(has_received_own_hit=True, stored_site_context=True)
        return action, decision

    def observe_own_reward(self, hits):
        assert type(hits) is int and hits >= 0
        self.has_received_own_hit = self.has_received_own_hit or hits > 0


class RewardContextEvaluator(OptionMemoryEvaluator):
    def __init__(self, original, memory, config):
        super().__init__(original, memory, config)
        self.controller = RewardContextController(memory)
        self.initial_learned_hash = self.learned_hash()

    def learned_hash(self):
        from fly_connectome.control.durable import fingerprint
        return fingerprint((super().learned_hash(), SCHEMA, 'stored_site_context=True',
            'reset_false_then_update_after_actual_scalar_own_hit_reward'))

    def _begin_episode(self):
        self.controller = RewardContextController(self.option_memory)
        FreshRBFEvaluator._begin_episode(self)

    def step(self, *, check_frozen=True):
        record = super().step(check_frozen=check_frozen)
        self.controller.observe_own_reward(int(record['hits']))
        return record

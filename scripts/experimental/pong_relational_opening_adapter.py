"""Same frozen relation from the first valid current observed scene."""
from pong_relational_feedback import RelationalPolicy,RelationalController

CONTRACT='same-frozen-observed-relational-servo-ungated-from-first-valid-scene-v1'
load_model=RelationalPolicy.load


class OpeningController(RelationalController):
    def choose(self,scene,baseline_action):
        decision=self.policy.inspect(scene,self.previous_action)
        action=int(decision['action']) if decision['supported'] else int(baseline_action)
        self.pending_action=action
        decision.update(action=action,baseline_action=int(baseline_action),ordinal=self.ordinal,
            past_hit=self.past_hit,previous_actual_action=self.previous_action,opening_gate_removed=True)
        return action,decision


new_controller=OpeningController


def choose(controller,scene,baseline_action,baseline_q):
    return controller.choose(scene,baseline_action)


def observe(controller,record):
    controller.observe(record['hits'],record['terminated'])

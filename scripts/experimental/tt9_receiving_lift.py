"""Reuse the learned receiving phase inside an alternating-service game."""
from .tt7_contact import ReceivingTiming, TIMINGS
from .tt8_service_player import ServicePlayer


class LiftServicePlayer(ReceivingTiming, ServicePlayer):
    def __init__(self, *args, contact_memory, fixed_lift=None, **kwargs):
        self.contact_memory = contact_memory
        self.fixed_lift = fixed_lift
        self.explore_contact = fixed_lift is None
        self.timings = TIMINGS
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.receives = []

    def _stroke_goals(self, estimate):
        goals = super()._stroke_goals(estimate)
        if self.serve_plan is not None and self.scene.tick <= self.serve_plan[0] + 4:
            return goals
        return self.shape_contact(estimate, goals, self.fixed_lift)

    def finish(self, reward):
        result = super().finish(reward)
        self.score_contacts(reward, self.fixed_lift)
        result['contact_timing'] = self.receives
        return result

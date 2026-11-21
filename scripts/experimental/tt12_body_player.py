"""Keep the opening service fixed while comparing camera-trained receiving timing."""

from .tt6_continuous import PredictiveServo
from .tt6_timing import TimingMemory
from .tt7_flight import FlightPlayer
from .tt8_service_player import ServicePlayer
from .tt8_service_search import SearchPlayer


class ReceivingTiming:
    """Run both state estimators on the same RGB and command history before switching."""

    def __init__(self, *args, receiving_model, service_belief, **kwargs):
        self.receiving_model = receiving_model
        self.service_belief = service_belief
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        # Every toss starts from the same service prior. The receiver keeps its
        # own delay posterior between points, including after an own-serve reply.
        self.service_servo = self.servo
        self.service_servo.observer.memory = TimingMemory(self.service_belief)
        self.service_servo.observer.belief = self.service_servo.observer.memory.probabilities.copy()
        self.receiving_servo = PredictiveServo(self.receiving_model, self.timing_memory, learn=self.adjust)
        self.receive_from = None
        self.visual_role = None
        self.far_sightings = 0

    def update(self, frames, preceding=None):
        self.preceding = preceding
        return super().update(frames, preceding)

    def _stroke_goals(self, estimate):
        if self.receive_from is not None:
            return FlightPlayer._stroke_goals(self, estimate)

        tracker = self.scene.tracker
        actor = tracker.actor
        self.receiving_servo.advance(self.preceding)
        self.receiving_servo.observe(actor, tracker.actor_pitch if actor is not None and actor.observed else None)
        ball = tracker.ball
        tick = self.scene.tick
        if ball is not None and ball.observed:
            y = ball.position[1]
            if self.visual_role is None and tick <= 10:
                if y < -1.0:
                    self.visual_role = 'own'
                elif y > .30:
                    self.visual_role = 'rival'
            if self.visual_role == 'own' and self.serve_plan is not None:
                self.far_sightings = self.far_sightings + 1 if y > .16 else 0
        if self.visual_role == 'rival' or self.far_sightings >= 2:
            self.receive_from = tick
            self.servo = self.receiving_servo
            return FlightPlayer._stroke_goals(self, estimate)
        return super()._stroke_goals(estimate)

    def finish(self, reward):
        result = super().finish(reward)
        result['timing_route'] = dict(visual_role=self.visual_role, first_receiver_frame=self.receive_from)
        return result


class ClassicServePlayer(ReceivingTiming, ServicePlayer):
    """Use an existing optical toss stroke through the same receiving comparison."""


class DriveServePlayer(ReceivingTiming, SearchPlayer):
    """Use a fixed forward-drive profile through the same receiving comparison."""

"""Record raw optical blade and ball sightings during a six-joint return."""

from .tt6_rotary_adapt import CoherentPredictiveServo
from .tt12_body_player import ClassicServePlayer


class RotaryReturnPlayer(ClassicServePlayer):
    def update(self, frames, preceding=None):
        command = super().update(frames, preceding)
        actor = self.scene.tracker.actor
        sighting = actor is not None and actor.observed
        self.trace[-1].update(blade=list(actor.position) if sighting else None,
                              blade_pitch=self.scene.tracker.actor_pitch if sighting else None)
        return command


class DriftCorrectedReturnPlayer(RotaryReturnPlayer):
    """Accept repeated near-impact blade sightings when the fitted pose drifts."""

    def reset(self):
        super().reset()
        self.receiving_servo = CoherentPredictiveServo(self.receiving_model, self.timing_memory, learn=self.adjust)

"""Learn lateral blade offset as well as the timing of a falling-ball strike."""
import numpy as np

from .tt9_precontact import ApproachMemory, ApproachPlayer, REGULARIZATION


class RegistrationMemory(ApproachMemory):
    def lateral_coefficients(self):
        if not self.samples:
            return np.zeros(len(REGULARIZATION))
        design = np.asarray([sample['context'] for sample in self.samples])
        reliability = np.sqrt([sample['weight'] for sample in self.samples])
        design = reliability[:, None] * design
        offsets = reliability * np.array([sample['lateral'] for sample in self.samples])
        return np.linalg.solve(design.T @ design + np.diag(REGULARIZATION),
                               design.T @ offsets)

    def record(self, features, advance, lateral, measurement, reward):
        previous = len(self.samples)
        self.observe(features, advance, measurement, reward)
        if len(self.samples) == previous:
            return
        shift = float(np.clip(measurement['lateral'], -.085, .085))
        gain = .40 if reward > 0 else .85
        self.samples[-1]['lateral'] = float(np.clip(lateral + gain * shift, -.095, .095))


class RegistrationPlayer(ApproachPlayer):
    """Use a learned lateral offset in the calibrated player-eye coordinates."""

    def __init__(self, *args, registration, intervention='full', learn_registration=False, **kwargs):
        if intervention not in ('full', 'timing', 'fixed'):
            raise ValueError('unknown registration mode')
        self.intervention = intervention
        self.learn_registration = learn_registration
        super().__init__(*args, approach=registration,
                         mode='fixed' if intervention == 'fixed' else 'contextual', **kwargs)

    def reset(self):
        super().reset()
        self.lateral = 0.

    def _stroke_goals(self, estimate):
        planned = self.features is not None
        goal, lead, pitch, horizon = super()._stroke_goals(estimate)
        if self.servo.drive is None or self.intervention != 'full':
            return goal, lead, pitch, horizon
        if not planned and self.features is not None:
            self.lateral = float(np.clip(self.features @ self.approach.lateral_coefficients(), -.095, .095))
        goal[0] += self.lateral
        lead[0] += self.lateral
        return goal, lead, pitch, horizon

    def finish(self, reward):
        result = super().finish(reward)
        evidence = result['precontact']
        if evidence is not None:
            evidence['lateral_offset'] = self.lateral
            if self.learn_registration and self.features is not None:
                self.approach.record(self.features, self.timing_lead, self.lateral, evidence['measurement'], reward)
        return result

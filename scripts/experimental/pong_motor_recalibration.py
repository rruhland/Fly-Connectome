"""Re-estimate three motor gains after public evidence of an unsuccessful command map."""
import math

import numpy as np

from pong_categorical_motor_fir import CategoricalMotorController, CategoricalMotorFIR


class BoundedCategoricalMotor(CategoricalMotorFIR):
    """Limit uncertainty in action-history directions with no recent excitation."""
    def observe(self, features, displacement, elapsed_seconds):
        result = super().observe(features, displacement, elapsed_seconds)
        if self.updates_enabled and self.updates % 128 == 0:
            values, axes = np.linalg.eigh(self.covariance)
            values = np.clip(values, 1e-9, self.settings['initial_covariance'])
            self.covariance = (axes * values) @ axes.T
            self.covariance = (self.covariance + self.covariance.T) / 2
        return result


class MotorGainLearner(BoundedCategoricalMotor):
    """Retain the learned time response, then fit a small linear model across buttons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = []
        self.gains = []
        self.gain_covariance = []
        self.probe_sequence = []
        self.calibrations = 0
        self.mismatch_count = 0
        self.empty_losses = 0
        self.game_hits = 0

    def observe(self, features, displacement, elapsed_seconds):
        if not self.kernel:
            result = super().observe(features, displacement, elapsed_seconds)
        else:
            prediction = self.predict(features)
            residual = displacement - prediction
            if self.updates_enabled:
                kernel = np.asarray(self.kernel)
                x = features.reshape(3, -1) @ kernel
                covariance = np.asarray(self.gain_covariance)
                # Bound uncertainty in directions that have seen little excitation.
                forgetting = math.exp(-elapsed_seconds / 3.)
                cross = covariance @ x
                gain = cross / (forgetting + x @ cross)
                covariance = (covariance - np.outer(gain, cross)) / forgetting
                if np.trace(covariance) > 3000:
                    covariance *= 3000 / np.trace(covariance)
                self.gain_covariance = ((covariance + covariance.T) / 2).tolist()
                self.gains = (np.asarray(self.gains) + gain * residual).tolist()
                self.theta = np.outer(self.gains, kernel).ravel()
                self.updates += 1
                self.elapsed_training_seconds += elapsed_seconds
            self.observations += 1
            self.prediction_count += 1
            self.preupdate_squared_error += residual**2
            self.zero_model_squared_error += displacement**2
            result = dict(prediction=prediction, displacement=float(displacement),
                          residual=residual, updated=self.updates_enabled, updates=self.updates)
        scale = max(abs(result['prediction']), abs(displacement))
        if scale >= 2/128 and abs(result['residual']) > .8 * scale:
            self.mismatch_count += 1
        return result

    def feedback(self, hits, reward, terminated, truncated):
        self.game_hits += hits
        if not (terminated or truncated):
            return
        empty_loss = terminated and reward < 0 and self.game_hits == 0
        self.empty_losses = self.empty_losses + 1 if empty_loss else 0
        if (self.updates_enabled and self.updates >= 512 and empty_loss
                and (self.mismatch_count >= 3 or self.empty_losses >= 2)
                and not self.probe_sequence):
            self._calibrate()
        self.game_hits = self.mismatch_count = 0

    def _calibrate(self):
        if not self.kernel:
            taps = self.theta.reshape(3,-1)
            totals = taps.sum(axis=1)
            shape = taps[np.argmax(totals)] - taps[np.argmin(totals)]
            if shape.sum() < .04:
                return
            self.kernel = (shape / shape.sum()).tolist()
            self.gains = totals.tolist()
        self.gain_covariance = (np.eye(3)*600).tolist()
        self.probe_sequence = np.repeat(self.rng.permutation(3), 12).tolist()
        self.probe_remaining = 0
        self.calibrations += 1

    def perturb(self, action, supported):
        if supported and self.probe_sequence:
            self.dither_count += 1
            return self.probe_sequence.pop(0), True
        return super().perturb(action, supported)


class MotorGainController(CategoricalMotorController):
    def observe(self, **kwargs):
        result = super().observe(**kwargs)
        self.learner.feedback(kwargs['own_hits'], kwargs['reward'], kwargs['terminated'], kwargs['truncated'])
        return result

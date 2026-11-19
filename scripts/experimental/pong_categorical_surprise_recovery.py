"""Residual-triggered recovery for a public visual categorical motor response."""

import math

import numpy as np

from pong_categorical_motor_fir import CategoricalMotorFIR, PROBE_HOLD_FRAMES


class SurpriseRecoveryFIR(CategoricalMotorFIR):
    """Increase excitation and response uncertainty after visual prediction drift."""

    fast_error_seconds = .20
    reference_error_seconds = 8.
    minimum_error = 1 / 128
    surprise_ratio = 1.75
    recovery_probe_probability = .5
    recovery_supported_choices = 160

    def __init__(self, random_seed=2026110501, *, settings=None, updates_enabled=True):
        super().__init__(random_seed, settings=settings, updates_enabled=updates_enabled)
        self.fast_abs_error = 0.
        self.reference_abs_error = 0.
        self.recovery_choices_remaining = 0
        self.surprise_episodes = 0

    def observe(self, features, displacement, elapsed_seconds):
        update = super().observe(features, displacement, elapsed_seconds)
        absolute_error = abs(update['residual'])
        fast_weight = -math.expm1(-elapsed_seconds / self.fast_error_seconds)
        self.fast_abs_error += fast_weight * (absolute_error - self.fast_abs_error)

        if self.recovery_choices_remaining:
            return update

        threshold = self.surprise_ratio * max(self.minimum_error, self.reference_abs_error)
        calibrated = self.updates >= 2 * self.settings['initial_dither_updates']
        if calibrated and self.fast_abs_error > threshold:
            self.recovery_choices_remaining = self.recovery_supported_choices
            self.surprise_episodes += 1
            self.fast_abs_error = self.reference_abs_error
            if self.updates_enabled:
                self.covariance = np.eye(len(self.theta)) * self.settings['initial_covariance']
        else:
            reference_weight = -math.expm1(-elapsed_seconds / self.reference_error_seconds)
            self.reference_abs_error += reference_weight * (absolute_error - self.reference_abs_error)
        return update

    def perturb(self, action, supported):
        if not supported or not self.recovery_choices_remaining:
            return super().perturb(action, supported)

        self.recovery_choices_remaining -= 1
        if self.probe_remaining:
            self.probe_remaining -= 1
        elif self.rng.random() < self.recovery_probe_probability:
            self.probe_action = int(self.rng.integers(3))
            self.probe_remaining = PROBE_HOLD_FRAMES - 1
            self.probe_blocks += 1
        else:
            return int(action), False

        self.dither_count += 1
        return self.probe_action, True

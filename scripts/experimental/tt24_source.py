"""Read a Pong-earned visual duration estimate in any co-registered camera coordinates."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


CHECKPOINT = Path(__file__).resolve().parents[2] / 'artifacts' / 'tt24' / 'source-weights.json'
MODES = ('pong', 'cold', 'shuffled')
FEATURES = ('forward_range', 'cross_track_range', 'radial_range', 'relative_closing',
            'relative_miss', 'self_toward_object', 'self_toward_line', 'self_speed')
INTERACTIONS = np.triu_indices(len(FEATURES), k=1)


def visual_features(actor, mover, *, dt, scale):
    """Describe past track geometry in spans and mover-relative travel units.

    Tracks have shape (..., observations, dimensions) and share coordinates and
    timestamps. A static optical anchor is a valid controlled track. Keep all
    spatial coordinates in the same units and supply the apparent half-span of
    the controllable surface in those units. NaN marks an unavailable landmark.
    """
    a, b = (np.asarray(track, float) for track in (actor, mover))
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] < 1 or a.shape[-2] < 4:
        raise ValueError('need paired tracks with at least four observations in shared coordinates')
    if not np.isfinite(dt) or dt <= 0.:
        raise ValueError('camera interval must be positive seconds')
    scale = np.asarray(scale, float)
    if not np.all(np.isfinite(scale) & (scale > 0.)):
        raise ValueError('optical half-span must be positive')
    # Recent observations are sufficient. Use the same physical registration on
    # both tracks so camera roll, origin and a common change of units cancel.
    a, b = (track[..., -5:, :] for track in (a, b))
    t = dt * (np.arange(a.shape[-2], dtype=float) - (a.shape[-2] - 1) / 2.)
    va, vb = (np.einsum('t,...td->...d', t / (t @ t), track) for track in (a, b))
    r = (b[..., -1, :] - a[..., -1, :]) / scale[..., None]
    vb_speed = np.linalg.norm(vb, axis=-1)
    speed = np.maximum(vb_speed, .015 * scale)
    ball = vb / speed[..., None]
    self = va / speed[..., None]
    drift = ball - self
    radial = np.linalg.norm(r, axis=-1)
    along = np.sum(r * ball, axis=-1)
    across_vector = r - along[..., None] * ball
    across = np.linalg.norm(across_vector, axis=-1)
    closing = -np.sum(r * drift, axis=-1) / np.maximum(radial, .02)
    time = np.maximum(0., -np.sum(r * drift, axis=-1) / np.maximum(np.sum(drift**2, axis=-1), .02))
    miss = np.linalg.norm(r + time[..., None] * drift, axis=-1)
    toward = np.sum(r * self, axis=-1) / np.maximum(radial, .02)
    across_toward = np.sum(across_vector * self, axis=-1) / np.maximum(across, .05)
    features = np.stack((-along, across, radial, closing, miss, toward, across_toward,
                         np.linalg.norm(self, axis=-1)), axis=-1)
    return np.where((vb_speed > .015 * scale)[..., None], features, np.nan)


def basis(normalized):
    """Shared smooth basis for a regularized linear readout."""
    x = np.clip(normalized, -4., 4.)
    left, right = INTERACTIONS
    hinges = np.maximum(x[..., :, None] - np.array((-1., 0., 1.)), 0.)
    return np.concatenate((np.ones_like(x[..., :1]), x, x[..., left] * x[..., right],
                           hinges.reshape(x.shape[:-1] + (-1,))), axis=-1)


@lru_cache(maxsize=8)
def _read_weights(path):
    weight = json.loads(path.read_text())
    if weight['schema'] != 'tt24-visual-duration-source-1' or tuple(weight['features']) != FEATURES:
        raise ValueError('unsupported optical duration weights')
    return weight


@dataclass(frozen=True)
class VisualConsequence:
    """Frozen comparative visual duration score learned in native Pong."""

    mode: str
    prior: float
    mean: np.ndarray
    scale: np.ndarray
    core: np.ndarray
    outer: np.ndarray
    weights: np.ndarray

    @classmethod
    def load(cls, mode='pong', path=CHECKPOINT):
        return cls.from_checkpoint(mode, _read_weights(Path(path)))

    @classmethod
    def from_checkpoint(cls, mode, checkpoint):
        if mode not in MODES:
            raise ValueError(f'unknown optical source treatment {mode!r}')
        return cls(mode, checkpoint['prior'], np.asarray(checkpoint['mean']), np.asarray(checkpoint['scale']),
                   np.asarray(checkpoint['core']), np.asarray(checkpoint['outer']),
                   np.asarray(checkpoint['weights'][mode]))

    def score_features(self, features, *, details=False):
        features = np.asarray(features, float)
        valid = np.isfinite(features).all(axis=-1)
        safe = np.where(valid[..., None], features, self.mean)
        # Queries outside the native duration sample receive progressively less
        # source influence, including under a matched shuffled checkpoint.
        low, high = self.core
        outside_low, outside_high = self.outer
        width = np.maximum(np.maximum(low - outside_low, outside_high - high), 1e-6)
        excess = np.maximum(np.maximum(low - safe, safe - high), 0.) / width
        support = np.where(valid, np.clip(1. - np.max(excess, axis=-1), 0., 1.), 0.)
        estimate = np.clip(basis((safe - self.mean) / self.scale) @ self.weights, .001, .999)
        score = self.prior + support * (estimate - self.prior)
        if not details:
            return score
        return dict(score=score, support=support, features=features)

    def score_history(self, actor, mover, *, dt, scale, details=False):
        """Score continuation from actual optical track history available now."""
        return self.score_features(visual_features(actor, mover, dt=dt, scale=scale), details=details)

    def predict(self, actor, mover, *, dt, scale):
        """Score synchronized future visual hypotheses with no outcome access."""
        a, b = (np.asarray(track, float) for track in (actor, mover))
        if a.shape != b.shape or a.ndim < 2 or a.shape[-2] < 4:
            raise ValueError('need at least four paired forecast positions')
        width = min(a.shape[-2], 5)
        last = tuple(np.swapaxes(np.lib.stride_tricks.sliding_window_view(track, width, axis=-2), -1, -2)
                     for track in (a, b))
        scored = self.score_history(*last, dt=dt, scale=np.asarray(scale)[..., None], details=True)
        delta = scored['score'] - self.prior
        return self.prior + np.sum(delta, axis=-1) / np.maximum(1., np.sum(scored['support'] > 0., axis=-1))

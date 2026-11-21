"""Rank RGB role components by shape, metric feasibility and recent motion."""

import numpy as np

from .tt19_view import _in_court


_ROLE_COLORS = {'ball': ((.065, .044, 61., 1),),
                'blade': ((.993, .027, 57., 4), (.878, .030, 68., 4))}


def _candidates(colors, role):
    height = colors.shape[0]
    limit = max(10, height // 3) if role == 'ball' else height
    candidates = []
    for hue, tolerance, chroma, min_area in _ROLE_COLORS[role]:
        for blob in colors.components(hue, tolerance=tolerance, minimum_chroma=chroma,
                                      min_area=min_area, max_span=limit):
            pixels = blob.pixels
            if len(pixels) < (5 if role == 'ball' else 16):
                continue
            delta = np.abs(colors.hue[pixels[:, 1], pixels[:, 0]] - hue)
            mismatch = float(np.median(np.minimum(delta, 1. - delta))) / tolerance
            candidates.append((mismatch, -float(blob.weights.sum()), blob))
    return sorted(candidates, key=lambda value: value[:2])[:8]


def _prediction(camera, path, tick, frame_seconds, role):
    if path is None or not 0 < tick - path.tick <= 4:
        return None
    speed = path.speed.copy()
    limit = (12. if role == 'ball' else 5.) * frame_seconds
    speed *= min(1., limit / max(limit, np.linalg.norm(speed)))
    position = np.asarray(path.position) + speed * (tick - path.tick)
    projected = camera.project(position)
    if projected[2] <= .04:
        return None
    return projected[:2]


def choose_component(colors, role, camera, fitter, bounds, path, tick, frame_seconds, fallback):
    """Pick among same-color alternatives using only current and preceding RGB estimates."""
    if camera is None:
        return fallback
    width, length = bounds
    predicted = _prediction(camera, path, tick, frame_seconds, role)
    scored = []
    for mismatch, _, blob in _candidates(colors, role):
        fit = fitter(camera, blob)
        spot = None if fit is None else fit[0]
        if not _in_court(role, spot):
            continue
        x, y, height = spot
        # The local play region is a soft prior. Long shots remain eligible.
        lane = max(0., abs(x) - width - .20) + max(0., abs(y) - length - .45)
        lane += max(0., height - 1.0)
        span = float(max(np.ptp(blob.pixels, axis=0)))
        motion = (0. if predicted is None else
                  min(2., np.linalg.norm(blob.center - predicted) / max(16., 1.7 * span)))
        score = mismatch + .9 * lane + .30 * motion
        if role == 'blade':
            score += max(0., y + .28) + max(0., height - .70)
        scored.append((score, blob))
    return min(scored, key=lambda candidate: candidate[0])[1] if scored else fallback

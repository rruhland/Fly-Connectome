"""Update a clamped robot's nominal table footprint from visible court lines."""

import numpy as np
from scipy.optimize import least_squares

from .tt4_arena import TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_HEIGHT
from .tt29_registration import _COURT_LINES, _line, _pixels, _project


def survey(frames, rig):
    measurements = []
    for stream, image in frames.items():
        camera = rig.nominal.get(stream)
        if camera is None:
            continue
        expected = _project(_COURT_LINES, camera)
        middle = _project(np.array((0.0, 0.0, TABLE_HEIGHT)), camera)
        pixels, cloth = _pixels(image)
        fits = [
            _line(a, b, middle, pixels, cloth, image.shape[0] / 288.0, front=i == 2)
            for i, (a, b) in enumerate(expected)
        ]
        measurements.append(
            (camera, np.stack([f[0] for f in fits]), np.array([f[1] for f in fits]))
        )
    if len(measurements) < 2:
        raise ValueError(
            "Two fixed views are required to distinguish a footprint change"
        )
    original = np.array([TABLE_HALF_WIDTH, TABLE_HALF_LENGTH])

    def errors(footprint):
        lines = _COURT_LINES.copy()
        lines[..., :2] *= footprint / original
        return np.concatenate(
            [
                (
                    np.einsum("lpi,li->lp", _project(lines, cam), axes)
                    - offsets[:, None]
                ).ravel()
                for cam, axes, offsets in measurements
            ]
        )

    fitted = least_squares(
        errors, original, bounds=(original * 0.75, original * 1.25), loss="soft_l1"
    )
    rms = float(np.sqrt(np.mean(errors(fitted.x) ** 2)))
    if rms > 1.2:
        raise ValueError("The two cameras disagree about the table footprint")
    return fitted.x, rms

"""RGB outline should follow controlled surfaces when they match static scenery."""

import numpy as np
import pytest

pytest.importorskip('mujoco')
from scripts.experimental.tt5_vision import _face


def test_motion_supported_outline_excludes_matching_room_surface():
    background = np.full((48, 80, 3), (48, 93, 32), dtype=np.int16)
    paint = np.array((75, 125, 177))
    background[13:30, 5:27] = paint
    image = background.copy()
    y, x = np.indices(image.shape[:2])
    image[(x - 52)**2 + (y - 24)**2 <= 7**2] = paint

    aperture = (slice(0, 48), slice(0, 80))
    raw = _face(image, background, paint, None, aperture)
    changed = _face(image, background, paint, None, aperture, motion_supported=True)

    np.testing.assert_allclose(raw.center, (15.5, 21.))
    np.testing.assert_allclose(changed.center, (52., 24.))

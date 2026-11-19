"""Camera-coordinate contact projection checks."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_contact_geometry import CAMERA_PIXEL, project_contact


def test_projection_advances_contact_to_visible_facing_edge():
    history = [(t, .04 + .2 * t, .4 - .5 * t) for t in (0., .02, .04, .06, .08, .10, .12)]
    intercept = .04 + .2 * (.12 + .6)
    context = SimpleNamespace(intercept=intercept, seconds=.6)
    projection = project_contact(history, context, .12)
    assert projection.position == pytest.approx(intercept - .2 * CAMERA_PIXEL / .5)
    assert CAMERA_PIXEL / 2 <= projection.uncertainty <= CAMERA_PIXEL


def test_stale_camera_sample_uses_original_intercept():
    history = [(t, .04 + .2 * t, .4 - .5 * t) for t in (0., .02, .04, .06, .08)]
    context = SimpleNamespace(intercept=.25, seconds=.6)
    projection = project_contact(history, context, .2)
    assert projection.position == .25
    assert projection.uncertainty == CAMERA_PIXEL

"""Scoped opt-in for standard serial training. No pinned runner is modified."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from . import tt29_partner, tt33_joint
from .tt29_fast_camera import CachedStereoRenderer
from .tt54_opponent_approx import BASE, approximate
from .tt54_packed_stereo import PackedStereo


@contextmanager
def enabled(*, cameras=True, opponent=True):
    """Keep other renderer and opponent variants unchanged."""
    with ExitStack() as stack:
        if cameras and tt33_joint.CachedStereoRenderer is CachedStereoRenderer:
            stack.enter_context(patch.object(tt33_joint,"CachedStereoRenderer",PackedStereo))
        if opponent and tt29_partner._flight_approx is BASE:
            stack.enter_context(patch.object(tt29_partner,"_flight_approx",approximate))
        yield

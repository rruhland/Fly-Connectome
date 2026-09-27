"""Approved M1A.5 hybrid visual state and locally learned spatial forecasts."""

from importlib.resources import as_file, files

from .state import ProbabilisticVisualState
from .interface import VisualStateEncoder

__all__ = ['ProbabilisticVisualState', 'VisualStateEncoder', 'load_default', 'load_legacy_default']


def load_default():
    """Load the approved CPU baseline into a fresh 64x64 scene."""
    return load_legacy_default()


def load_legacy_default():
    """Load the unchanged version-1 prior for historical comparisons."""
    with as_file(files(__package__).joinpath('data/default.pt')) as path:
        return ProbabilisticVisualState.load(path)

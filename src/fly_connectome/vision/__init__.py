"""Approved M1A.5 hybrid visual state and locally learned spatial forecasts."""

from importlib.resources import as_file, files

from .state import ProbabilisticVisualState

__all__ = ['ProbabilisticVisualState', 'load_default']


def load_default():
    """Load the approved CPU baseline into a fresh 64x64 scene."""
    with as_file(files(__package__).joinpath('data/default.pt')) as path:
        return ProbabilisticVisualState.load(path)

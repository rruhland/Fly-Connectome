"""Gap-exit, quiet, visible, and post-gap targets are distinct frames."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trace_phase_conflict import phase_indices


def test_phase_indices_cover_causal_forecast_targets():
    assert phase_indices((7, 8, 9)) == dict(
        exit=9, quiet=8, visible=5, post=10)
    assert phase_indices((6, 7)) == dict(
        exit=7, quiet=6, visible=4, post=8)
    assert phase_indices((8, 9, 10, 11)) == dict(
        exit=11, quiet=10, visible=6, post=12)

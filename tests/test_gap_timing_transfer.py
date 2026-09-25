"""The frozen timing suite spans unseen generic shapes and gap durations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gap_timing_transfer import WINDOWS, timing_cases


def test_timing_cases_cover_distinct_untrained_windows():
    cases = timing_cases()
    assert len(cases) == 144
    for window, hidden in WINDOWS.items():
        rows = [row for row in cases if row[0] == window]
        assert len(rows) == 48
        assert len({row[1] for row in rows}) == 3
        assert len({row[2] for row in rows}) == 4
        assert len({row[3] for row in rows}) == 2
        assert len({row[4] for row in rows}) == 2
        assert all(row[5] == hidden for row in rows)
        assert hidden[-1]+3 < 15

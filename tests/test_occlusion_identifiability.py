"""Visually identical prefixes can have different future events."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from occlusion_identifiability import compare_pair, matched_prefix_suite


def test_equal_visual_history_does_not_determine_reappearance_time():
    rows = matched_prefix_suite()[:1]
    first = compare_pair(rows, 0, 1, through=8)
    second = compare_pair(rows, 1, 2, through=9)
    for result in (first, second):
        assert result['identical_prefixes'] == 1
        assert result['different_targets'] == 1
        assert result['target_disagreement_pixels'] > 0

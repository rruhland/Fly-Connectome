"""Threshold-free ranking detects informative and uninformative output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_trace_ranking import ranking


def test_ranking_separates_event_and_quiet_scores():
    assert ranking([.8, .9], [.1, .2])['event_vs_quiet_auc'] == 1
    assert ranking([0, 0], [0, 0])['event_vs_quiet_auc'] == .5

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from live_local_evaluate import safe_score


def test_quiet_only_and_event_only_phases_report_without_nonfinite_values():
    quiet = safe_score(np.zeros(3), np.array([0., .2, 0.]))
    assert quiet['on_anticipation'] is None
    assert quiet['off_anticipation'] is None
    assert quiet['quiet_alarm_fraction'] == 1/3
    event = safe_score(np.array([-1., 1.]), np.array([-.5, .4]))
    assert event['quiet_alarm_fraction'] is None
    assert event['quiet_mse'] is None

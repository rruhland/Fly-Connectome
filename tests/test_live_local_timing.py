import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from live_local_timing import mixed_boundaries, score_run, choice
from temporal_visual import recurrent_boundaries


def test_mixed_boundaries_match_fixed_protocol_when_dwells_equal():
    blanks = np.array([12, 16, 20])
    dwells = np.array([3, 3, 3])
    np.testing.assert_array_equal(mixed_boundaries(blanks, dwells),
                                  recurrent_boundaries(blanks, dwell=3)//8)


def test_mixed_boundaries_keep_each_future_frame_after_first_cycle():
    issues = mixed_boundaries(np.array([2, 3]), np.array([2, 4]))
    lengths = np.array([2+8*2+1, 3+8*4+1])
    assert issues[0] == 0
    assert issues[-1] == lengths.sum()-2
    assert len(issues) == lengths.sum()-2*(2+4)-1


def test_score_uses_next_frame_target_and_issued_prediction():
    data = dict(target=np.array([0., -1., 0., 1.]),
                gated=np.array([-.8, .3, .9, 0.]))
    result = score_run(data, np.array([0, 1, 2]))
    assert result['learned']['on_anticipation'] == .8
    assert result['learned']['off_anticipation'] == .9
    assert result['persistence']['off_anticipation'] == 0


def test_validation_choice_requires_both_signs_and_quiet_limit():
    def candidate(on, off, quiet, mse):
        return dict(validation=dict(learned=dict(on_anticipation=on,
            off_anticipation=off, false_alarm_fraction=quiet, mse=mse)))
    rows = {'0.1':candidate(.8,.09,.01,.02),
            '1.0':candidate(.2,.2,.04,.09)}
    assert choice(rows) == '1.0'
    rows['0.3'] = candidate(.5,.5,.06,.01)
    assert choice(rows) == '1.0'

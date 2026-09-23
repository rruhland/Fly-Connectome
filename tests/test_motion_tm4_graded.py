import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_tm4_graded import roc_auc


def test_roc_auc_counts_tied_scores_as_half_a_pair():
    scores = np.array([0., 1., 1., 2.])
    events = np.array([False, False, True, True])
    assert roc_auc(scores, events) == .875

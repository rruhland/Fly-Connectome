import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_lc11_wiring import class_summary, summarize_side


def test_class_summary_counts_contacts_and_reached_targets():
    pre = np.array([1, 1, 2, 3])
    post = np.array([10, 11, 10, 10])
    contacts = np.array([4, 2, 5, 20])
    result = class_summary(pre, post, contacts, np.array([10, 11]),
                           np.array([1, 2]))
    assert result['edges'] == 3
    assert result['edges_at_least_3'] == 2
    assert result['contacts'] == 11
    assert result['targets_reached'] == 2
    assert result['median_contacts_per_reached_target'] == 5.5


def test_right_roster_and_left_same_side_are_distinct():
    annotations = {
        1: dict(type='T2', somaSide='R', status='Traced'),
        2: dict(type='T3', somaSide='L', status='Traced'),
        3: dict(type='T3', somaSide='R', status='Traced'),
        10: dict(type='LC11', somaSide='R', status='Traced'),
        11: dict(type='LC11', somaSide='L', status='Traced'),
    }
    pre = np.array([1, 2, 3, 1])
    post = np.array([10, 11, 10, 11])
    contacts = np.array([5, 7, 9, 1])
    right = summarize_side(annotations, np.array([1, 3]), pre, post,
                           contacts, 'R')
    left = summarize_side(annotations, np.array([1, 3]), pre, post,
                          contacts, 'L')
    assert right['selected_input']['contacts'] == 14
    assert right['same_side_source_classes']['T2']['contacts'] == 5
    assert right['same_side_source_classes']['T3']['contacts'] == 9
    assert left['selected_input'] is None
    assert left['same_side_source_classes']['T3']['contacts'] == 7
    assert left['total_contacts'] == 8

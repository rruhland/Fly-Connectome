"""Affinity arms are scored on the same event evidence and live files."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_local_affinity_binding import (affinity_report, audit_generic, audit_stress,
                                        evaluate_forecasts, evaluate_state,
                                        file_counts)
from run_decisive_representation_audit import make_cases
from run_local_motion_assemblies import case_events


def diagonal_event(y, x):
    event = torch.zeros((2, 32, 64))
    event[1, y, x] = 1.
    event[1, y+1, x+1] = 1.
    return event


LINKS = {(-1., 1): True, (-1., -1): True,
         (1., 1): True, (1., -1): True}


def test_each_affinity_arm_scores_the_same_next_event():
    rows = evaluate_forecasts([[diagonal_event(5, 5),
                                diagonal_event(5, 6)]],
                              {'aligned': LINKS, 'shuffled': {}})
    assert all(row['events'] == 1 and row['targets'] == 2
               for row in rows.values())


def test_live_file_count_reflects_diagonal_binding():
    rows = file_counts([diagonal_event(5, 5)],
                       {'aligned': LINKS, 'shuffled': {}})
    assert rows['four'] == [2]
    assert rows['eight'] == [1]
    assert rows['aligned'] == [1]
    assert rows['shuffled'] == [2]


def test_generic_audit_scores_selected_files_without_oracle_input():
    rows = audit_generic([[diagonal_event(5, 5)]], [1],
                         {'aligned': LINKS, 'shuffled': {}})
    assert rows[1]['four']['wrong_count'] == 1
    assert rows[1]['aligned']['wrong_count'] == 0


def test_clean_state_audit_scores_all_arms_on_same_case():
    case = make_cases()[0]
    rows = evaluate_state([case], [case_events(case)],
                          {'aligned': LINKS, 'shuffled': {}})
    assert rows['calibration']['four']['frames'] == 9
    assert rows['calibration']['aligned']['frames'] == 9


def test_stress_audit_counts_active_frames_for_each_arm():
    events = [diagonal_event(5, 5) for _ in range(4)]
    rows = audit_stress({'pair': (events, 1, range(4))},
                        {'aligned': LINKS, 'shuffled': {}})
    assert rows['pair']['four']['frames'] == 4
    assert rows['pair']['aligned']['exact_count'] == 4


def test_affinity_report_preserves_links_in_json_result():
    saved = json.loads(json.dumps(affinity_report(
        {'aligned': LINKS, 'shuffled': {}, 'counts': {}})))
    assert saved['aligned']['+1,+1']
    assert not saved['shuffled']['+1,+1']

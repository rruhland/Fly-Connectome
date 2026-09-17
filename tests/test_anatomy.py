import pytest

from fly_connectome.anatomy import anatomy_report, choose_threshold, select_roster
from fly_connectome.graph import Graph


def graph():
    return Graph.from_contacts([10, 20, 30, 40, 50, 60],
        [10, 20, 30, 30, 40, 50], [20, 30, 20, 40, 50, 40],
        [5, 3, 5, 5, 1, 1], [1]*6, 0.1)


def test_report_counts_reachability_regions_and_recurrence():
    report = anatomy_report(graph().threshold(3), [10], [40],
                            ['lamina', 'medulla', 'motion', 'motor', 'other', 'other'])
    assert report['contacts'] == 18
    assert report['reachable_motor_ids'] == [40]
    assert report['recurrent_components'] == [[20, 30]]
    assert report['isolated_body_ids'] == [50, 60]


def test_canonical_threshold_is_chosen_only_from_anatomy():
    assert choose_threshold(graph(), [10], [40],
        ['lamina', 'medulla', 'motion', 'motor', 'other', 'other'],
        ['lamina', 'medulla', 'motion', 'motor']) == 3
    with pytest.raises(ValueError, match='preregistered circuit'):
        choose_threshold(graph(), [10], [60], ['x']*6, ['x'])


def test_path_selection_recurrence_and_size_cap():
    # Paths 10->20->30->40; 50 is a weak reciprocal partner of 40.
    assert select_roster(graph(), [10], [40], max_hops=3,
                         reciprocal_contacts=3, scc_cap=1) == [10, 20, 30, 40]
    assert select_roster(graph(), [10], [40], max_hops=3,
                         reciprocal_contacts=3, scc_cap=2) == [10, 20, 30, 40, 50]
    assert select_roster(graph(), [10], [40], max_hops=2,
                         reciprocal_contacts=3, scc_cap=1) == [10, 40]

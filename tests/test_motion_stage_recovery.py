import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_stage_recovery import annotated_columns, choose_current
from fly_connectome.sensor import Retina


def test_recovery_selection_requires_local_change_arrival_and_quiet_blank():
    rows = [dict(rest_current=value, changed_local_cells=changed,
                 target_arrivals=arrivals, blank_rate=rate, finite=True)
            for value, changed, arrivals, rate in (
                (0., 0, 0, 0.), (.85, 6, 0, 0.), (.95, 6, 2, .02),
                (1.05, 5, 1, .009), (1.15, 8, 4, .008))]
    assert choose_current(rows) == 1.05
    assert choose_current(rows[:3]) is None


def test_recovery_columns_follow_measured_annotations():
    retina = Retina(2, 2, [[0, 0], [1, 0]], [0, -1, -1],
                    ['L1', 'Mi4', 'Tm9'], {'L1': 'on'})
    metadata = dict(graph=dict(body_ids=[10, 20, 30]),
                    retina=dict(cell_types=['L1', 'Mi4', 'Tm9']))
    annotations = pa.table(dict(bodyId=[20, 30],
                                assignedOlHex1=[0., 1.],
                                assignedOlHex2=[0., 0.]))
    assert annotated_columns(metadata, retina, annotations).tolist() == [-1, 0, 1]

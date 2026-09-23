import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_t3_target import infer_t3_columns, select_current, static_dot
from fly_connectome.sensor import Retina


def test_t3_locality_uses_measured_mi1_tm1_inputs():
    types = ['L1', 'Mi1', 'Tm1', 'T3', 'T3']
    retina = Retina(2, 2, [[0, 0], [1, 0]], [0, -1, -1, -1, -1],
                    types, {'L1': 'on'})
    metadata = dict(retina=dict(cell_types=types),
                    graph=dict(body_ids=[10, 20, 30, 40, 50],
                               pre=[1, 2], post=[3, 4], contacts=[5, 7]))
    annotations = pa.table(dict(bodyId=[20, 30],
                                assignedOlHex1=[0., 1.],
                                assignedOlHex2=[0., 0.]))
    inferred, mass = infer_t3_columns(metadata, retina, annotations)
    assert inferred.tolist() == [-1, -1, -1, 0, 1]
    assert mass.tolist() == [0, 0, 0, 5, 7]


def test_object_column_inference_can_use_a_different_measured_afferent():
    types = ['Tm2', 'T2']
    retina = Retina(2, 2, [[0, 0]], [0, -1], types, {})
    metadata = dict(retina=dict(cell_types=types),
                    graph=dict(body_ids=[20, 40], pre=[0], post=[1],
                               contacts=[6]))
    annotations = pa.table(dict(bodyId=[20], assignedOlHex1=[0.],
                                assignedOlHex2=[0.]))
    inferred, mass = infer_t3_columns(
        metadata, retina, annotations, target_type='T2',
        source_types=('Tm2', 'Mi1'))
    assert inferred.tolist() == [-1, 0]
    assert mass.tolist() == [0, 6]


def test_t3_calibration_requires_both_polarities_and_quiet_blank():
    rows = [dict(rest_current=0., on_excess=0, off_excess=0,
                 blank_rate=0., finite=True),
            dict(rest_current=.85, on_excess=6, off_excess=4,
                 blank_rate=0., finite=True),
            dict(rest_current=.95, on_excess=5, off_excess=7,
                 blank_rate=.005, finite=True)]
    assert select_current(rows) == .95
    assert select_current(rows[:2]) is None


def test_static_dot_stays_at_one_position():
    images = static_dot(18, 'on')
    assert all(image.equal(images[8]) for image in images[2:15])
    assert not images[0].any() and not images[-1].any()

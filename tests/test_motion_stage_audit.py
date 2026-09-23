import sys
from pathlib import Path

import torch
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_stage_audit import frames_for_condition
from motion_stage_locality import infer_columns
from fly_connectome.sensor import Retina


def test_opposing_dot_traversals_have_identical_occupancy_and_event_counts():
    right = frames_for_condition(18, 1, 'on')
    left = frames_for_condition(18, -1, 'on')
    assert len(right) == len(left) == 18
    assert not right[0].any() and not right[1].any()
    assert all(torch.equal(a, b) for a, b in zip(right[2:15], reversed(left[2:15])))
    assert [int((a != b).sum()) for a, b in zip(right[1:], right[:-1])] == [
        int((a != b).sum()) for a, b in zip(left[1:], left[:-1])]
    off = frames_for_condition(18, 1, 'off')
    assert all(torch.equal(~a, b) for a, b in zip(right, off))


def test_opposing_bar_traversals_have_identical_events():
    right = frames_for_condition(46, 1, 'on', kind='bar')
    left = frames_for_condition(46, -1, 'on', kind='bar')
    assert all(torch.equal(a, b) for a, b in zip(right[2:15], reversed(left[2:15])))
    assert [int((a != b).sum()) for a, b in zip(right[1:], right[:-1])] == [
        int((a != b).sum()) for a, b in zip(left[1:], left[:-1])]
    assert int(right[2].sum()) == 39


def test_motion_cell_locality_uses_measured_direct_afferent_columns():
    types = ['L1', 'Mi1', 'Tm1', 'T4a', 'T5a']
    retina = Retina(2, 2, [[0, 0], [1, 0]], [0, -1, -1, -1, -1],
                    types, {'L1': 'on'})
    metadata = dict(graph=dict(body_ids=[10, 20, 30, 40, 50],
                               pre=[1, 2], post=[3, 4], contacts=[5, 7]),
                    retina=dict(cell_types=types))
    annotations = pa.table(dict(bodyId=[20, 30],
                                assignedOlHex1=[0., 1.],
                                assignedOlHex2=[0., 0.]))
    inferred, mass = infer_columns(metadata, retina, annotations)
    assert inferred.tolist() == [-1, -1, -1, 0, 1]
    assert mass.tolist() == [0, 0, 0, 5, 7]

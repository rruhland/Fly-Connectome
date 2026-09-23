import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from motion_stage_timing import frames, infer_tm3_columns
from fly_connectome.sensor import Retina


def test_tm3_column_inference_uses_only_measured_l1_contacts():
    types = ['L1', 'L1', 'Tm3', 'Tm3']
    retina = Retina(2, 2, [[0, 0], [1, 0]], [0, 1, -1, -1], types, {'L1': 'on'})
    metadata = dict(retina=dict(cell_types=types, neuron_columns=[0, 1, -1, -1]),
                    graph=dict(pre=[0, 1, 1], post=[2, 2, 3], contacts=[3, 1, 2]))
    inferred, mass, spread = infer_tm3_columns(metadata, retina)
    assert inferred.tolist() == [-1, -1, 0, 1]
    assert mass.tolist() == [0, 0, 4, 2]
    assert spread[2] > 0 and spread[3] == 0


def test_timing_probe_has_matched_opposed_sequences():
    right = frames(18, 'right')
    left = frames(18, 'left')
    assert len(right) == len(left) == 8
    assert all(a.equal(b) for a, b in zip(right[2:5], reversed(left[2:5])))
    assert all(not image.any() for image in frames(18, 'blank'))

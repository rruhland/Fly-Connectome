import numpy as np

from scripts.experimental.tt20_carry_serves import paired
from scripts.experimental.tt20_stroke_carry import StrokeValues, restore


def test_switch_extends_existing_actuator_values():
    first = StrokeValues(2)
    for i, merit in enumerate((.5, -.4, 1., 1.4)):
        first.record(i, merit)
    before = first.state()
    next_eye = restore({'state': before}, 2)
    for i, merit in enumerate((.4, 2.5, .1, .1)):
        next_eye.record(i, merit)
    assert before == first.state()
    assert np.allclose(next_eye.total, [.9, 2.1, 1.1, 1.5])
    assert next_eye.state()['pick'] == 1
    assert next_eye.state()['count'] == [2] * 4


def test_paired_score_and_service_can_differ():
    def outcome(won, legal):
        return {'reward': 1 if won else -1,'outcome': {'legal_serve': legal}}
    rows = [{'0': outcome(False, True),'1': outcome(True, True)},
            {'0': outcome(False, False),'1': outcome(False, True)}]
    assert paired(rows, 0, 1, 'wins')['selected'] == 1
    assert paired(rows, 0, 1, 'legal')['selected'] == 2

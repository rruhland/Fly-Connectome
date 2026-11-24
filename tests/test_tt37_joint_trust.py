import numpy as np

from scripts.experimental.tt37_joint_trust import TrustField


class Field:
    def __init__(self, target):
        self.target = np.full(6, target)

    def get(self, optical, q, *, reference=None):
        return self.target, 1.


ARRIVAL = np.array([.1, -1.0, 1.1, 0., -3.1, .8])


def test_fixed_gate_does_not_change_first_returns():
    field = TrustField(Field(1.), Field(0.), "gate")
    q = np.zeros(6)
    for score in (0, 0):
        field.score_feedback(score)
        target, conf = field.get(ARRIVAL, q)
        assert conf == 1.
        np.testing.assert_equal(target, 1.)
    field.score_feedback(1)
    target, conf = field.get(ARRIVAL, q)
    assert conf == 1.
    assert np.all(target < 1.)
    assert not field.records


def test_live_reward_labels_only_later_observed_approaches():
    field = TrustField(Field(1.), Field(0.), "active")
    q = np.zeros(6)
    field.score_feedback(1)
    assert not field.records
    field.get(ARRIVAL, q)
    field.score_feedback(2)
    assert [row["reward"] for row in field.records] == [1]
    field.get(ARRIVAL, q)
    field.end_point(terminal=False)
    assert [row["reward"] for row in field.records] == [1]
    field.end_point(terminal=True)
    assert [row["reward"] for row in field.records] == [1, 0]
    assert [row["coherent"] for row in field.records] == [0, 1]


def test_carried_checkpoint_can_select_raw_without_learning_at_test():
    evidence = [{"state": ARRIVAL.tolist(), "coherent": coherent, "reward": int(not coherent)}
                for coherent in (0, 1) for _ in range(3)]
    field = TrustField(Field(1.), Field(0.), "carry", evidence)
    field.score_feedback(1)
    target, weight = field.get(ARRIVAL, np.zeros(6))
    assert weight == 1.
    np.testing.assert_equal(target, 1.)
    field.score_feedback(2)
    assert field.records == evidence

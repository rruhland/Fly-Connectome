import numpy as np
import pytest

from fly_connectome.graph import Graph


def fixture():
    # Ordered contact rows: reverse edges are independent; body 40 is isolated.
    return Graph.from_contacts(
        body_ids=[10, 20, 30, 40],
        pre=[10, 10, 20, 20, 30], post=[20, 20, 10, 30, 20],
        counts=[2, 3, 1, 3, 5], signs=[1, -1, 1, 1], gain=0.2,
    )


def test_contact_aggregation_direction_and_linear_weights():
    g = fixture()
    assert g.pre.tolist() == [0, 1, 1, 2]
    assert g.post.tolist() == [1, 0, 2, 1]
    assert g.contacts.tolist() == [5, 1, 3, 5]
    np.testing.assert_allclose(g.initial_weights(), [1, -0.2, -0.6, 1])


def test_nested_thresholds_keep_isolated_neurons():
    g = fixture()
    variants = [g.threshold(t) for t in (1, 3, 5)]
    assert [len(v.pre) for v in variants] == [4, 3, 2]
    for v in variants:
        assert v.body_ids.tolist() == [10, 20, 30, 40]
    assert variants[2].pre.tolist() == [0, 2]
    assert variants[2].post.tolist() == [1, 1]


def test_graph_is_immutable_and_rejects_invalid_anatomy():
    g = fixture()
    with pytest.raises(ValueError):
        g.pre[0] = 3
    for changes in ({"signs": [0, 1]}, {"counts": [-1]}, {"pre": [99]},
                    {"body_ids": [10, 10]}, {"gain": float("nan")}):
        args = dict(body_ids=[10, 20], pre=[10], post=[20], counts=[1],
                    signs=[1, -1], gain=0.1)
        args.update(changes)
        with pytest.raises(ValueError):
            Graph.from_contacts(**args)


def test_empty_graph_retains_roster():
    g = Graph.from_contacts([10], [], [], [], [1], 0.1)
    assert g.threshold(5).body_ids.tolist() == [10]
    assert g.initial_weights().size == 0


def test_input_order_does_not_change_graph_identity():
    a = fixture()
    b = Graph.from_contacts([10, 20, 30, 40], [30, 20, 10, 20],
                            [20, 30, 20, 10], [5, 3, 5, 1], [1, -1, 1, 1], 0.2)
    assert a.identity() == b.identity()
    assert a.identity() != a.threshold(3).identity()

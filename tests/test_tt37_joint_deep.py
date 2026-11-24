import numpy as np
import pytest

from scripts.experimental.tt37_joint_deep import DeepField, cards_for
from scripts.experimental.tt37_joint_deep_report import totals


def test_motion_branches_are_driven_by_current_observation():
    states = np.zeros((32, 11))
    states[:16, 4] = -3.0
    states[16:, 4] = 3.0
    labels = np.repeat(np.arange(2) + 0.25, 16)[:, None] * np.ones((32, 6))
    field = DeepField(dict(states=states, targets_now=labels, targets_lead=labels + .1), "local")
    for i in (0, 16):
        target, support = field.get(states[i, :6], states[i, 6:])
        np.testing.assert_allclose(target, labels[i])
        assert support == 1.0
    optical = states[0, :6].copy()
    optical[0] = 2.
    assert field.get(optical, states[0, 6:]) == (None, 0.)


def test_familiar_launches_keep_first_seed_and_redraw_only_replies():
    first = dict(id=3, seed=41, reply_seed=100, opening="anchor")
    other = dict(id=7, seed=42, reply_seed=101)
    receipt = dict(practice=[dict(card=first), dict(card=other)])
    cards = cards_for(receipt, None, "fresh_replies")
    assert [c["id"] for c in cards] == [3, 3, 7, 7]
    assert [c["seed"] for c in cards] == [41, 41, 42, 42]
    assert len({c["reply_seed"] for c in cards}) == 4
    assert not {100, 101} & {c["reply_seed"] for c in cards}
    assert cards[0]["opening"] == "anchor"


def test_lane_denominators_do_not_count_a_failed_return_as_legal():
    row = dict(
        placements=dict(delivered=[
            dict(leg=0, observed_lane="left", near_contact={}, near_landed=True),
            dict(leg=1, observed_lane="right", near_contact={}, near_landed=False),
        ]),
        score=dict(learner=1, returns=3), field_ticks=2, runtime_cpu_s=1.,
    )
    score = totals(dict(rows=[row]))
    assert score["lanes"]["opening_legal"]["left"] == 1
    assert score["lanes"]["later_contacts"]["right"] == 1
    assert not score["lanes"]["later_legal"]
    row["score"]["learner"] = 2
    with pytest.raises(ValueError, match="bookkeeping"):
        totals(dict(rows=[row]))

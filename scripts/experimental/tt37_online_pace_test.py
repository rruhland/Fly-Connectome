"""Ensure the paced direct controller does not receive premature continuation credit."""
import numpy as np

from scripts.experimental.tt37_online_pace import Pace, Preference


def test_first_return_is_not_credited_with_unobserved_success():
    memory = Preference()
    ctrl = Pace(None, None, memory, np.random.default_rng(3), 0., True)
    ctrl.stroke = dict(onset=10, context=[0.]*12, primitive=2)
    ctrl.finish(32,[],True)
    assert memory.rows == []
    ctrl.close(72,[])
    assert memory.rows[0]["continuation"] == 1.
    assert not memory.rows[0]["next_observed"]


def test_measured_following_return_is_credited_only_after_the_following_return():
    memory = Preference()
    ctrl = Pace(None,None,memory,np.random.default_rng(3),0.,True)
    ctrl.stroke = dict(onset=20,context=[0.]*12,primitive=2)
    ctrl.finish(48,[],True)
    ctrl.stroke = dict(onset=73,context=[0.]*12,primitive=0)
    ctrl.finish(100,[],True)
    assert len(memory.rows) == 1 and memory.rows[0]["delay_expired_at"] == 100
    assert memory.rows[0]["next_legal"] and memory.rows[0]["continuation"] > 1.6

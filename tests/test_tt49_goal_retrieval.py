from collections import deque
from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt49_goal_retrieval as tactic


def test_online_outgoing_label_requires_two_fresh_stable_bounced_readings(monkeypatch):
    obj = tactic.GoalFragments.__new__(tactic.GoalFragments)
    obj.new_point()
    obj.pending = ("goal-pending", 3, 9, (0., .53, 2.4, 3.3))
    obj.online = deque([(None, None, obj.pending)], maxlen=768)
    x = np.array([0., -1.05, .97, 0., -3.7, 1.4])
    monkeypatch.setattr(tactic, "physical_projection", lambda *a: (0., 1.052, .12))
    obj.settle(20, SimpleNamespace(time=20 / 60, bounced=False), x)
    obj.settle(21, SimpleNamespace(time=20 / 60, bounced=True), x)
    assert len(obj.lines) == 0
    obj.settle(22, SimpleNamespace(time=22 / 60, bounced=True), x)
    assert obj.online[0][2][0] == "goal-pending"
    obj.settle(24, SimpleNamespace(time=24 / 60, bounced=True), x)
    assert obj.online[0][2][0] == "goal-confirmed"
    assert obj.confirmed == 1 and obj.pending is None

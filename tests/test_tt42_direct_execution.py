from collections import Counter
from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt42_direct_execution import BASE, Execution, innovation, mpc


def ready(tick, *, fail_check=False):
    actor = object.__new__(Execution)
    actor.diag = Counter()
    actor.notes = []
    actor.best = np.zeros((mpc.HORIZON, 6))
    actor.forecast = 10, np.zeros((mpc.HORIZON, 11))
    actor.scales = np.ones(3)
    actor.hit = 20
    actor.tick = 10
    actor.lease = 5
    actor.next = 15
    actor.last_check = 10
    actor.support = 2
    actor.fast = 0
    calls = []

    def search(*args):
        calls.append(args[0])
        actor.notes.append({"chosen": None})

    actor.search = search
    actor.refresh_incumbent = lambda *_: not fail_check
    eye = SimpleNamespace(filter=SimpleNamespace(track=[(tick * mpc.DT, None)]))
    return actor, eye, calls


def test_early_actor_rechecks_only_on_fresh_innovation_and_invalid_incumbent():
    actor, eye, calls = ready(12, fail_check=True)
    actor.update_motion(12, eye, np.array([2., 0., 0., 0., 0., 0.]), np.zeros(6), None, None)
    assert calls == [12]
    assert actor.best is None
    assert actor.diag["early_replans"] == 1
    assert actor.notes[-1]["execution_trigger"] == "innovation"

    actor, eye, calls = ready(11, fail_check=True)
    actor.update_motion(12, eye, np.array([2., 0., 0., 0., 0., 0.]), np.zeros(6), None, None)
    assert not calls


def test_accepted_buffer_preserved_if_fresh_observation_corroborates_it():
    actor, eye, calls = ready(16)
    before = actor.best
    actor.update_motion(16, eye, np.zeros(6), np.zeros(6), None, None)
    assert not calls
    assert actor.best is before
    assert actor.lease >= 8
    assert actor.diag["preserved_due_calls"] == 1


def test_ranker_is_ordinary_and_gripper_is_not_an_innovation_input():
    assert Execution.search is BASE.search
    assert Execution.robust is BASE.robust
    np.testing.assert_equal(innovation(np.zeros(6), np.array([0., 0., 0., 0., 0., .4]), np.zeros(11)), 0.)

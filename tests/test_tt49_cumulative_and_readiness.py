from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental.tt49_body_readiness import Player
from scripts.experimental.tt49_cumulative_memory import Growing


def test_growing_bank_preserves_stock_lookup_across_earned_ties_and_far_queries():
    rng = np.random.default_rng(4935)
    goals = lambda i: np.full((mpc.HORIZON, 6), float(i))
    memory = Growing.__new__(Growing)
    memory.rows = [(rng.normal(size=16), goals(i), (i // 3, 0)) for i in range(40)]
    memory.tree = cKDTree(np.array([r[0] for r in memory.rows]))
    zero = np.zeros(16)
    memory.online = deque(
        [(zero.copy(), goals(i + 40), ("live", i // 2)) for i in range(18)]
    )
    memory.states = memory.stamp = None
    for state in (zero, rng.normal(size=16), np.full(16, 12.0)):
        key = state * mpc.OPEN_NORM
        x, q, ref = key[:6], key[6:11], key[6:11] + key[11:]
        for limit in (2, 7):
            expected = mpc.Fragments.priors(memory, x, q, ref, limit)
            actual = memory.priors(x, q, ref, limit)
            assert [(a[0, 0], b) for a, b in expected] == [
                (a[0, 0], b) for a, b in actual
            ]
    memory.online.append((key / mpc.OPEN_NORM, goals(1000), ("live", 101)))
    assert memory.priors(x, q, ref, 2)[0][0][0, 0] == 1000.0


def test_readiness_credits_prior_departure_after_next_outgoing_has_started():
    learner = Player.__new__(Player)
    learner.readiness = SimpleNamespace(exp=[])
    learner.diag = Counter()
    old = np.arange(11, dtype=float)
    learner.last = old + 100.0
    learner.first = None
    learner.pending_reward = (old, (4, np.ones(5)), 23)
    x = np.array([0.0, -1.2, 1.0, 0.0, -2.0, 0.0])
    tape = [(t, x, np.full(6, float(t)), None, None) for t in range(5, 29)]
    learner.confirm(tape)
    source, early, late, offset = learner.readiness.exp[0]
    assert np.array_equal(source, old)
    assert np.all(early == 1.0) and np.all(late == 20.0)
    assert offset == (20 - 4) * mpc.DT
    assert learner.pending_reward is None
    assert learner.diag["earned_readiness_transitions"] == 1

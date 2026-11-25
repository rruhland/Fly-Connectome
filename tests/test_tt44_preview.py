import numpy as np

from scripts.experimental import tt44_preview as p


class Field:
    def get(self, x, q):
        return np.r_[2. * x[1], .2, .3, .4, .5, .6], 1.


def motion(y):
    return np.array([.04, y, .925, .08, -3.8, -.1])


def test_future_feedback_is_a_bounded_short_preview():
    obs = motion(-.36)
    plan = p.preview(obs, 7)
    assert -0.61 <= plan[1] < obs[1]
    assert plan[2] < obs[2]
    assert plan[4] == obs[4]
    assert np.allclose(p.preview(obs, 0), obs)
    assert np.array_equal(obs, motion(-.36))


def test_local_actual_reward_changes_greedy_delay_and_checkpoint(tmp_path):
    bandit = p.Bandit()
    key = np.zeros(p.FEATURES)
    assert bandit.greedy(key) == 0
    for k in range(5):
        bandit.update(key + .05 * k, 0, 0)
        bandit.update(key + .05 * k, 2, 1)
        bandit.update(key + 2. - .05 * k, 0, 1)
        bandit.update(key + 2. - .05 * k, 2, 0)
    assert bandit.greedy(key) == 2
    assert bandit.greedy(key + 2.) == 0
    path = tmp_path / 'law.npz'
    bandit.save(path)
    assert p.Bandit.load(path).greedy(key) == 2


def test_matched_exploration_counts_once_and_never_updates_control():
    code = next(i for i in range(100) if p.exploration(i, 0) == (True, 2))
    memories = [p.Bandit(), p.Bandit()]
    policies = [p.Passage(policy, mem, serial=code) for policy, mem in
                zip(('adapt', 'no-update'), memories, strict=True)]
    q = np.zeros(6)
    for controller in policies:
        for t in range(5):
            x = motion(.61 - t * .06)
            controller.get(t, x, q, q, Field(), q)
        assert controller.current["effect_ticks"] == 3
        assert controller.current["delay_ticks"] == 4
        assert controller.current["forced_exploration"]
        controller.opportunity = True
        controller.heard(0, 8)
        controller.heard(None, 9)
        assert len(controller.trace) == 1
        assert controller.trace[0]["earned"] == 0
    assert memories[0].score == [0]
    assert memories[1].score == []


def test_censored_approach_never_updates():
    key = p.Bandit()
    controller = p.Passage("adapt", key, serial=11)
    for t in range(4):
        controller.get(t, motion(.6 - .06 * t), np.zeros(6), np.zeros(6), Field(), np.zeros(6))
    controller.heard(None, 8)
    assert controller.trace[0]["earned"] is None
    assert key.score == []

import inspect
import json

import pytest

from pong_carried_structure_learner import CarriedStructureLearner, state_hash
from run_pong_carried_learning import feedback, apply_actuator_intervention


def observe(learner, hits=0, reward=0., terminated=False, truncated=False):
    return learner.observe(own_hits=hits, reward=reward, terminated=terminated,
        truncated=truncated, elapsed_seconds=.02)


def test_feedback_boundary_cannot_receive_phase_seed_or_reporting_opportunities():
    assert list(inspect.signature(CarriedStructureLearner.begin_game).parameters) == ['self']
    assert set(inspect.signature(CarriedStructureLearner.observe).parameters) == {
        'self', 'own_hits', 'reward', 'terminated', 'truncated', 'elapsed_seconds'}
    row = dict(hits=1, reward=0., terminated=False, truncated=False,
        elapsed_seconds=.02, seed=42, phase='secret', own_opportunities=999,
        reverse=True, policy_id='expert', action=2)
    assert feedback(row) == dict(own_hits=1, reward=0., terminated=False,
        truncated=False, elapsed_seconds=.02)


@pytest.mark.parametrize('kind', ['capped_beta', 'discounted_ucb'])
def test_hits_and_only_public_negative_terminal_scores_update_evidence(kind):
    learner = CarriedStructureLearner(kind, arms=2)
    arm, _ = learner.begin_game()
    observe(learner, hits=2)
    assert learner.lifetime_success[arm] == 2
    # A positive score is an opponent miss, not a failed return.
    update = observe(learner, reward=1., terminated=True)
    assert update['unfinished_censored'] and update['failures'] == 0
    assert sum(learner.lifetime_failure) == 0
    arm, _ = learner.begin_game()
    observe(learner, hits=1)
    update = observe(learner, reward=-1., terminated=True)
    assert update['failures'] == 1 and not update['unfinished_censored']
    assert learner.lifetime_success[arm] >= 1
    assert learner.lifetime_failure[arm] == 1
    arm, _ = learner.begin_game()
    before = sum(learner.lifetime_failure)
    # This includes a time cap after a simulator-only missed paddle plane: no
    # public loss score has arrived, so the unfinished return remains censored.
    update = observe(learner, truncated=True)
    assert update['unfinished_censored'] and sum(learner.lifetime_failure) == before


def test_a_structure_commits_through_all_intermediate_rewards():
    learner = CarriedStructureLearner('capped_beta', arms=3)
    arm, _ = learner.begin_game()
    observe(learner, hits=1)
    observe(learner, hits=1)
    assert learner.active_arm == arm
    with pytest.raises(ValueError):
        learner.begin_game()
    observe(learner, truncated=True)
    assert learner.active_arm is None


@pytest.mark.parametrize('kind', ['capped_beta', 'discounted_ucb'])
def test_game_reset_carries_learned_evidence_and_json_resume_is_exact(kind):
    live = CarriedStructureLearner(kind, arms=4, random_seed=121)
    for game in range(30):
        restored = CarriedStructureLearner.from_state(json.loads(json.dumps(live.state())))
        assert state_hash(live.state()) == state_hash(restored.state())
        assert live.begin_game() == restored.begin_game()
        for learner in (live, restored):
            observe(learner, hits=game % 3)
            observe(learner, reward=-1. if game % 4 else 0.,
                terminated=bool(game % 4), truncated=not bool(game % 4))
        assert state_hash(live.state()) == state_hash(restored.state())
    assert sum(live.lifetime_success) == 30
    assert sum(live.lifetime_failure) == 22
    assert live.games == 30


def test_surprise_resweep_preserves_all_lifetime_evidence_and_caps_effective_memory():
    learner = CarriedStructureLearner('capped_beta', arms=3, random_seed=6)
    first = learner.priority[0]
    for _ in range(3):
        arm, _ = learner.begin_game()
        observe(learner, hits=20 if arm == first else 0, reward=-1., terminated=True)
    assert learner.incumbent == first
    before = list(learner.lifetime_success)
    for _ in range(2):
        arm, mode = learner.begin_game()
        assert mode == 'exploit' and arm == first
        # Earlier hits must not hide two consecutive own-loss game endings.
        observe(learner, hits=1, reward=-1., terminated=True)
    assert learner.sweeps == 1 and sorted(learner.queue) == list(range(3))
    assert learner.lifetime_success[first] == before[first] + 2
    assert learner.lifetime_failure[first] == 3
    assert all(s + f <= 8. + 1e-12 for s, f in zip(learner.success, learner.failure))
    assert learner.begin_game()[1] == 'surprise_sweep'


def test_frozen_branch_retains_acquired_evidence_without_future_reward_updates():
    learner = CarriedStructureLearner('capped_beta', arms=3)
    for game in range(3):
        learner.begin_game()
        observe(learner, hits=game + 1, truncated=True)
    frozen = learner.frozen_branch()
    success, failure = list(frozen.success), list(frozen.failure)
    lifetime = list(frozen.lifetime_success)
    best = learner.best_arm()
    for _ in range(3):
        assert frozen.begin_game() == (best, 'frozen_learned_incumbent')
        update = observe(frozen, hits=1, reward=-1., terminated=True)
        assert update['failures'] == update['successes'] == 0
    assert frozen.success == success and frozen.failure == failure
    assert frozen.lifetime_success == lifetime


def test_reset_control_rotates_draws_but_discards_reward_knowledge():
    learner = CarriedStructureLearner('reset_beta', arms=16, random_seed=17)
    chosen = []
    for _ in range(24):
        arm, _ = learner.begin_game()
        chosen.append(arm)
        assert sum(learner.success) == sum(learner.failure) == 0
        assert sum(learner.lifetime_success) == 0
        observe(learner, hits=10, truncated=True)
    assert len(set(chosen)) > 5


def test_reversal_changes_only_physical_action_not_requested_command_history():
    class Environment:
        def __init__(self):
            self.physical = []
        def step(self, action):
            self.physical.append(action)
            return dict(hits=0, reward=0., terminated=False,
                truncated=False, elapsed_seconds=.02)
    class Controller:
        previous_action = 1
        def act(self, env, requested):
            result = env.step(requested)
            self.previous_action = requested
            return result
    env, controller = Environment(), Controller()
    apply_actuator_intervention(env, True)
    row = controller.act(env, 0)
    assert env.physical == [2] and controller.previous_action == 0
    assert feedback(row)['own_hits'] == 0
    controller.act(env, 1)
    controller.act(env, 2)
    assert env.physical == [2, 1, 0] and controller.previous_action == 2


def test_all_arms_discount_on_public_outcome_not_on_censored_time():
    learner = CarriedStructureLearner('discounted_ucb', arms=3)
    arm, _ = learner.begin_game()
    observe(learner, hits=1, truncated=True)
    before = learner.success[arm]
    learner.begin_game()
    observe(learner, hits=1, truncated=True)
    assert learner.success[arm] == pytest.approx(before * .95)
    before = list(learner.success)
    learner.begin_game()
    observe(learner, truncated=True)
    assert learner.success == before

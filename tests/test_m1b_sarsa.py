"""Independent online forward-view replay, not a second Dutch-trace update."""
import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.sarsa import TrueOnlineSarsa
from fly_connectome.control.environment import PongCamera


def forward_prefixes(initial, features, rewards, discounts, alpha, lam):
    # At each horizon rebuild all previous updates from the original weights.
    # Stored bootstrap j uses online weights BEFORE observing transition j-1.
    online = initial.copy()
    bootstraps = []
    results = []
    for horizon in range(1, len(rewards)+1):
        bootstraps.append(float(online @ features[horizon]))
        targets = np.zeros(horizon)
        target = bootstraps[-1]
        for k in reversed(range(horizon)):
            if k == horizon-1:
                target = rewards[k]+discounts[k]*target
            else:
                target = rewards[k]+discounts[k]*((1-lam)*bootstraps[k]+lam*target)
            targets[k] = target
        replay = initial.copy()
        for k in range(horizon):
            replay += alpha*(targets[k]-replay @ features[k])*features[k]
        online = replay
        results.append(online.copy())
    return results


@pytest.mark.parametrize('lam', [0., .3, .9, 1.])
@pytest.mark.parametrize('terminal', [False, True])
def test_every_prefix_matches_independent_forward_view(lam, terminal):
    # Overlapping nonorthogonal features, nonzero initial weights, variable dt.
    features = np.array([[1,.2,0], [.5,1,.1], [1,-.3,.7], [.2,.8,1], [.7,0,.2]])
    rewards = [.1,-.7,0.,1.]
    durations = [.01,.17,.025,.4]
    gamma = .43
    discounts = gamma**np.array(durations)
    if terminal:
        discounts[-1] = 0
    initial = np.array([.3,-.2,.1])
    expected = forward_prefixes(initial, features, rewards, discounts, .07, lam)
    agent = TrueOnlineSarsa(1, 3, alpha=.07, lam=lam, gamma_per_second=gamma,
                            epsilon=0, weights=initial[None])
    agent.start(np.arange(3), features[0])
    for k, target in enumerate(expected):
        agent.advance(rewards[k], durations[k], np.arange(3), features[k+1],
                      terminated=terminal and k==3)
        np.testing.assert_allclose(agent.weights[0], target, rtol=1e-13, atol=1e-14)


def test_pending_action_reused_and_frozen_eval_preserves_rng_and_parameters():
    agent = TrueOnlineSarsa(3, 2, epsilon=1, seed=19)
    agent.start([0], [1.])
    action, info = agent.advance(1., .02, [1], [1.])
    assert action == info['bootstrap_action'] == agent.pending_action
    before = copy.deepcopy(agent.__dict__)
    frozen = agent.frozen()
    for _ in range(10):
        frozen.choose([0,1], [1.,2.])
    np.testing.assert_array_equal(agent.weights, before['weights'])
    np.testing.assert_array_equal(agent.traces, before['traces'])
    assert agent.rng.bit_generator.state == before['rng'].bit_generator.state
    assert agent.pending_action == action
    with pytest.raises(ValueError, match='reuse pending'):
        agent.start([1], [1.])


@pytest.mark.parametrize('terminal', [False, True])
def test_end_bootstrap_and_episode_reset(terminal):
    agent = TrueOnlineSarsa(1, 2, alpha=.1, gamma_per_second=.25, epsilon=0,
                            weights=[[2.,4.]])
    agent.start([0], [1.])
    action, info = agent.advance(1., .5, [1], [1.], terminated=terminal, truncated=not terminal)
    assert info['q_next'] == (0 if terminal else 4)
    assert info['outgoing_discount'] == (0 if terminal else .5)
    assert agent.weights[0,0] == pytest.approx(1.9 if terminal else 2.1)
    assert action is None and agent.pending_features is None
    assert not agent.traces.any() and agent.q_old == agent.incoming_discount == 0
    agent.start([0], [1.])
    assert not agent.traces.any()


def test_score_stops_first_tick_and_preserves_genuine_terminal_frame():
    env = PongCamera(1)
    env.game.ball[0] = torch.tensor([.999, .8])
    env.game.ball_velocity[0] = torch.tensor([.55, 0.])
    rng = env.game.rng.clone()
    result = env.step(1)
    assert result.terminated and not result.truncated and result.reward == 1
    assert result.physics_ticks == env.total_ticks == 1
    assert result.elapsed_seconds == env.game.config.dt
    assert env.game.ball[0,0] > 1
    assert torch.equal(result.frame, env.game.render(64,64)[0])
    assert torch.equal(rng, env.game.rng)  # No hidden reset/serve RNG draw.
    with pytest.raises(ValueError, match='reset required'):
        env.step(1)
    reset_frame = env.reset()
    assert not torch.equal(result.frame, reset_frame)


def test_actual_tick_durations_and_time_limit_final_observation():
    env = PongCamera(1, max_episode_ticks=4)
    a, b = env.step(1), env.step(1)
    assert a.physics_ticks == b.physics_ticks == 2
    assert b.truncated and not b.terminated and b.reward == 0
    assert torch.equal(b.frame, env.game.render(64,64)[0])
    env.reset()
    c = env.step(1)
    assert c.physics_ticks == 3  # Global 2,2,3,2,3 cadence retained across reset.


def test_next_action_is_selected_before_update_and_retained_after_greedy_flip():
    agent = TrueOnlineSarsa(2, 1, alpha=1., lam=0, gamma_per_second=1,
                            epsilon=0, weights=[[1.],[0.]])
    assert agent.start([0], [1.]) == 0
    action, info = agent.advance(-2., .1, [0], [1.])
    assert agent.weights[:,0].tolist() == [-1.,0.]
    assert int(agent.weights[:,0].argmax()) == 1
    assert action == info['bootstrap_action'] == agent.pending_action == 0


def test_truncated_prefix_matches_independent_reference_and_clears_traces():
    features = np.array([[1.,.5], [.2,1.], [.7,-.4]])
    expected = forward_prefixes(np.array([.3,-.1]), features, [.2,.7], [.8**.03,.8**.11], .2,.7)
    agent = TrueOnlineSarsa(1,2,alpha=.2,lam=.7,gamma_per_second=.8,epsilon=0,weights=[[.3,-.1]])
    agent.start([0,1],features[0])
    agent.advance(.2,.03,[0,1],features[1])
    agent.advance(.7,.11,[0,1],features[2],truncated=True)
    np.testing.assert_allclose(agent.weights[0],expected[-1],rtol=1e-13,atol=1e-14)
    assert not agent.traces.any()


def test_action_block_features_match_forward_reference(monkeypatch):
    observations = np.array([[1.,.5],[.2,1.],[.7,-.4],[1.,.6]])
    actions = [0,1,0,1]
    blocks = np.zeros((4,4))
    for k, action in enumerate(actions):
        blocks[k,action*2:(action+1)*2] = observations[k]
    rewards, durations = [.1,-.3,.6], [.01,.2,.07]
    initial = np.array([.2,-.1,.4,.3])
    expected = forward_prefixes(initial,blocks,rewards,.7**np.array(durations),.1,.8)
    learner = TrueOnlineSarsa(2,2,alpha=.1,lam=.8,gamma_per_second=.7,weights=initial.reshape(2,2))
    prescribed = iter(actions)
    monkeypatch.setattr(learner,'_sample',lambda x: next(prescribed))
    learner.start([0,1],observations[0])
    for k in range(3):
        learner.advance(rewards[k],durations[k],[0,1],observations[k+1])
        np.testing.assert_allclose(learner.weights.ravel(),expected[k],rtol=1e-13,atol=1e-14)


def test_session_evaluation_isolates_perception_physics_and_all_rngs():
    import random
    from fly_connectome.control.session import LearningSession
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from prove_m1b_frozen_recovery import fingerprint
    torch.set_num_threads(1)
    session = LearningSession(max_episode_ticks=1)
    session.learner.weights[:] = .001
    before = fingerprint(session)
    globals_before = fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))
    records = session.evaluate(2)
    assert any(value != 0 for row in records for value in row['q'])
    assert fingerprint(session) == before
    assert fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state())) == globals_before


def test_session_truncation_uses_final_camera_features_not_reset(monkeypatch):
    from fly_connectome.control.session import LearningSession
    torch.set_num_threads(1)
    session = LearningSession(max_episode_ticks=1)
    session.learner.weights[:] = .001
    captured = {}
    advance = session.learner.advance
    def inspect(reward, elapsed, indices, values, **flags):
        captured.update(indices=indices.copy(),values=values.copy(),flags=flags)
        assert session.env.needs_reset
        assert session.env.game.rally_steps.item() == 1
        return advance(reward,elapsed,indices,values,**flags)
    monkeypatch.setattr(session.learner,'advance',inspect)
    row = session.step()
    expected = .001*sum(captured['values'])
    assert row['q_next'] == pytest.approx(expected)
    assert row['truncated'] and not row['terminated']
    assert session.env.game.rally_steps.item() == 0
    assert session.learner.pending_action is not None

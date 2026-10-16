import random
from dataclasses import replace
import numpy as np
import pytest
import torch
from fly_connectome.control.session import LearningSession
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig


def session(**kwargs):
    torch.set_num_threads(1)
    random.seed(99890);np.random.seed(99890);torch.manual_seed(99890)
    return LearningSession(99890,alpha=.001,lam=.995,opponent_speed=.25,**kwargs)


def test_hold1_preserves_records_weights_and_complete_state():
    first=session();records=[first.step() for _ in range(20)];state=fingerprint(first)
    second=session(action_hold_frames=1)
    assert [second.step() for _ in range(20)]==records
    assert fingerprint(second)==state and not hasattr(second,'hold_frames')


def test_hold4_observes_every_camera_and_updates_only_at_boundary():
    current=session(action_hold_frames=4);observations=[];observe=current.observe
    def spy(frame):
        observations.append(frame.clone());return observe(frame)
    current.observe=spy
    rows=[current.step() for _ in range(4)]
    assert len(observations)==4 and current.transition_cameras==4
    assert [r['macro_update'] for r in rows]==[False,False,False,True]
    assert len({r['action'] for r in rows})==1
    assert rows[0]['q_next'] is None and rows[0]['delta'] is None
    assert current.learner.transitions==1 and current.macro_choices==1
    assert current.learner.lam==.995**4 and current.hold_frames==0


def test_macro_target_exactly_composes_discounted_camera_rewards():
    # Synthetic reward fixture checks the general SMDP sum, not the real objective.
    current=session(action_hold_frames=4);step=current.env.step
    rewards=[.2,-.3,.4,1.];durations=[]
    initial=current.learner.pending_features.copy();action=current.learner.pending_action
    def synthetic(action):
        result=step(action);index=len(durations);durations.append(result.elapsed_seconds)
        if index==3:current.env.needs_reset=True
        return replace(result,reward=rewards[index],terminated=index==3,truncated=False)
    current.env.step=synthetic
    rows=[current.step() for _ in range(4)]
    prefix=0.;target=0.
    for reward,duration in zip(rewards,durations):
        target+=.95**prefix*reward;prefix+=duration
    assert rows[-1]['macro_reward']==pytest.approx(target,abs=1e-15)
    assert rows[-1]['delta']==pytest.approx(target,abs=1e-15)
    assert rows[-1]['macro_elapsed_seconds']==pytest.approx(9/120)
    expected=np.zeros_like(current.learner.weights);expected[action]=.001*target*initial
    np.testing.assert_allclose(current.learner.weights,expected,atol=1e-15,rtol=1e-15)
    assert rows[-1]['outgoing_discount']==0 and current.hold_frames==0


def test_real_score_ends_partial_hold_and_discounts_only_prior_camera_time():
    current=session(action_hold_frames=4)
    first=current.step()
    current.env.game.ball[0]=torch.tensor([-.01,.05])
    current.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
    last=current.step()
    assert last['terminated'] and last['hold_frame']==2 and last['macro_update']
    assert last['reward']==-1 and last['macro_reward']==-.95**first['elapsed_seconds']
    assert last['physics_ticks']==1 and last['macro_elapsed_seconds']==pytest.approx(3/120)
    assert current.hold_frames==0 and current.learner.transitions==1
    assert current.initial_cameras==2 and current.macro_choices==1


def test_partial_truncation_bootstraps_actual_final_camera_then_resets():
    current=session(action_hold_frames=4,max_episode_ticks=5)
    current.learner.weights[:]=np.random.default_rng(99891).normal(0,.001,current.learner.weights.shape)
    advance=current.learner.advance;captured={}
    def spy(reward,elapsed,indices,values,**flags):
        captured.update(weights=current.learner.weights.copy(),indices=indices.copy(),values=values.copy())
        return advance(reward,elapsed,indices,values,**flags)
    current.learner.advance=spy
    rows=[current.step() for _ in range(3)];last=rows[-1]
    assert last['truncated'] and not last['terminated'] and last['hold_frame']==3
    assert last['outgoing_discount']==pytest.approx(.95**(5/120))
    final=np.zeros(current.encoder.dimension);final[captured['indices']]=captured['values']
    expected=float(captured['weights'][last['bootstrap_action']]@final)
    assert last['q_next']==pytest.approx(expected,abs=1e-15)
    reset=np.zeros_like(final);idx,values=current.policy_features(current.latest_encoded);reset[idx]=values
    assert not np.array_equal(final,reset)
    assert current.hold_frames==0 and current.initial_cameras==2
    assert not current.learner.traces.any() and current.learner.q_old==0


def test_mid_hold_checkpoint_replay_matches_uninterrupted_complete_graph(tmp_path):
    uninterrupted=session(action_hold_frames=4);expected=[uninterrupted.step() for _ in range(12)]
    complete=fingerprint(uninterrupted)
    run=DurableRun(tmp_path/'hold.db',session=session(action_hold_frames=4),snapshot_transitions=2)
    assert [run.step() for _ in range(11)]==expected[:11]
    state=fingerprint(run.session)
    assert run.session.hold_frames==3 and run.session.macro_choices==3 and run.session.learner.transitions==2
    assert run.db.execute('SELECT count FROM replay_active').fetchone()[0]==10
    run.close_without_checkpoint()
    recovered=DurableRun(tmp_path/'hold.db')
    assert recovered.replayed==1 and fingerprint(recovered.session)==state
    assert recovered.step()==expected[-1] and fingerprint(recovered.session)==complete
    assert recovered.db.execute('SELECT count(*) FROM replay_samples').fetchone()[0]==12
    recovered.stop()


def test_frozen_random_hold_rng_cadence_early_end_and_replay(tmp_path):
    torch.set_num_threads(1)
    config=PongEvaluationConfig(99892,episodes=3,max_episode_ticks=5,policy_kind='random',
        action_seed=99893,opponent_speed=.25,action_hold_frames=4)
    first=FreshPongEvaluator(None,config);rows=[]
    while not first.complete:rows.append(first.step())
    rng=np.random.default_rng(99893)
    assert [r['action'] for r in rows if r['macro_choice']]==[int(rng.integers(3)) for _ in range(3)]
    assert first.macro_choices==3 and first.transition_cameras==len(rows)
    assert first.hold_remaining==0 and first.held_action is None
    assert first.summary()['truncations']==3 and first.summary()['macro_choices']==3
    run=DurableRun(tmp_path/'frozen.db',session=FreshPongEvaluator(None,config),snapshot_transitions=2)
    assert [run.step() for _ in range(5)]==rows[:5]
    state=fingerprint(run.session);run.close_without_checkpoint()
    restored=DurableRun(tmp_path/'frozen.db')
    assert restored.replayed==1 and fingerprint(restored.session)==state
    remainder=[]
    while not restored.session.complete:remainder.append(restored.step())
    assert remainder==rows[5:] and fingerprint(restored.session)==fingerprint(first)
    restored.stop()


def test_learning_replay_across_partial_macro_truncations(tmp_path):
    first=session(action_hold_frames=4,max_episode_ticks=5)
    rows=[first.step() for _ in range(20)];complete=fingerprint(first)
    assert any(r['truncated'] and r['hold_frame']<4 for r in rows)
    run=DurableRun(tmp_path/'partial.db',session=session(action_hold_frames=4,max_episode_ticks=5),snapshot_transitions=2)
    assert [run.step() for _ in range(7)]==rows[:7]
    state=fingerprint(run.session);run.close_without_checkpoint()
    recovered=DurableRun(tmp_path/'partial.db')
    assert recovered.replayed==1 and fingerprint(recovered.session)==state
    assert [recovered.step() for _ in range(13)]==rows[7:]
    assert fingerprint(recovered.session)==complete
    assert recovered.session.learner.transitions<recovered.session.transitions==20
    recovered.stop()

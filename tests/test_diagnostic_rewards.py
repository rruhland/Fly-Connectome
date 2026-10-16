from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,random,sqlite3
import numpy as np
import pytest
import torch
from fly_connectome.pong import Pong,Physics
from fly_connectome.control.rewards import REWARD_VERSION,validate_reward,learning_reward
from fly_connectome.control.session import LearningSession
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig

def spec(hit=0.,survival=0.):
    return dict(version=REWARD_VERSION,own_hit_bonus=hit,survival_per_second=survival)

def force_contact(game,left=True):
    p=game.config;plane=p.paddle_x+p.ball_radius if left else 1-p.paddle_x-p.ball_radius
    game.ball[0]=torch.tensor([plane+(.001 if left else -.001),.5])
    game.ball_velocity[0]=torch.tensor([-.55 if left else .55,0.])

def test_hits_are_own_paddle_only_and_diagnostic_scalar_is_unclipped():
    game=Pong([319273],config=Physics(opponent_speed=.25));force_contact(game)
    result=game.step(torch.tensor([0.]),auto_reset=False)
    assert result.hits.item()==1 and result.scores.item()==0
    game=Pong([319273],config=Physics(opponent_speed=.25));force_contact(game,False)
    result=game.step(torch.tensor([0.]),auto_reset=False)
    assert result.hits.item()==0 and result.scores.item()==0
    value,parts=learning_reward(SimpleNamespace(reward=-1.,hits=1,elapsed_seconds=.5),spec(1.,.1))
    assert value==pytest.approx(.05) and parts['hit_reward']==1. and parts['survival_reward']==.05
    assert learning_reward(SimpleNamespace(reward=1.,hits=2,elapsed_seconds=1.),spec(1.))[0]==3.
    for bad in [spec(-1),spec(survival=float('nan')),dict(spec(),extra=1)]:
        with pytest.raises(ValueError):validate_reward(bad)

def projection(tmp_path):
    matrix=np.zeros((4,4389));matrix[0,4384]=1.;matrix[1,4388]=1.;matrix[2,4253]=1.;matrix[3,0]=1.
    path=tmp_path/'matrix.npy';np.save(path,matrix,allow_pickle=False)
    return dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=256.,calibration={})

@pytest.mark.parametrize('reward',[spec(1.),spec(survival=.1)])
def test_shaped_reward_td_and_crash_replay_keep_raw_score_exact(tmp_path,reward):
    torch.set_num_threads(1);config=projection(tmp_path)
    def create(path):
        random.seed(82931);np.random.seed(82931);torch.manual_seed(82931)
        s=LearningSession(172900127,max_episode_ticks=25,alpha=.001,lam=.995,
            opponent_speed=.25,policy_projection=config,reward_shaping=reward)
        force_contact(s.env.game);s.learner.pending_action=1
        return DurableRun(path,session=s,snapshot_transitions=2)
    a=create(tmp_path/'continuous.db');first=a.step()
    assert first['hits']==1 and first['reward']==0.
    assert first['learning_reward']==pytest.approx(reward['own_hit_bonus']+reward['survival_per_second']*first['elapsed_seconds'])
    assert first['delta']==first['learning_reward']
    for _ in range(7):a.step()
    expected=fingerprint(a.session);weights=a.session.learner.weights.copy();a.stop()
    b=create(tmp_path/'split.db')
    for _ in range(3):b.step()
    b.close_without_checkpoint();b=DurableRun(tmp_path/'split.db');assert b.replayed==1
    for _ in range(5):b.step()
    assert fingerprint(b.session)==expected;np.testing.assert_array_equal(b.session.learner.weights,weights)
    mapped,cap=b.session.mapped_evaluation();b.stop()
    with sqlite3.connect(tmp_path/'continuous.db') as c,sqlite3.connect(tmp_path/'split.db') as d:
        assert c.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==d.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()
    frozen=FreshPongEvaluator(mapped,PongEvaluationConfig(719283,episodes=1,max_episode_ticks=5,opponent_speed=.25,policy_scaling=cap))
    before=frozen.learned_hash();theta=weights.copy()
    while not frozen.complete:
        row=frozen.step();assert 'learning_reward' not in row
    assert frozen.learned_hash()==before
    np.testing.assert_array_equal(weights,theta)

def test_zero_coefficients_preserve_unshaped_trajectory_and_config_is_owned():
    torch.set_num_threads(1)
    def session(reward):
        random.seed(8213);np.random.seed(8213);torch.manual_seed(8213)
        return LearningSession(739183,max_episode_ticks=25,alpha=.001,lam=.995,reward_shaping=reward)
    a=session(None);config=spec();b=session(config);config['own_hit_bonus']=9
    for _ in range(8):
        x=a.step();y=b.step()
        assert y.pop('learning_reward')==x['reward']
        assert y.pop('hit_reward')==y.pop('survival_reward')==0.
        assert x==y
    np.testing.assert_array_equal(a.learner.weights,b.learner.weights)
    assert b.reward_shaping==spec() and not hasattr(a,'reward_shaping')

import copy,hashlib,json,random,sqlite3,sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
from fly_connectome.control.exploration import BlockExploration
from fly_connectome.control.sarsa import TrueOnlineSarsa
from fly_connectome.control.session import LearningSession
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.rewards import REWARD_VERSION


def projection(tmp_path):
    matrix=np.zeros((4,4389));matrix[0,4384]=1.;matrix[1,4388]=1.;matrix[2,4253]=1.;matrix[3,0]=1.
    path=tmp_path/'T.npy';np.save(path,matrix,allow_pickle=False)
    return dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        max_norm2=256.,calibration={})


def reward():return dict(version=REWARD_VERSION,score_coefficient=0.,own_hit_bonus=1.,survival_per_second=0.)


def test_exploratory_drive_persists_but_greedy_recomputes_each_camera():
    class FixedRng:
        def __init__(self,u):self.u=u;self.gates=0;self.actions=0
        def random(self):self.gates+=1;return self.u
        def integers(self,n):self.actions+=1;return 2
        def choice(self,ties):return ties[0]
    for u in (.1,.3):
        rng=FixedRng(u);block=BlockExploration(10)
        for i in range(20):
            q=np.zeros(3);q[i%2]=1.
            action=block.choose(q,.2,rng)
            assert action==(2 if u<.2 else i%2)
            assert block.last_selection['block_frame']==i%10+1
        assert rng.gates==2 and rng.actions==(2 if u<.2 else 0)
    block=BlockExploration(10);rng=np.random.default_rng(741);count=0
    for _ in range(10000):block.choose(np.array([1.,0.,0.]),.2,rng);count+=block.last_selection['exploratory']
    assert .16<count/10000<.24  # A .2 burst initiation probability would instead approach .71.
    for value in (0,-1,True,1.2):
        with pytest.raises(ValueError):BlockExploration(value)


def test_explicit_block1_exactly_preserves_legacy_actions_rng_and_updates():
    kwargs=dict(alpha=.01,lam=.995,gamma_per_second=.95,epsilon=.2,seed=739)
    old=TrueOnlineSarsa(3,4,**kwargs);new=TrueOnlineSarsa(3,4,exploration_block_frames=1,**kwargs)
    x=np.arange(4);values=np.array([1.,.2,-.1,.4])
    old.start(x,values);new.start(x,values)
    for i in range(200):
        assert old.pending_action==new.pending_action
        terminated=i%19==18;truncated=i%23==22 and not terminated
        values=np.array([1.,np.sin(i),np.cos(i),.25])
        args=(float(i%17==0),.025,x,values)
        _,a=old.advance(*args,terminated=terminated,truncated=truncated)
        _,b=new.advance(*args,terminated=terminated,truncated=truncated)
        assert a=={k:v for k,v in b.items() if k!='bootstrap_exploration'}
        assert np.array_equal(old.weights,new.weights) and np.array_equal(old.traces,new.traces)
        assert old.rng.bit_generator.state==new.rng.bit_generator.state
        if terminated or truncated:
            assert new.exploration.remaining==0 and new.exploration.last_selection is None
            assert (b['bootstrap_exploration'] is None)==terminated
            old.start(x,values);new.start(x,values)
    assert not hasattr(old,'exploration')


def test_selection_metadata_is_preupdate_and_owned_across_reset():
    s=TrueOnlineSarsa(2,1,weights=[[1.],[0.]],alpha=1.,epsilon=0.,exploration_block_frames=10)
    s.start(np.array([0]),np.array([1.]))
    _,info=s.advance(-100.,.02,np.array([0]),np.array([1.]))
    assert info['bootstrap_action']==0 and not info['bootstrap_exploration']['non_greedy']
    assert s.weights[0,0]<s.weights[1,0]
    saved=copy.deepcopy(info['bootstrap_exploration']);s.reset_episode()
    assert info['bootstrap_exploration']==saved and s.exploration.last_selection is None


def test_partial_block_truncation_and_crash_replay_are_exact(tmp_path):
    torch.set_num_threads(1);config=projection(tmp_path)
    def create(path):
        random.seed(99);np.random.seed(99);torch.manual_seed(99)
        s=LearningSession(3,max_episode_ticks=37,alpha=.001,lam=.995,opponent_speed=.65,
            policy_projection=config,reward_shaping=reward(),exploration_block_frames=10)
        s.env.game.ball[0]=torch.tensor([s.env.game.config.paddle_x+s.env.game.config.ball_radius+.001,.5])
        s.env.game.ball_velocity[0]=torch.tensor([-.55,0.])
        return DurableRun(path,session=s,snapshot_transitions=4)
    a=create(tmp_path/'full.db');rows=[a.step() for _ in range(24)]
    assert rows[0]['hits']==rows[0]['learning_reward']==1.
    assert all(r['exploration']['action']==r['action'] and r['delta'] is not None for r in rows)
    for prev,nxt in zip(rows,rows[1:]):
        if not prev['terminated'] and not prev['truncated']:
            assert prev['bootstrap_exploration']==nxt['exploration']
    end=next(i for i,r in enumerate(rows) if r['truncated'])
    assert rows[end]['exploration']['block_frame']<10
    assert rows[end]['bootstrap_exploration'] is not None
    assert rows[end+1]['exploration']['block_frame']==1
    assert rows[end+1]['exploration']['block']>rows[end]['bootstrap_exploration']['block']
    expected=fingerprint(a.session);a.stop()
    b=create(tmp_path/'split.db')
    for _ in range(7):b.step()
    assert b.session.learner.exploration.last_selection['exploratory']
    b.close_without_checkpoint();b=DurableRun(tmp_path/'split.db');assert b.replayed==3
    for _ in range(17):b.step()
    assert fingerprint(b.session)==expected;b.stop()
    with sqlite3.connect(tmp_path/'full.db') as c,sqlite3.connect(tmp_path/'split.db') as d:
        assert c.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==d.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()


def test_worker_config_counts_checkpoint_and_greedy_export(tmp_path):
    import run_pong_development as worker
    from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
    config=projection(tmp_path);worker.atomic_json(tmp_path/'projection.json',config)
    job=dict(id='learn',kind='learn',seed=3,alpha=.001,lam=.995,opponent_speed=.65,
        action_hold_frames=1,exploration_block_frames=10,projection_from='projection.json',
        reward_shaping=reward(),transitions=16,checkpoints=[16],compact_reports=True)
    worker.atomic_json(tmp_path/'manifest.json',dict(jobs=[job],source=worker.source_signature(),
        runtime=worker.runtime_signature(),driver_hash=hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()))
    r=worker.worker(tmp_path,'learn',emit=False)
    assert r['complete'] and r['learner_updates']==16 and r['exploration_block_frames']==10
    assert r['exploratory_transitions']==10 and r['exploration_blocks_started']==2
    r=worker.worker(tmp_path,'learn',emit=False,verify_only=True);assert r['new_transitions']==0
    weights=np.load(tmp_path/'learn_at16.npy');cap=json.loads((tmp_path/'learn_at16.cap.json').read_text())
    evaluation=FreshPongEvaluator(weights,PongEvaluationConfig(99,episodes=1,max_episode_ticks=12,policy_scaling=cap,opponent_speed=.65))
    while not evaluation.complete:
        row=evaluation.step();assert 'exploration' not in row and 'bootstrap_exploration' not in row


def test_allocation_fresh_pairing_and_both_greedy_eval_at_hold1():
    import run_sustained_exploration as study
    import run_reliable_return_learning as previous
    learning,evaluation=study.declared_jobs()
    assert sum(j['transitions'] for j in learning)==48000
    assert [j['exploration_block_frames'] for j in learning]==[1,10]
    assert all(j['seed']==736187003 and j['action_hold_frames']==1 and j['reward_shaping']==reward() for j in learning)
    schedule=study.serve_schedule()
    assert len(set(schedule))==24 and not set(schedule)&set(study.base.FINAL)
    assert not set(schedule)&set(previous.serve_schedule())
    assert len(evaluation)==168
    assert all(j['config']['action_hold_frames']==1 and 'exploration_block_frames' not in j for j in evaluation if j['policy'] in study.ARMS)


def test_paired_arm_contrast_is_distinct_from_max_control_screen():
    from return_metrics import paired_evidence
    names=('independent','sustained','initial','up','noop','random1','random10')
    data={name:[dict(own_returns=int(name=='sustained'),own_opportunities=1,
        first_incoming=dict(success=name=='sustained',tick=50,pre_ball=[.064,.5],pre_velocity=[-.5,.1]))
        for _ in range(24)] for name in names}
    data['noop']=copy.deepcopy(data['sustained'])
    result=paired_evidence(data,arms=names[:2],comparison=('sustained','independent'),draws=100)
    assert result['paired_arm_comparison']['first_incoming_lift_ci95']==[1.,1.]
    assert not result['arms']['sustained']['promising_development_signal']

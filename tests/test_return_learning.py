from pathlib import Path
import hashlib,random,sqlite3,sys
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
from return_metrics import ReturnCounter,replay_records,aggregate
from fly_connectome.pong import Pong,Physics
from fly_connectome.control.environment import PongCamera
from fly_connectome.control.rewards import learning_reward,REWARD_VERSION
from fly_connectome.control.projection import PROJECTION_VERSION
from fly_connectome.control.session import LearningSession
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint

def reward():return dict(version=REWARD_VERSION,score_coefficient=0.,own_hit_bonus=1.,survival_per_second=0.)

def test_hit_only_ignores_game_scores_and_legacy_components_are_unchanged():
    for score in [-1.,1.]:
        r=SimpleNamespace(reward=score,hits=0,elapsed_seconds=.02)
        value,parts=learning_reward(r,reward());assert value==parts['score_reward']==0.
        r.hits=1;assert learning_reward(r,reward())[0]==1.
    legacy=dict(reward());legacy.pop('score_coefficient')
    value,parts=learning_reward(SimpleNamespace(reward=-1.,hits=1,elapsed_seconds=.02),legacy)
    assert value==0. and 'score_reward' not in parts

def test_incoming_opportunities_exclude_opponent_going_serve_and_count_misses():
    game=Pong([17291],config=Physics(opponent_speed=.65));original=game.step;c=ReturnCounter()
    def step(x,y,vx):
        game.ball[0]=torch.tensor([x,y]);game.ball_velocity[0]=torch.tensor([vx,0.])
        return c.step(game,original,torch.tensor([0.]),auto_reset=False)
    step(.5,.5,.55);assert c.opportunities==0 and c.first_incoming is None
    rp=1-game.config.paddle_x-game.config.ball_radius
    step(rp-.001,.5,.55);assert c.opponent_returns==1 and c.opportunities==0
    lp=game.config.paddle_x+game.config.ball_radius
    step(lp+.001,.5,-.55);assert c.opportunities==c.returns==1 and c.first_incoming['success']
    step(rp-.001,.5,.55);assert c.post_own_opponent_returns==1
    step(lp+.001,.8,-.55);assert c.opportunities==2 and c.returns==1 and c.maximum_streak==1

def test_first_incoming_state_is_identical_before_policy_contact():
    first=[]
    for action in [0,2]:
        env=PongCamera(1370788210,max_episode_ticks=1200,opponent_speed=.65);rows=[]
        while not env.needs_reset:
            r=env.step(action);rows.append(dict(action=action,**r.__dict__))
        data=replay_records(rows,seed=1370788210,opponent_speed=.65,max_episode_ticks=1200)
        first.append(data[0]['first_incoming'])
        assert aggregate(data)['own_opportunities']>=1
    assert {k:v for k,v in first[0].items() if k!='success'}=={k:v for k,v in first[1].items() if k!='success'}

def test_hit_only_hold10_full_and_partial_macro_crash_replay(tmp_path):
    torch.set_num_threads(1);matrix=np.zeros((4,4389));matrix[0,4384]=1.;matrix[1,4388]=1.;matrix[2,4253]=1.;matrix[3,0]=1.
    path=tmp_path/'T.npy';np.save(path,matrix,allow_pickle=False)
    config=dict(version=PROJECTION_VERSION,matrix_file=str(path),shape=list(matrix.shape),
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),array_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),max_norm2=256.,calibration={})
    def create(db):
        random.seed(81237);np.random.seed(81237);torch.manual_seed(81237)
        s=LearningSession(172900127,max_episode_ticks=37,alpha=.001,lam=.995,opponent_speed=.65,
            action_hold_frames=10,policy_projection=config,reward_shaping=reward())
        s.env.game.ball[0]=torch.tensor([s.env.game.config.paddle_x+s.env.game.config.ball_radius+.001,.5])
        s.env.game.ball_velocity[0]=torch.tensor([-.55,0.]);s.learner.pending_action=1
        return DurableRun(db,session=s,snapshot_transitions=4)
    a=create(tmp_path/'full.db');rows=[a.step() for _ in range(16)]
    assert rows[0]['learning_reward']==1. and rows[9]['macro_reward']==1.
    assert rows[-1]['truncated'] and rows[-1]['macro_update'] and rows[-1]['hold_frame']<10
    expected=fingerprint(a.session);a.stop()
    b=create(tmp_path/'split.db')
    for _ in range(7):b.step()
    b.close_without_checkpoint();b=DurableRun(tmp_path/'split.db');assert b.replayed==3
    for _ in range(9):b.step()
    assert fingerprint(b.session)==expected;b.stop()
    with sqlite3.connect(tmp_path/'full.db') as c,sqlite3.connect(tmp_path/'split.db') as d:
        assert c.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==d.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()

def test_fixed_allocation_fresh_schedule_and_no_final_streams():
    import run_reliable_return_learning as study
    learning,evaluation=study.declared_jobs()
    assert len(learning)==2 and sum(j['transitions'] for j in learning)==24000
    assert [j['action_hold_frames'] for j in learning]==[1,10]
    assert all(j['reward_shaping']==reward() and j['opponent_speed']==.65 for j in learning)
    schedule=study.serve_schedule();assert schedule==study.serve_schedule() and len(set(schedule))==24
    assert not set(schedule)&set(study.base.FINAL)
    assert len(evaluation)==7*24
    for job in evaluation:
        assert 'reward_shaping' not in job and job['config']['episodes']==1
        assert job['config']['seed']==schedule[job['serve_index']]
        assert job['config']['max_episode_ticks']==1200 and job['config']['opponent_speed']==.65
    for i in range(24):
        random=[j for j in evaluation if j['serve_index']==i and j['policy'].startswith('random')]
        assert len(random)==2 and random[0]['config']['action_seed']==random[1]['config']['action_seed']

def test_verification_only_never_advances_incomplete_frozen_job(tmp_path):
    import json,pytest
    import run_pong_development as driver
    from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
    config=PongEvaluationConfig(193727,episodes=1,max_episode_ticks=1200,policy_kind='noop',skip_baseline_perception=True,opponent_speed=.65)
    job=dict(id='incomplete',kind='evaluate',config=driver.asdict(config),weights_from=None)
    driver.atomic_json(tmp_path/'manifest.json',dict(source=driver.source_signature(),runtime=driver.runtime_signature(),
        driver_hash=hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),jobs=[job]))
    run=DurableRun(tmp_path/'incomplete.db',session=FreshPongEvaluator(None,config));run.step();run.stop()
    with pytest.raises(ValueError,match='actually complete'):driver.worker(tmp_path,'incomplete',emit=False,verify_only=True)
    with sqlite3.connect(tmp_path/'incomplete.db') as c:assert c.execute('SELECT count(*) FROM replay_samples').fetchone()[0]==1

def test_paired_evidence_compares_strongest_control_and_checks_first_geometry():
    import copy,pytest
    from return_metrics import paired_evidence
    names=('hold1','hold10','initial','up','noop','random1','random10')
    data={name:[dict(own_returns=2 if name.startswith('hold') else 0,own_opportunities=2,
        first_incoming=dict(success=name.startswith('hold'),tick=50,pre_ball=[.064,.5],pre_velocity=[-.5,.1]))
        for _ in range(24)] for name in names}
    result=paired_evidence(data,draws=100)
    assert result['serve_clusters']==24 and result['arms']['hold1']['promising_development_signal']
    data['noop']=copy.deepcopy(data['hold1'])
    assert not paired_evidence(data,draws=100)['arms']['hold1']['promising_development_signal']
    data['noop'][0]['first_incoming']['tick']=51
    with pytest.raises(ValueError,match='trajectory differs'):paired_evidence(data,draws=100)

import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import LearningSession
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.codec import fingerprint

root=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('winnable_screen',root/'scripts/experimental/run_winnable_pong_screen.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
spec2=importlib.util.spec_from_file_location('pong_worker',root/'scripts/experimental/run_pong_development.py')
worker=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(worker)


def report(wins,hit_wins,truncations=0,complete=True):
    return dict(wins=wins,own_hit_followed_wins=hit_wins,truncations=truncations,complete=complete,episodes=12)


def test_gate_requires_both_baseline_lifts_but_timeouts_are_nonwins():
    baselines=[report(2,0),report(3,1),report(2,0),report(4,2)]
    assert screen.choose_checkpoint([report(6,4),report(7,4)],baselines) is None
    assert screen.choose_checkpoint([report(7,5,5),report(7,5,2)],baselines)==(7,2000)
    assert screen.choose_checkpoint([report(8,6,complete=False),report(7,5)],baselines)==(7,6000)
    with pytest.raises(ValueError,match='four complete'):screen.choose_checkpoint([report(8,6)]*2,baselines[:3])


def test_only_opponent_speed_changes_and_holdout_is_unused():
    jobs=screen.declared_jobs()
    for job in jobs:
        assert job.get('scaling_from') is None
        if job['kind']=='learn':assert job['alpha']==.001 and job['lam']==.995 and job['opponent_speed']==.25
        else:assert job['config']['seed']==screen.DEV and job['config']['opponent_speed']==.25
    assert len([j for j in jobs if j['id'] in ('eval_initial','eval_up','eval_random','eval_noop')])==4


def test_default_physics_preserved_and_difficulty_pinned_across_replay(tmp_path):
    torch.set_num_threads(1)
    original=PongCamera(8819);explicit=PongCamera(8819,opponent_speed=.45)
    for action in (0,2,1,0,0,2):
        a,b=original.step(action),explicit.step(action)
        assert torch.equal(a.frame,b.frame) and a.reward==b.reward
    session=LearningSession(8820,opponent_speed=.25)
    assert session.config['opponent_speed']==.25 and session.env.game.config.opponent_speed==.25
    run=DurableRun(tmp_path/'difficulty.db',session=session,snapshot_transitions=3)
    for _ in range(5):run.step()
    state=fingerprint(run.session);run.close_without_checkpoint()
    recovered=DurableRun(tmp_path/'difficulty.db');assert recovered.replayed==2
    assert fingerprint(recovered.session)==state;recovered.stop()


def test_score_split_counts_hits_within_completed_rallies_only():
    rows=[dict(reward=reward,hits=hits,elapsed_seconds=.02,terminated=done,truncated=False)
          for reward,hits,done in ((0,0,False),(1,0,True),(0,1,False),(1,0,True),(0,1,False))]
    assert [r['hits'] for r in worker.completed_rallies(rows)]==[0,1]
    torch.set_num_threads(1)
    evaluator=FreshPongEvaluator(None,PongEvaluationConfig(8821,policy_kind='noop',opponent_speed=.25))
    evaluator.rallies=[dict(reward=1,hits=0,terminated=True,truncated=False),dict(reward=1,hits=1,terminated=True,truncated=False)]
    evaluator.episodes=2;evaluator.physics_seconds=1.
    summary=evaluator.summary()
    assert summary['natural_serve_wins']==1 and summary['own_hit_followed_wins']==1

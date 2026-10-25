"""Small mathematical/accounting tests; no candidate camera observations."""
import copy,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import fast_pong_rollout as f
import run_practical_pong as t


def test_phase_weighting_and_weaker_phase_priority():
    bank=dict(left=[1,2],right=[3,4,5,6]);h={'1':1,'2':1,'3':1,'4':1,'5':0,'6':0}
    score=t.phases(bank,h);assert score['rates']==dict(left=1.,right=.5) and score['key']==[.5,.75]
    incumbent=dict(label='incumbent',score=score)
    # Higher mean alone may hide a weaker phase loss; it must not win.
    bad=dict(label='child',score=dict(key=[.25,.9]))
    good=dict(label='combined',score=dict(key=[.6,.7]))
    assert t.choose([incumbent,bad]) is incumbent and t.choose([incumbent,good,bad]) is good
    tied=dict(label='combined',score=copy.deepcopy(score));assert t.choose([incumbent,tied]) is incumbent
    assert t.choose([good,dict(label='child',score=good['score'])]) is good


def test_failure_curriculum_is_prospective_thresholded_and_capacity_bounded():
    bank=dict(left=list(range(16)),right=list(range(20,36)))
    probe={p:dict(seeds=list(range(100 if p=='left' else 200,116 if p=='left' else 216)),current_hits=[0]*4+[1]*12,current_rate=.75) for p in bank}
    assert not t.expansion(bank,dict(rates=dict(left=.84,right=.9)),probe)
    probe['left']['current_rate']=probe['right']['current_rate']=.8
    assert not t.expansion(bank,dict(rates=dict(left=.9,right=.9)),probe)
    probe['right']['current_rate']=.75
    additions=t.expansion(bank,dict(rates=dict(left=.9,right=.9)),probe)
    assert len(additions)==8 and [v['phase'] for v in additions]==['left','right']*4 and all(v['hit']==0 for v in additions)
    bank['left']+=list(range(300,330));a=t.expansion(bank,dict(rates=dict(left=.9,right=.9)),probe)
    assert len(a)==2 # total62, respect hard64, alternate phases


def test_failed_initial_camera_has_conservative_image_and_transition_bound(monkeypatch):
    class Broken:
        def __init__(self,*a,**k):self.complete=False;self.transitions=0;self.initial_learned_hash='test'
        def step(self,**k):raise RuntimeError('after initial and next observations, before counters')
        def summary(self):return dict(complete=False,transitions=0,initial_cameras=0)
    monkeypatch.setattr(f,'FreshPongEvaluator',Broken)
    result=f.run_episode(np.zeros((3,4389)),f.PongEvaluationConfig(seed=1,episodes=1))
    assert result['failed'] and not result['observations_exact'] and result['ambiguous_last_step']
    assert result['transition_camera_upper_bound']==result['initial_camera_upper_bound']==1


def test_no_grant_and_invalid_holds_before_observations(monkeypatch):
    class Empty:
        def __init__(self,*a,**k):self.complete=False;self.transitions=0;self.initial_learned_hash='test'
        def verify_frozen(self):pass
        def summary(self):return dict(complete=False,transitions=0,initial_cameras=0)
        def step(self,**k):raise AssertionError('must not issue camera')
    monkeypatch.setattr(f,'FreshPongEvaluator',Empty)
    r=f.run_episode(np.zeros((3,4389)),f.PongEvaluationConfig(seed=1,episodes=1),max_transitions=0)
    assert r['observations_exact'] and r['initial_camera_upper_bound']==r['transition_camera_upper_bound']==0
    with pytest.raises(ValueError):f.run_episode(np.zeros((3,4389)),f.PongEvaluationConfig(seed=1,episodes=1,action_hold_frames=2))


def test_all_episode_accounting_ignores_duplicate_dispatch_and_charges_loss(tmp_path):
    (tmp_path/'episodes').mkdir()
    j=dict(id='j1',camera_grant=500);lost=dict(id='j2',camera_grant=500)
    (tmp_path/'dispatch.jsonl').write_text('\n'.join(__import__('json').dumps(dict(job=v)) for v in (j,j,lost)))
    t.atomic_json(tmp_path/'episodes/j1.json',dict(id='j1',purpose='train',transitions=12,initial_cameras=1,complete=True,observations_exact=True))
    a=t.accounting(tmp_path)
    assert a['actual_observed']['train']==12 and a['actual_observed']['initials']==1 and a['unacknowledged_or_ambiguous_observation_upper_bound']==501 and not a['observations_exact']

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from pong_safe_mutations import harmonic,disagreement,mutation_scale,balanced_buffer
import run_practical_pong_v2 as t


def test_harmonic_avoids_three_for_one_trade_and_anchor_dominates():
    assert harmonic(14/21,21/25)>harmonic(15/21,18/25)
    bank=dict(left=[1,2],right=[3,4,5,6]);score=t.phases(bank,{'1':1,'2':1,'3':1,'4':1,'5':0,'6':0})
    assert score['key']==[harmonic(1.,.5),.75]
    incumbent=dict(label='incumbent',score=dict(key=[.6,.7]));anchor=dict(label='anchor',score=dict(key=[.7,.75]))
    assert t.choose([incumbent,anchor]) is anchor
    assert t.choose([anchor,dict(label='combined',score=anchor['score'])]) is anchor
    assert harmonic(0.,0.)==harmonic(0.,1.)==0.


def test_forward_disagreement_monotone_and_capped_without_rewards():
    theta=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
    direction=np.array([[-2.,0.],[2.,1.],[0.,-1.]])
    X=np.array([[1.,0.],[1.,.2],[.2,1.],[.9,1.],[.5,.4]]);w=np.full(5,.2)
    fractions=[disagreement(theta,direction,X,s,w) for s in np.linspace(0,1,50)]
    assert all(b>=a for a,b in zip(fractions,fractions[1:]))
    scale,info=mutation_scale(theta,direction,X,w,.1,maximum=1.)
    assert 0<=scale<=1 and info['achieved']>=.1 and not info['cap_limited']
    s,small=mutation_scale(theta,direction,X,w,.9,maximum=.01)
    assert s==.01 and small['cap_limited']


def test_buffer_weights_equal_phase_then_episode_not_long_trajectory():
    cases=[dict(phase='left',X=np.ones((1,2))),dict(phase='left',X=np.ones((3,2))*2),dict(phase='right',X=np.ones((8,2))*3)]
    X,w=balanced_buffer(cases)
    assert np.isclose(w[:1].sum(),.25) and np.isclose(w[1:4].sum(),.25) and np.isclose(w[4:].sum(),.5)
    assert X.shape==(12,2) and np.isclose(w.sum(),1.)


def test_cache_canonical_defaults_match_actual_prior_receipt():
    import json
    receipt=next(json.loads(p.read_text()) for p in (t.PREVIOUS/'episodes').glob('initial_left_*.json'))
    minimal={k:receipt['config'][k] for k in ('seed','episodes','max_episode_ticks','policy_kind','action_hold_frames','opponent_speed','policy_scaling')}
    assert t.config_key(minimal)==t.config_key(receipt['config'])
    cache=t.previous_cache()
    assert (receipt['weights_sha256'],t.config_key(minimal)) in cache
    assert cache[(receipt['weights_sha256'],t.config_key(minimal))]['hit']==int(receipt['report']['hits']>0)

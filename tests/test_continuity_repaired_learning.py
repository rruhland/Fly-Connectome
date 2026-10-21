"""Fixed allocation and generalized paired evidence keep old defaults intact."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_continuity_repaired_learning as study
import run_sustained_exploration as historical
from return_metrics import paired_evidence


def test_only_vision_changes_from_historical_independent_learner_config():
    learning,evaluation=study.declared_jobs()
    previous,_=historical.declared_jobs()
    old={k:v for k,v in previous[0].items() if k!='id'}
    assert {k:v for k,v in learning[0].items() if k!='id'}==old
    assert len(learning)==1 and learning[0]['transitions']==24000
    assert len(evaluation)==6*32
    serves=study.serve_schedule()
    assert len(set(serves))==32 and not set(serves)&set(study.base.FINAL)
    assert not set(serves)&set(historical.serve_schedule())
    for job in evaluation:
        assert job['config']['seed']==serves[job['serve_index']]
        assert job['config']['episodes']==1 and job['config']['max_episode_ticks']==1200
        assert job['config']['opponent_speed']==.65 and 'reward_shaping' not in job
        assert job['config']['skip_baseline_perception']==(job['policy']!='corrected')
    for i in range(32):
        random=[j for j in evaluation if j['serve_index']==i and j['policy'].startswith('random')]
        assert random[0]['config']['action_seed']==random[1]['config']['action_seed']


def test_32_serve_and_variable_control_evidence_checks_every_first_trajectory():
    names=('corrected','initial','up','noop','random1','random10','extra')
    data={name:[dict(own_returns=int(name=='corrected'),own_opportunities=1,
        first_incoming=dict(success=name=='corrected',tick=50,
            pre_ball=[.064,.5],pre_velocity=[-.5,.1])) for _ in range(32)] for name in names}
    kwargs=dict(arms=('corrected',),expected_serves=32,draws=100,controls=names[1:])
    result=paired_evidence(data,**kwargs)
    assert result['serve_clusters']==32 and result['arms']['corrected']['promising_development_signal']
    data['extra']=copy.deepcopy(data['corrected'])
    assert not paired_evidence(data,**kwargs)['arms']['corrected']['promising_development_signal']
    data['extra'][31]['first_incoming']['tick']=51
    with pytest.raises(ValueError,match='trajectory differs'):paired_evidence(data,**kwargs)
    data['extra'].pop()
    assert paired_evidence(data,**kwargs)==dict(complete=False)


def test_legacy_24_serve_default_statistics_are_unchanged():
    names=('hold1','hold10','initial','up','noop','random1','random10')
    data={name:[dict(own_returns=int(i%3==0),own_opportunities=1+(i%5==0),
        first_incoming=dict(success=i%3==0,tick=50,
            pre_ball=[.064,.5],pre_velocity=[-.5,.1])) for i in range(24)] for name in names}
    assert paired_evidence(data,draws=100)==paired_evidence(data,draws=100,
        expected_serves=24,controls=('initial','up','noop','random1','random10'))

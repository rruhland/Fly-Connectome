from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_reward_diagnostic as study

def test_only_reward_differs_in_fixed_three_arm_allocation():
    jobs=study.declared_jobs();learning=jobs[:3];evaluation=jobs[3:]
    assert len(learning)==len(evaluation)==3 and sum(j['transitions'] for j in learning)==36000
    expected=[(0.,0.),(1.,0.),(0.,.1)]
    normalized=[]
    for job,coefficients in zip(learning,expected):
        reward=job['reward_shaping']
        assert (reward['own_hit_bonus'],reward['survival_per_second'])==coefficients
        assert job['checkpoints']==[12000] and job['seed']==172900127
        normalized.append({k:v for k,v in job.items() if k not in ('id','reward_shaping')})
    assert normalized[0]==normalized[1]==normalized[2]
    for job in evaluation:
        assert 'reward_shaping' not in job and job['config']['policy_kind']=='trained'
        assert job['config']['seed']==904271833 and job['config']['episodes']==24
        assert job['config']['opponent_speed']==.25 and job['config']['max_episode_ticks']==2400
    assert study.TRAIN_SECONDS==720. and study.TOTAL_SECONDS==1500.

"""Explicitly labeled scalar diagnostic rewards; never policy observations."""
import math

REWARD_VERSION='pong-diagnostic-scalar-reward-v1'

def validate_reward(config):
    keys={'version','own_hit_bonus','survival_per_second'}
    if (not isinstance(config,dict) or set(config) not in (keys,keys|{'score_coefficient'})
            or config['version']!=REWARD_VERSION
            or any(type(config[k]) not in (int,float) or not math.isfinite(config[k]) or config[k]<0
                   for k in ('own_hit_bonus','survival_per_second'))
            or ('score_coefficient' in config and (type(config['score_coefficient']) not in (int,float)
                or not math.isfinite(config['score_coefficient']) or not 0<=config['score_coefficient']<=1))):
        raise ValueError('explicit finite nonnegative diagnostic reward coefficients required')

def learning_reward(result,config):
    """Raw terminal game score stays separate; no clipping, fading or relabeling."""
    validate_reward(config)
    hit=config['own_hit_bonus']*result.hits
    survival=config['survival_per_second']*result.elapsed_seconds
    score=result.reward if 'score_coefficient' not in config else config['score_coefficient']*result.reward
    value=score+hit+survival
    if not all(math.isfinite(v) for v in (hit,survival,value)):
        raise FloatingPointError('nonfinite diagnostic reward')
    components=dict(learning_reward=value,hit_reward=hit,survival_reward=survival)
    if 'score_coefficient' in config:components['score_reward']=score
    return value,components

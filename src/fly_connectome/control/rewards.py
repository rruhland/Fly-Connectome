"""Explicitly labeled scalar diagnostic rewards; never policy observations."""
import math

REWARD_VERSION='pong-diagnostic-scalar-reward-v1'

def validate_reward(config):
    if (not isinstance(config,dict) or set(config)!={'version','own_hit_bonus','survival_per_second'}
            or config['version']!=REWARD_VERSION
            or any(type(config[k]) not in (int,float) or not math.isfinite(config[k]) or config[k]<0
                   for k in ('own_hit_bonus','survival_per_second'))):
        raise ValueError('explicit finite nonnegative diagnostic reward coefficients required')

def learning_reward(result,config):
    """Raw terminal game score stays separate; no clipping, fading or relabeling."""
    validate_reward(config)
    hit=config['own_hit_bonus']*result.hits
    survival=config['survival_per_second']*result.elapsed_seconds
    value=result.reward+hit+survival
    if not all(math.isfinite(v) for v in (hit,survival,value)):
        raise FloatingPointError('nonfinite diagnostic reward')
    return value,dict(learning_reward=value,hit_reward=hit,survival_reward=survival)

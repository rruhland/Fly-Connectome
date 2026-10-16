"""Fixed radial policy-input cap; raw transport and greedy policy class survive."""
import copy
import hashlib
import json
import numpy as np

CAP_VERSION='fixed-radial-l2-cap-v1:identity-below:after-base'


def validate_cap(config):
    try:
        valid=(config['version']==CAP_VERSION
               and isinstance(config['max_norm2'],(int,float))
               and np.isfinite(config['max_norm2']) and config['max_norm2']>0
               and 'base_transform' in config and isinstance(config['calibration'],dict))
        json.dumps(config,sort_keys=True,allow_nan=False)
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('declared finite positive fixed radial cap required')
    from .scaling import validate_scaling
    validate_scaling(config['base_transform'])


def cap_values(values,max_norm2):
    """Identity below the bound; stable for arbitrarily large finite inputs."""
    if not np.isfinite(max_norm2) or max_norm2<=0:raise ValueError('finite positive squared norm bound required')
    raw=np.asarray(values)
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('finite policy inputs required')
    largest=float(np.max(np.abs(values),initial=0.))
    if not largest:return raw
    relative=values/largest
    relative_norm=float(np.sqrt(relative@relative))
    bound=float(np.sqrt(max_norm2))
    if largest<=bound/relative_norm:return raw
    return relative*(bound/relative_norm)


class FixedRadialCap:
    def __init__(self,layout,dimension,config):
        validate_cap(config)
        from .scaling import make_policy_transform
        self.config=copy.deepcopy(config)
        self.max_norm2=float(config['max_norm2'])
        self.base=make_policy_transform(layout,dimension,config['base_transform'])
        raw=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.configuration_hash=hashlib.sha256(raw).hexdigest()

    def features(self,encoded):
        from .scaling import policy_features
        indices,values=policy_features(encoded,self.base)
        return indices,cap_values(values,self.max_norm2)

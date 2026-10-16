"""Fixed, score-free conditioning at the policy boundary; raw transport is intact."""
import copy
import hashlib
import json
import numpy as np

SCALING_VERSION = 'fixed-block-rms-v1:energy-matched:uncentered'
BLOCKS = ('current','future','future_outside','relations','context','metadata')


def block_squared_norms(encoded):
    indices=encoded['indices'].numpy(); values=encoded['values'].numpy().astype(np.float64)
    return {name:float(values[(indices>=start)&(indices<end)] @ values[(indices>=start)&(indices<end)])
            for name,(start,end) in encoded['layout'].items()}


def energy_matched_scales(mean_squared_norms, *, minimum=.125, maximum=8.):
    names=tuple(mean_squared_norms)
    means=np.array([mean_squared_norms[name] for name in names],dtype=np.float64)
    total=float(means.sum())
    if (not len(means) or not np.isfinite(means).all() or not np.isfinite(total)
            or (means<0).any() or not 0<minimum<=1<=maximum):
        raise ValueError('finite nonnegative block energies and bounds spanning one required')
    active=means>0
    if not active.any():return {name:1. for name in names}
    fractions=means/total
    def scales(multiplier):
        result=np.ones_like(means)
        result[active]=np.clip(multiplier/np.sqrt(fractions[active]),minimum,maximum)
        return result
    low,high=0.,1.
    for _ in range(80):
        middle=(low+high)/2
        if float(fractions @ scales(middle)**2)<1.:low=middle
        else:high=middle
    return {name:float(value) for name,value in zip(names,scales((low+high)/2))}


def validate_scaling(config):
    if config is None:return
    if not isinstance(config,dict):raise ValueError('declared transform configuration required')
    from .capping import CAP_VERSION,validate_cap
    if config.get('version')==CAP_VERSION:
        validate_cap(config)
        return
    from .sharing import SHARING_VERSION,validate_sharing
    if config.get('version')==SHARING_VERSION:
        validate_sharing(config)
        return
    try:
        valid=(config['version']==SCALING_VERSION and set(config['scales'])==set(BLOCKS)
               and all(np.isfinite(value) and value>0 for value in config['scales'].values())
               and isinstance(config['calibration'],dict))
        json.dumps(config,sort_keys=True,allow_nan=False)
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('declared finite positive fixed scaling configuration required')


class FixedBlockScaling:
    def __init__(self,layout,dimension,config):
        validate_scaling(config)
        if config is None or config['version']!=SCALING_VERSION or set(layout)!=set(config['scales']):
            raise ValueError('exact declared block configuration required')
        self.config=copy.deepcopy(config)
        raw=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.configuration_hash=hashlib.sha256(raw).hexdigest()
        self.multipliers=np.ones(dimension,dtype=np.float64)
        for name,(start,end) in layout.items():self.multipliers[start:end]=config['scales'][name]
        self.multipliers.flags.writeable=False

    def features(self,encoded):
        indices=encoded['indices'].numpy()
        return indices,encoded['values'].numpy()*self.multipliers[indices]


def policy_features(encoded,scaler=None):
    if scaler is not None:return scaler.features(encoded)
    return encoded['indices'].numpy(),encoded['values'].numpy()


def make_policy_transform(layout,dimension,config):
    if config is None:return None
    validate_scaling(config)
    from .sharing import SHARING_VERSION,FixedSpatialSharing
    from .capping import CAP_VERSION,FixedRadialCap
    cls=(FixedRadialCap if config['version']==CAP_VERSION else
         FixedSpatialSharing if config['version']==SHARING_VERSION else FixedBlockScaling)
    return cls(layout,dimension,config)


def policy_block_squared_norms(encoded,scaler=None):
    indices,values=policy_features(encoded,scaler)
    values=values.astype(np.float64)
    return {name:float(values[(indices>=start)&(indices<end)] @ values[(indices>=start)&(indices<end)])
            for name,(start,end) in encoded['layout'].items()}

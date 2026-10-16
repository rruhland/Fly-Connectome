"""Invertible fixed spatial credit sharing at the policy boundary.

No fitted readout, score, entity identity or simulator variable enters this map.
The raw structured transport and its sparse values remain unchanged.
"""
import copy
import hashlib
import json
import numpy as np

SHARING_VERSION='fixed-spatial-sharing-v1:16x16:identity+mean2+mean4:energy-matched'
SPATIAL_BLOCKS=('current','future')
POINT_NORMALIZER=np.sqrt(2.0625)


def project(planes,side):
    count,height,width=planes.shape
    return planes.reshape(count,height//side,side,width//side,side).mean((2,4)).repeat(side,1).repeat(side,2)


def share(planes):
    planes=np.asarray(planes,dtype=np.float64)
    return (planes+project(planes,2)+project(planes,4))/POINT_NORMALIZER


def unshare(planes):
    planes=np.asarray(planes,dtype=np.float64)
    return POINT_NORMALIZER*(planes-.5*project(planes,2)-project(planes,4)/6)


def energy_compensation(mean_raw,mean_shared):
    result={}
    for name in SPATIAL_BLOCKS:
        raw,shared=float(mean_raw[name]),float(mean_shared[name])
        if not np.isfinite([raw,shared]).all() or raw<0 or shared<0 or (raw==0)!=(shared==0):
            raise ValueError('matching finite nonnegative observed block energies required')
        result[name]=float(np.sqrt(raw/shared)) if raw else 1.
    return result


def validate_sharing(config):
    try:
        valid=(config['version']==SHARING_VERSION
               and set(config['block_compensation'])==set(SPATIAL_BLOCKS)
               and all(np.isfinite(v) and v>0 for v in config['block_compensation'].values())
               and isinstance(config['calibration'],dict))
        json.dumps(config,sort_keys=True,allow_nan=False)
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('declared finite fixed spatial-sharing configuration required')


class FixedSpatialSharing:
    def __init__(self,layout,dimension,config):
        validate_sharing(config)
        self.config=copy.deepcopy(config)
        self.layout=copy.deepcopy(layout)
        self.dimension=dimension
        for name in SPATIAL_BLOCKS:
            start,end=layout[name]
            if (end-start)%256:raise ValueError('complete 16x16 spatial planes required')
        raw=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.configuration_hash=hashlib.sha256(raw).hexdigest()
        self.compensation=tuple(config['block_compensation'][name] for name in SPATIAL_BLOCKS)

    def features(self,encoded):
        dense=np.zeros(self.dimension,dtype=np.float64)
        dense[encoded['indices'].numpy()]=encoded['values'].numpy()
        for name,compensation in zip(SPATIAL_BLOCKS,self.compensation):
            start,end=self.layout[name]
            dense[start:end]=compensation*share(dense[start:end].reshape(-1,16,16)).ravel()
        indices=np.flatnonzero(dense)
        return indices,dense[indices]

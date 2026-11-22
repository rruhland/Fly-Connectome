"""Rescore many within-episode source-label shuffles on actual held SO-101 outcomes."""

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_action import CHECK, WIDTH
from .tt23_anonymous_target import target_context
from .tt23_optical_contact import OUT


def run():
    path=OUT/'anonymous-action-arm.json.gz'
    physical=_read(path)
    source=_read(CHECK)
    rows=physical['rows']
    ready=MotorPrimitive(np.array(_read(MACRO)['parameters']),None).ready_world
    query=[target_context(r['sight'],ready) for r in rows]
    seen=np.array([x is not None for x in query])
    grid=np.array([np.zeros(4) if x is None else x for x in query])
    pos=np.array(source['model']['x'])
    w=np.exp(-.5*np.minimum(100.,np.sum(((grid[:,None]-pos)/WIDTH)**2,axis=2)))
    gains=np.array(source['model']['value'])[::2]
    target=np.array([[[a['contacts']>0,a['legal']>0] for a in r['options']] for r in rows])
    def select(value):
        labels=np.stack([value,value[:,::-1]],axis=1).reshape(len(pos),-1)
        choices=np.argmax(w@labels,axis=1)
        choices[~seen]=3
        return choices
    expected=np.array([r['choices']['learned'] for r in rows])
    assert np.array_equal(select(gains),expected)
    actual=target[np.arange(len(rows)),expected].sum(0)
    rng=np.random.default_rng(2086921004)
    trials=[]
    for _ in range(1024):
        choices=select(np.take_along_axis(gains,rng.random(gains.shape).argsort(axis=1),axis=1))
        trials.append(target[np.arange(len(rows)),choices].sum(0))
    trials=np.array(trials)
    results=dict(arm_hash=_hash(path),source_hash=_hash(CHECK),source=actual.tolist(),
                 null=trials.tolist(),null_mean=trials.mean(0).tolist(),null95=np.quantile(trials,[.025,.975],axis=0).tolist(),
                 one_sided_p=((1+(trials>=actual).sum(0))/(1+len(trials))).tolist())
    _write(OUT/'anonymous-action-null.json.gz',results)
    print({k:v for k,v in results.items() if k!='null'})


if __name__=='__main__':
    run()

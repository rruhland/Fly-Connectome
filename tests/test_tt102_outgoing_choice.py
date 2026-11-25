import numpy as np

from scripts.experimental import tt102_outgoing_choice as study
from scripts.experimental.tt4_transfer_trial import _hash, _read


def test_extractor_uses_both_arcs_and_no_post_forecast_reply():
    path=np.full((80,17),np.nan)
    for k in (7,8,9):
        t=(k-9.5)*study.DT
        path[k,:6]=[.1+.22*t,.62+3*t,.778-2.9*t-4.905*t*t,.22,3.,-2.9-9.81*t]
    for k in (10,11,12):
        t=(k-9.5)*study.DT
        path[k,:6]=[.1+.2*t,.62+2.5*t,.778+2.2*t-4.905*t*t,.20,2.5,2.2-9.81*t]
    f=study.from_trace(path,9.5*study.DT)
    assert f is not None
    np.testing.assert_allclose(f[[0,1,5,6]],[.1,.62,.1,.62],atol=.001)
    assert f[4]<-2.0 and f[9]>1.8
    path[10:13]=np.nan
    assert study.from_trace(path,9.5*study.DT) is None


def test_nominal_forecast_never_train_on_prospective_roster():
    p=_read(study.PLAN)
    assert p['fit']==_hash(study.MODEL)
    assert set(p['calibration_ids']).isdisjoint(p['assessment_ids'])
    assert set(p['training_ids']).isdisjoint(p['calibration_ids']+p['assessment_ids'])
    a=_read(study.ACHIEVED)
    assert a['usable']==17 and a['protocol']==_hash(study.PLAN)

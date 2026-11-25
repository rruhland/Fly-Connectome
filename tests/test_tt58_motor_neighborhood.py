import numpy as np

from scripts.experimental import tt58_motor_neighborhood as motor
from scripts.experimental import tt58_motor_neighborhood_plan as plan


def entry(depth_y=-.62):
    return {
        'origin': {'id': 42},
        'p': [0.,depth_y,1.,0.,-3.,1.],
        'q': np.zeros(6), 'ref': np.zeros(6),
        'projected': [0.,1.,0.,-3.,0.],
        'depth': -1.52, 'impact_q': np.zeros(6),
        'refs': np.linspace(.01,.26,26)[:,None]*np.ones((1,6)),
    }


def target(eta=.25):
    return np.array([eta,0.,-1.52,1.,0.,-3.,0.,0.,0.,0.])


def test_anchor_resamples_a_measured_reference_path_and_rejects_large_gap():
    donor=entry()
    same=motor.synthesize(donor,np.zeros(6),np.zeros(6),target(),target(),None,'anchor')
    delayed=motor.synthesize(donor,np.zeros(6),np.zeros(6),target(.25+motor.mpc.DT),target(),None,'anchor')
    assert np.allclose(same,donor['refs'])
    assert np.allclose(delayed[1:],same[:-1])
    assert motor.synthesize(donor,np.zeros(6),np.full(6,0.8),target(),target(),None,'anchor') is None
    assert motor.synthesize(donor,np.zeros(6),np.zeros(6),target(.60),target(),None,'anchor') is None


def test_bank_is_keyed_by_stereo_and_encoder_not_release_id():
    a=entry()
    b=entry()
    b['origin']={'id':8}
    b['q']=np.full(6,.28)
    b['ref']=b['q'].copy()
    bank=motor.MotorBank([a,b])
    ref=np.zeros(6)
    x=np.array(a['p'])
    assert bank.candidates(x,ref,ref,target(),'bank')[0][1]['origin']['id']==42
    assert bank.candidates(x,b['q'],b['ref'],target(),'bank')[0][1]['origin']['id']==8
    x[1]=-.52
    assert not bank.candidates(x,ref,ref,target(),'bank')


def test_frozen_held_and_train_use_distinct_actual_far_strokes_and_bounces():
    protocol=plan._read(plan.PLAN)
    scouts=plan._read(plan.OUT)
    assert scouts['complete']
    assert scouts['protocol_sha256']==plan._hash(plan.PLAN)
    assert len(scouts['rows'])==3
    for row in scouts['rows']:
        train=row['assigned']['practice']; held=row['assigned']['held']
        assert len(train)==4 and len(held)==3
        assert not {r['seed'] for r in train}&{r['seed'] for r in held}
        assert not {r['exact_offer'] for r in train}&{r['exact_offer'] for r in held}

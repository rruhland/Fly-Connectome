from pathlib import Path

import numpy as np

from scripts.experimental import tt88_inverse_body as study
from scripts.experimental.tt4_transfer_trial import _hash, _read


def test_patch_velocity_includes_the_actual_offset_of_the_rotating_blade():
    value=np.r_[np.zeros(10),[1.,2.,3.],[0.,1.,0.],[0.,2.,0.],[.1,.2,.3]]
    goal=study.world_goal(value,(.05,.03))
    np.testing.assert_allclose(goal,[1.05,2.,3.03,0.,1.,0.,.16,.2,.20])


def test_broad_body_receipt_and_no_native_position_discrepancy_on_stock():
    p=_read(study.BENCH)
    assert p['complete']
    assert p['model']==_hash(study.MODEL)
    receipt=_read(study.ROOT/'training-receipt.json')
    assert receipt['motor_sha256']==_hash(study.MODEL)
    assert receipt['physical_advances']>1_000_000
    assert receipt['independent_rows']==2880
    native=[r for r in p['rows'] if r['kind']=='9254-attained']
    assert len(native)==6
    assert max(r['full_native_error_q'] for r in native)<.0025
    for r in p['rows']:
        for key in ('baseline_rates','learned_rates'):
            assert np.max(abs(np.asarray(r[key])))<=.8
    held=[r for r in p['rows'] if r['kind']=='held-babble']
    assert len({r['episode'] for r in held})==12 and len(held)==48
    measures=('weighted','pos_m','angle_deg','velocity_mps')
    for key in measures:
        base,learned=[np.median([r[arm][key] for r in held]) for arm in ('baseline','learned')]
        assert learned<.75*base


def test_learned_motor_transfer_gets_real_repeatable_mounted_routine_answers():
    root=study.ROOT/'installed'
    bf=_read(root/'body-benchmark.json')
    assert bf['complete'] and len(bf['rows'])==30
    groups=('kinematic','centered_reused','mounted_learned')
    for k in ('weighted','pos_m','velocity_mps'):
        a,b,c=[np.median([r['achieved'][g][k] for r in bf['rows']]) for g in groups]
        assert c<b<a
    proof=_read(root/'public-matched.json.gz')
    assert proof['complete']
    routine=next(r for r in proof['rows'] if r['role']=='routine')
    a,b,c=routine['assay']
    assert a['issued'] and not a['landed']
    for x in (b,c):
        assert x['issued'] and x['landed'] and x['answered'] and not x['clearance']['penetration_samples']
        assert x['contact']['radial_m']<=.058
        assert x['repeat_sha256']==study.tt72.digest_json(x['events'])

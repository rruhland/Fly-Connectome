import numpy as np

from scripts.experimental import tt72_retarget_trajectory as study
from scripts.experimental.tt4_transfer_trial import _read


def test_teacher_receipt_has_earned_exact_replays_and_real_rgb_samples():
    d = _read(study.DONORS)
    assert d['complete']
    assert tuple(x['case']['id'] for x in d['rows']) == study.IDS
    _, archived = study.receipt()
    for x in d['rows']:
        assert any(row['rgb_pv'] is not None for row in x['capture'][x['fork']:])
        t = next(y for y in archived[x['case']['id'], .12]['motion']['physical']
                 if study.qual.received(y, True))
        assert x['event_sha256'] == t['repeat_sha256']
        assert x['impact']['time'] == t['contact']['time_s']
        assert x['impact']['p'] == t['contact']['racket']


def test_retarget_preserves_donor_geometry_and_twist_at_contact():
    for x in _read(study.DONORS)['rows']:
        course = study.Course(x)
        p, R, T = course.target(np.array([course.contact_time]), 0., 1.)
        np.testing.assert_allclose(p[0], x['impact']['p'], atol=1e-10)
        np.testing.assert_allclose(R[0], x['impact']['R'], atol=1e-10)
        np.testing.assert_allclose(T[0], x['impact']['twist'], atol=1e-10)
        shifted_p, shifted_R, shifted_T = course.target(
            np.array([course.contact_time + 2 * study.DT]), 2 * study.DT, .60)
        np.testing.assert_allclose(shifted_R, R, atol=1e-10)
        np.testing.assert_allclose(shifted_T, T, atol=1e-10)
        assert np.linalg.norm(shifted_p - p) > .10


def test_stock_trajectories_respect_delivery_and_motor_rate_contract():
    _, archived = study.receipt()
    for p in study.STOCK, study.ROOT / 'equivalence.json.gz':
        record = _read(p)
        assert record['complete']
        assert tuple(x['case']['id'] for x in record['rows']) == study.IDS
        for x in record['rows']:
            t = next(y for y in archived[x['case']['id'], .12]['motion']['physical']
                     if study.qual.received(y, True))
            for test in x['physically_tested']:
                assert study.first_ball(test, t['placements']['delivered'][0])
                assert test['same'] and test['identical_far_contact_and_bounce']
                assert np.abs(test['commands']).max() <= .8 + 1e-10
                assert test['clearance']['samples'] > 0
                if study.qual.received(test, True):
                    assert test['repeat_sha256'] == study.digest_json(test['events'])

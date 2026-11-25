import numpy as np

from scripts.experimental import tt77_centered_impulse as impulse
from scripts.experimental.tt4_transfer_trial import _read


def test_advancing_a_centered_tape_does_not_move_motor_before_received_preparation():
    ref = np.broadcast_to(np.linspace(-.6, .7, 100)[:, None], (100, 6)).copy()
    bases = [dict(ref=ref, at=42.)]
    raw = np.zeros((2, len(impulse.MAG)))
    raw[1, -1] = 3. / impulse.MAG[-1]
    poses = impulse.schedules(bases, [0, 0], raw)
    np.testing.assert_array_equal(poses[0], ref)
    np.testing.assert_array_equal(poses[0, :14], poses[1, :14])
    assert poses[1, 41, 0] > poses[0, 41, 0]


def test_native_soft_reward_counts_the_answered_stroke_not_the_service_far_bounce():
    r = dict(same=True, landed=True, answered=True, clearance=dict(penetration_samples=0),
             contact=dict(radial_m=.030, face_alignment=1., time_s=1.3), depart=[.0, 3.3, 2.5],
             events=[dict(kind='legal_return', side=1, time=.5, ball=[0., 1.15, .78]),
                     dict(kind='legal_return', side=1, time=2., ball=[0., .20, .78])])
    assert not impulse.actual_soft(r)
    r['events'][-1]['ball'][1] = .78
    assert impulse.actual_soft(r)
    r['depart'][1] = 5.
    assert not impulse.actual_soft(r)


def test_completed_practice_keeps_native_stock_delivery_fixed():
    d = _read(impulse.TRAIN)
    assert d['complete']
    sid = d['case']['id']
    _, archive = impulse.retarget.receipt()
    teacher = next(x for x in archive[sid, .12]['motion']['physical']
                   if impulse.qual.received(x, True))
    for outcome in (r for e in d['rounds'] for r in e['physical']):
        assert impulse.retarget.first_ball(outcome, teacher['placements']['delivered'][0])
        assert np.max(np.abs(outcome['commands'])) <= .8 + 1e-10
        assert outcome['actual_soft'] == impulse.actual_soft(outcome)
        if outcome['actual_soft']:
            assert outcome['repeat_sha256'] == impulse.digest_json(outcome['events'])

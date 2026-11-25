import numpy as np

from scripts.experimental import tt57_motor_school as school
from scripts.experimental import tt57_motor_school_transfer as transfer


def test_path_bank_warp_and_crossovers_are_repeatable_and_diverse():
    h = school.H
    zero = np.zeros((h, 6))
    one = zero.copy()
    one[:, 0] = np.linspace(.01, .4, h)
    assert np.array_equal(school.warp(one, 1., 0.), one)
    bank = [(14, one), (88, -one)]
    a, tags = school.initial(zero, bank, np.random.default_rng(19))
    b, tags_b = school.initial(zero, bank, np.random.default_rng(19))
    assert tags == tags_b and np.array_equal(a, b)
    np.testing.assert_equal(a[:4], [zero, one, -one, zero])
    assert len(np.unique(np.round(a[:, (3, 17), 1], 3), axis=0)) > 80
    assert np.all(a[:, :, 5] == 0.)
    assert np.all(np.isfinite(a))


def test_public_bank_retrieves_distinct_releases_and_abstains_distant(tmp_path):
    x = np.array([.01, -.73, .90, .02, -2.8, .65])
    q = np.array([.19, -1.4, -1.3, .96, .6, -.1])
    ref = q+.03
    k = school.context(x, q, ref, np.zeros(6), True)
    n = school.H
    motions = np.zeros((4,n,6))
    motions[0,:,0]=np.linspace(.01,.24,n)
    motions[1,:,0]=np.linspace(.01,.22,n)
    motions[2,:,0]=np.linspace(.01,.20,n)
    motions[3,:,2]=np.linspace(.01,.80,n)
    grid=np.array([x,x+[.005,0,0,0,0,0],x+[.018,0,0,0,0,0],x+[.45,0,0,0,0,0]])
    keys=np.array([school.context(g,q,ref,np.zeros(6),True) for g in grid])
    file=tmp_path/'school.npz'
    np.savez_compressed(file,x=grid,keys=keys,ref=[ref]*4,paths=motions,ids=[17,17,21,300],source='exercise')
    r=transfer.Readout(file)
    paths, meta=r(x,k,ref)
    assert {m['release'] for m in meta} == {17,21}
    assert len(paths)==5
    re=x+[.9,0,0,0,0,0]
    assert not r(re,school.context(re,q,ref,np.zeros(6),True),ref)[0]

import numpy as np

from scripts.experimental import tt90_joint_library as library


def test_body_library_requires_independent_positive_support_and_sham_only_relabels():
    ball=np.array([-.10,-.95,.927,.24,-3.29,1.20])
    q=np.array([.32,-1.72,-1.40,1.72,-1.18,-.1])
    key,clock=library.cue(ball,q,q)
    paths=np.array([np.broadcast_to(q+i/100.,(21,6)) for i in range(4)])
    fields=dict(keys=key+np.arange(4)[:,None]*.012,paths=paths,times=np.repeat(clock,4),
                ref=np.tile(q,(4,1)),releases=np.array([8,8,9,10]),legs=np.array([2,2,4,7]))
    t=library.Library(fields)
    good=t.propose(ball,q,q)
    assert [meta["issued_example_release"] for _,meta in good]==[8,9]
    assert len(good)==2
    assert t.propose(ball+np.array([.65,0,0,0,0,0]),q,q)==[]
    single=library.Library(dict(fields,releases=np.array([8,8,8,8])))
    assert single.propose(ball,q,q)==[]
    sham=library.Library(fields,sham=True)
    assert np.array_equal(np.sort(sham.assign),np.arange(4))
    assert np.array_equal(sham.fields["paths"],t.fields["paths"])
    assert [meta["key_release"] for _,meta in sham.propose(ball,q,q)]==[8,9]


def test_reconstructed_reference_clips_after_each_real_command():
    issued=library.references([[.8]*6,[.8]*6,[-.8]*6],np.zeros(6),np.repeat(.1,6),np.repeat(-.1,6),np.repeat(.1,6))
    assert np.allclose(issued[:,0],[0.,.08,.1,.02])

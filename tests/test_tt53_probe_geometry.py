import numpy as np

from scripts.experimental.tt53_probe_geometry import controls, equal_marginals, signs


def test_sham_preserves_actual_command_rate_marginals_under_clipping():
    rng=np.random.default_rng(7843)
    a=rng.normal(0.,2.3,(2,21,5))
    failure=rng.uniform(-.8,.8,(40,6))
    stride=np.r_[.05,.05,.06,.06,.08,.08]
    lo=np.full(6,-1.)
    hi=np.full(6,1.)
    start=np.array([.96,.973,-.89,.55,.98,.3])
    mask=signs("fixed-causal-prefix")
    assert all(len(np.unique(m))==2 for m in mask)
    u=controls(failure,a,stride,lo,hi,start)
    v=controls(failure,a*mask[:,None,:],stride,lo,hi,start)
    assert equal_marginals(u,v)
    assert np.any(u!=v)


def test_signs_are_frozen_per_independent_fork():
    assert np.array_equal(signs("matched-a"),signs("matched-a"))
    assert not np.array_equal(signs("matched-a"),signs("matched-b"))

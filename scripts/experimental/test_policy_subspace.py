import numpy as np
from pong_policy_subspace import build_basis,map_weights,population,acceptable


def test_basis_contrasts_zero_exact_and_bounded_forward_doses():
    rng=np.random.default_rng(77);base=rng.normal(size=(3,6));direct=base+rng.normal(size=(3,6));original=base+rng.normal(size=(3,6))
    layout={str(i):(i,i+1) for i in range(6)};X=rng.normal(size=(300,6));basis,mapping,info=build_basis(base,direct,original,layout,X)
    assert info['supported_dimension']>0 and np.max(abs(basis.sum(1)))<1e-12
    assert map_weights(base,basis,np.zeros(len(basis))).tobytes()==base.tobytes()
    outputs=np.einsum('ni,kai->nka',X,basis);weights=np.full(len(X),1/len(X));p=population(rng,np.zeros(len(basis)),mapping,outputs,weights,X@base.T)
    assert len(p)==16
    for c in p:
        assert np.max(abs(c['raw_coefficients']))<=1+1e-12 and c['legal_output_rms']<=.15+1e-12 and 0<=c['legal_action_disagreement']<=1


def test_selection_requires_native_harmonic_gain_and_bounded_phase_regression():
    anchor=dict(key=[.90,.90],rates=dict(left=.9,right=.9))
    assert not acceptable(anchor,anchor,anchor)
    assert acceptable(dict(key=[.91,.91],rates=dict(left=.88,right=.95)),anchor,anchor)
    assert not acceptable(dict(key=[.92,.92],rates=dict(left=.86,right=.99)),anchor,anchor)

"""Empty-input override keeps the original temporal step and exact baseline Q."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest
from pong_ogma_prior import OgmaRuntime
from pong_ogma_empty_guard import EmptyCurrentGuard
from pong_fixed_rbf import load_readout

ROOT=Path('/workspace/m1b-experiments/pong-ogma-prior-20261028-r1')
BASE=Path('/workspace/m1b-experiments/pong-rally-joint-output-20261028-r1/full/rbf-model.npz')


def test_temporal_advance_and_actual_action_feedback_preserved():
    baseline=load_readout(BASE);guard=EmptyCurrentGuard(ROOT/'libM1BOgmaPrior.so',baseline,ROOT/'final.bin')
    raw=OgmaRuntime(ROOT/'libM1BOgmaPrior.so',baseline,ROOT/'final.bin')
    X=sp.load_npz(ROOT/'screen8/game_00-raw.npz').tocsr();before=guard.weights();fallbacks=0
    for i in range(12):
        lo,hi=X.indptr[i:i+2];indices=X.indices[lo:hi];values=X.data[lo:hi]
        got=guard.raw_query(indices,values);unmodified=raw.raw_query(indices,values)
        assert got['unguarded_q']==unmodified['q'] and got['unguarded_action']==unmodified['action'] and got['residual']==unmodified['residual'] and got['p']==unmodified['p']
        if got['empty_current_fallback']:
            action,q=baseline.choose(indices,values);assert got['action']==action and np.array_equal(got['q'],q);fallbacks+=1
        else:assert got['action']==unmodified['action'] and got['q']==unmodified['q']
        # Both contexts receive the ACTUAL guarded action, not latent sampled action.
        for runtime in (guard,raw):runtime.feedback(got['action'],got['p'],0,2/120,False)
    assert fallbacks==2 and guard.weights()==before==raw.weights();guard.close();raw.close()


def test_guard_forbids_any_learning():
    m=EmptyCurrentGuard(ROOT/'libM1BOgmaPrior.so',load_readout(BASE),ROOT/'final.bin')
    with pytest.raises(ValueError):m.raw_query(np.array([],dtype=int),np.array([]),learn_actor=True)
    with pytest.raises(ValueError):m.raw_query(np.array([],dtype=int),np.array([]),learn_representation=True)
    m.close()

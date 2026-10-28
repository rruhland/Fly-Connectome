"""Zero-native checks of exact carry, coordinate mapping, reset and local credit."""
from pathlib import Path
import subprocess
import numpy as np
from pong_ogma_prior import OgmaRuntime,pack_current
from pong_fixed_rbf import load_readout

ROOT=Path('/workspace/Fly-Connectome-m1b')
SCRATCH=Path('/workspace/ogmaneo2-feasibility-c5be9')
LIBRARY=SCRATCH/'libM1BOgmaPrior.so'
MODEL=Path('/workspace/m1b-experiments/pong-rally-joint-output-20261028-r1/full/rbf-model.npz')


def test_exact_hand_placed_column_channel():
    channel,y,x=3,1,2
    packed=pack_current(np.array([channel*256+y*16+x,4384]),np.array([.625,999]))
    expected=channel+y*7+x*7*16  # actual Helpers.h::address3
    assert packed[expected]==.625 and np.count_nonzero(packed)==1


def test_zero_residual_literal_near_tie_carry_and_serialization(tmp_path):
    class NearTie:
        def choose(self,indices,values):
            q=np.array([1.,1.,np.nextafter(1.,2.)]);return 2,q
    m=OgmaRuntime(LIBRARY,NearTie());before=m.weights();p=tmp_path/'model.bin';m.save(p)
    n=OgmaRuntime(LIBRARY,NearTie(),p);assert n.weights()==before
    for j in range(12):
        r=m.raw_query(np.array([j]),np.array([.25]))
        assert r['action']==2 and r['q']==[1.,1.,np.nextafter(1.,2.)] and r['residual']==[0.,0.,0.]
    assert m.weights()==before;m.close();n.close()


def test_actual_local_actor_learning_preserves_representation_and_reset(tmp_path):
    baseline=load_readout(MODEL);m=OgmaRuntime(LIBRARY,baseline);rep=m.representation();initial=m.weights()
    for t in range(48):
        r=m.raw_query(np.array([t%16]),np.array([.25]),learn_actor=True)
        m.feedback(r['action'],r['p'],int(t==20),2/120,False)
    m.raw_query(np.array([0]),np.array([.25]),learn_actor=True)
    assert m.representation()==rep and m.weights()!=initial
    path=tmp_path/'trained.bin';m.save(path);n=OgmaRuntime(LIBRARY,baseline,path);assert n.weights()==m.weights()
    def sequence(runtime):
        runtime.reset();out=[]
        for t in range(12):
            r=runtime.raw_query(np.array([t]),np.array([.25]));out.append(r)
            runtime.feedback(r['action'],r['p'],0,2/120,False)
        return out
    assert sequence(m)==sequence(n)==sequence(m);m.close();n.close()


def test_core_probability_discount_terminal_alignment():
    binary=SCRATCH/'prior-test'
    subprocess.run(['g++','-std=c++14','-O2','-fopenmp','-I'+str(ROOT/'scripts/experimental/vendor/ogmaneo2_prior'),str(ROOT/'scripts/experimental/ogma_prior_core_test.cpp'),'-L'+str(SCRATCH),'-lM1BOgmaPrior','-Wl,-rpath,'+str(SCRATCH),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt60_school_bottleneck as study
from scripts.experimental import tt60_school_optical_gate as gate


def test_envelope_counts_a_distinct_terminal_class(monkeypatch):
    monkeypatch.setattr(study.school,'projected_height', lambda p:p['xyz'][2])
    offer = dict(near_landed=False, near_contact=dict(racket=[0.,0.,1.11],location=[0.,0.,1.052],step=100),
                 far_contact=dict(step=50), receive_line=dict(xyz=[0.,-1.1,1.045]))
    row = dict(placements=dict(delivered=[offer]),searches=[dict(tick=80,p=[0.,-.72,1.])],actions=[])
    assert study.group(row,0) == 'school_envelope'
    assert study.group(dict(row,searches=[]),0) == 'school_height_without_search'
    offer['near_contact'] = None
    assert study.group(row,0) == 'receive_no_contact'
    assert study.group(dict(row,placements=dict(delivered=[])),0) == 'no_legal_opponent_delivery'


def make_model(tmp_path):
    x = np.array([.01,-.66,.91,.05,-3.8,1.7])
    f = gate.feature(x,.0)
    model = tmp_path/'test.npz'
    np.savez_compressed(model,features=np.repeat(f[None,:],9,axis=0),
                        phase=[True]*8+[False],release=[15,15,31,44,55,62,70,70,111],
                        residual=np.arange(81).reshape(9,9)/1000,practice='test')
    return x,gate.Gate(model)


def test_uncertainty_uses_distinct_release_same_phase_and_withholds_all_group_rows(tmp_path):
    x, model = make_model(tmp_path)
    errors,w,neighbors = model.neighbors(x,.0,True,excluded=15)
    assert neighbors == [31,44,55,62,70]
    np.testing.assert_allclose(w.sum(),1.)
    np.testing.assert_array_equal(errors, model.residual[[2,3,4,5,6]])
    assert model.neighbors(x,.0,False,excluded=15) is None


def test_public_twin_hypotheses_restore_state_and_allow_nonnominal_support(tmp_path, monkeypatch):
    x, model = make_model(tmp_path)
    t = SimpleNamespace(ball=x[:3].copy(), velocity=x[3:].copy(), spin=np.zeros(3),_ball_dof=0)
    t.data = SimpleNamespace(qvel=np.r_[x[3:],t.spin])
    t.spin = t.data.qvel[3:]
    clinic = SimpleNamespace(sync=lambda:True,twin=t,predict=object())
    calls=[]
    def native(engine, paths, ref):
        assert engine is clinic.predict
        calls.append(np.r_[t.ball,t.velocity,t.spin])
        accept=np.array([len(calls) != 1])
        f=np.zeros((1,9)); f[:,1]=1.
        return None,None,f,np.array([1.]),accept

    monkeypatch.setattr(study.school,'native',native)
    v=model.decisions(clinic,np.zeros((1,26,6)),np.zeros(6),x,0.,True,excluded=15)
    assert v['nominal'] == [False] and v['center'] == [True] and v['support']==[5]
    np.testing.assert_allclose(v['probability'],[.87])
    np.testing.assert_allclose(np.r_[t.ball,t.velocity,t.spin],np.r_[x,np.zeros(3)])

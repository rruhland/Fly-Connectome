from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experimental import tt38_impact_margin as margin


def test_normal_impulse_chooses_departure_along_predicted_face():
    p = np.array([.07, -1.56, 1.07])
    v = np.array([.05, -3.65, .35])
    n, paddle = margin.face_and_velocity(p, v, 4.32, 0., .05)
    goal = margin.target_launch(p, 4.32)
    assert n[1] > .9
    assert 0 < paddle @ n < 1.52
    assert np.isclose((1.76 * paddle - .76 * v) @ n, goal @ n)


def test_margins_prefer_interceptable_deep_landing_over_legal_edge():
    interior = [1., .15, .74, .08, .72, .10, .13, 1.13, 1.05, 3.6, 3.6, 4.7, .021]
    edge = [1., .14, .63, .61, .14, .02, .66, .84, .85, 2.8, 3.6, 3.3, .055]
    feedback = np.array([[0.02, 1., .2, 4.3, 2.2, .1, -1.5, 1.04, .021],
                         [0.07, 1., .5, 3.3, 3.4, .1, -1.5, 1.04, .055],
                         [0.02, 0., 0., 0., 0., 0., 0., 0., 0.]])
    q, valid = margin.quality(np.array([interior, edge, [np.nan]*13]), feedback)
    assert q[0] > q[1] > q[2]
    assert valid.tolist() == [True, True, False]


def test_uncertainty_prefers_plate_interior_and_preserves_optical_state(monkeypatch):
    class Court:
        def __init__(self):
            self.twin = SimpleNamespace(launches=0, card=None, control_dt=1/60,
                ball=np.array([.05,-1.04,1.08]),velocity=np.array([0.,-3.95,.76]),
                _ball_dof=0,data=SimpleNamespace(qvel=np.array([0.,0.,0.,18.,0.,0.])))
            self.latest = object()
            self.previous = np.zeros(6)
            self.calls = 0

        def trajectories(self, goals, aim, cutoff, *, feedback, traces=None):
            self.calls += 1
            if traces is not None:
                traces[:] = 0.
            base = np.array([1.,.14,.8,.58,.21,.028,.62,.84,.89,2.8,3.6,4.,.056])
            rows = np.tile(base,(len(goals),1))
            feedback[:,1] = 1.
            feedback[:,3] = 4.3
            feedback[:,8] = .025
            for i,variant in enumerate(np.rint(goals[:,0,5]+.1).astype(int)):
                if variant == 1 and self.calls > 2:
                    rows[i] = np.nan
                elif variant:
                    rows[i,3:10] = [.02,.74,.105,.08,1.12,1.07,3.6]
            return np.zeros_like(goals),rows

    def fake_build(surface, obs, eta, va, time, legacy, realized):
        g = np.tile(legacy,(3,1,1))
        g[1,:,5] += 1.
        g[2,:,5] += 2.
        return g,[dict(family=x) for x in ('incumbent','normal_sweep','fixed_center_trim')]

    monkeypatch.setattr(margin, 'build', fake_build)
    engine = Court()
    surface = SimpleNamespace(ready=np.zeros(6))
    obs = np.array([.14,.04,1.08,0.,-4.,.7,*([np.nan]*6)])
    fit = np.zeros((2,5))
    p,v,w = engine.twin.ball.copy(),engine.twin.velocity.copy(),engine.twin.data.qvel.copy()
    cfg = margin.ImpactMargin()
    answer = cfg.plan(engine,surface,obs,np.zeros(10),[1.],fit,.14)
    assert answer['note']['status']=='issued'
    assert answer['note']['candidate']['family']=='fixed_center_trim'
    assert answer['note']['candidate_legal_cases']==7
    np.testing.assert_allclose(engine.twin.ball,p)
    np.testing.assert_allclose(engine.twin.velocity,v)
    np.testing.assert_allclose(engine.twin.data.qvel,w)
    engine.twin.card = {'private_release_id': 72}
    with pytest.raises(ValueError, match='physical point'):
        cfg.plan(engine,surface,obs,np.zeros(10),[1.],fit,.14)

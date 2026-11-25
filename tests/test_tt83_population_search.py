import numpy as np

from scripts.experimental import tt83_population_search as opt


def test_earned_prior_is_orthonormal_and_only_controls_five_joints():
    prior=opt.MotorPrior()
    assert prior.basis.shape==(opt.RANK,opt.H*5)
    assert np.allclose(prior.basis@prior.basis.T,np.eye(opt.RANK),atol=1.e-6)
    z=np.random.default_rng(123).normal(size=(4,opt.RANK))
    d=prior.decode(z)
    assert d.shape==(4,opt.H,6) and not d[:,:,5].any()
    assert np.allclose(prior.encode(d),z)


def test_actuator_projection_respects_every_rate_and_range():
    ref=np.zeros(6)
    goal=np.zeros((opt.H,6))
    goal[:9,0]=4.
    goal[9:18,0]=-4.
    lim=np.ones(6)*.30
    stride=np.ones(6)*.13
    path=opt._project_path(goal,ref,-lim,lim,stride)
    assert np.max(abs(path))<=.30
    assert np.max(abs(np.diff(np.r_[ref[None],path],axis=0)))<=.8*.13+1e-12
    assert path[3,0]>.14 and path[15,0]<-.15


def test_elites_keep_a_noncontact_subpopulation():
    paths=np.zeros((5,opt.H,6))
    paths[:,12:]=np.arange(5)[:,None,None]*.15
    scores=np.array([5.0,4.0,4.5,1.0,20.])
    f=np.zeros((5,9)); f[:4,1]=1.
    assert opt._elite(paths,scores,f,n=3)[:2]==[3,4]


def test_public_observer_adapter_decays_optical_estimate_and_falls_back():
    from types import SimpleNamespace
    from scripts.experimental.tt83_observer_adapter import PublicTracker

    lead=np.array([.04,-.8,.89,.1,-4.,1.])
    spin=np.array([20.,30.,-13.])
    when=1.2
    def state(t):
        x=lead.copy()
        x[:3] += (t-when)*x[3:]
        return x,spin*(1-(t-when))
    fit=SimpleNamespace(time=when,bounce=when-.025,state=state)
    old=SimpleNamespace(latest="legacy",filter=SimpleNamespace(rotation=lambda t:np.zeros(3)))
    bridge=PublicTracker(old)
    assert np.array_equal(bridge.update(when,None,(state(when),fit)),lead)
    assert bridge.latest.bounced and bridge.latest.time==when
    assert np.allclose(bridge.latest.advance(when+.01)[0],state(when+.01)[0][:3])
    assert np.allclose(bridge.rotation(when+.01),state(when+.01)[1])
    x=np.zeros(6)
    assert bridge.update(when+.28,x,(state(when+.28),fit)) is x
    assert bridge.latest=="legacy"


def test_sealed_population_comparison_matches_the_first_batch_and_total_budget():
    from scripts.experimental import tt83_population_eval as trial
    from scripts.experimental.tt4_transfer_trial import _hash,_read

    evidence=_read(trial.HELD)
    assert evidence["complete"] and evidence["plan"]==_hash(trial.PLAN)
    assert len(evidence["rows"])==13
    for row in evidence["rows"]:
        one,two=(row["arms"][k]["search"] for k in trial.MODES)
        assert one["screened"]==two["screened"]==104
        assert one["rounds"][0]==two["rounds"][0]
        assert one["audit"]["candidates"]==two["audit"]["candidates"]==9
        assert one["audit"]["refinement"]==two["audit"]["refinement"]==18
    assert sum(r["arms"]["iterative"]["search"]["unique_legal"] for r in evidence["rows"])==159
    assert sum(r["arms"]["static"]["search"]["unique_legal"] for r in evidence["rows"])==112

import torch

from fly_connectome.graph import Graph
from fly_connectome.dynamics import Network, Activity
from fly_connectome.plasticity import Plasticity, LearningConfig


def setup(batch=1):
    g = Graph.from_contacts([10, 20, 30], [10, 10], [20, 30], [1, 1], [1]*3, .5)
    n = Network(g, [1, 1], ['predictive', 'behavioral'], batch=batch)
    p = Plasticity(n, LearningConfig(eta_prediction=.1, eta_reward=.1, homeostasis_rate=0.))
    return n, p


def activity(batch, observed=0., predicted=0., spikes=True):
    obs = torch.zeros(batch, 3)
    pred = torch.zeros_like(obs)
    obs[:, 1] = observed
    pred[:, 1] = predicted
    return Activity(torch.full((batch, 3), spikes), obs, pred,
                    torch.arange(batch).repeat_interleave(2), torch.tensor([0, 1]).repeat(batch))


def test_confirmed_expired_and_unexpected_prediction_updates():
    for observed, expected in [(1., 0.), (0., -1.)]:
        n, p = setup()
        p.observe(activity(1, predicted=1.), torch.zeros(1))
        p.synchronize()
        before = n.magnitudes.clone()
        p.observe(activity(1, observed=observed), torch.zeros(1))
        p.synchronize()
        change = (n.magnitudes[0] - before[0]).item()
        assert abs(change) < 1e-6 if expected == 0 else change < 0
    n, p = setup()
    p.observe(activity(1, observed=1.), torch.zeros(1))
    p.synchronize()
    assert n.magnitudes[0] > .5
    assert n.magnitudes[1] == .5


def test_reward_only_changes_eligible_behavioral_edges_and_traces_decay():
    n, p = setup()
    p.observe(activity(1), torch.ones(1))
    p.synchronize()
    assert n.magnitudes[0] == .5
    assert n.magnitudes[1] > .5
    prior = p.values.clone()
    silent = Activity(torch.zeros(1, 3, dtype=torch.bool), torch.zeros(1, 3), torch.zeros(1, 3),
                      torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    p.observe(silent, torch.zeros(1))
    assert torch.all(p.values.abs() <= prior.abs())
    assert p.keys.numel() == 2


def test_batched_duplicate_and_permutation_reductions():
    one, p1 = setup()
    two, p2 = setup(2)
    p1.observe(activity(1, observed=1.), torch.ones(1))
    p2.observe(activity(2, observed=1.), torch.ones(2))
    p1.synchronize(); p2.synchronize()
    torch.testing.assert_close(one.magnitudes, two.magnitudes, rtol=0, atol=0)
    a, pa = setup(2)
    b, pb = setup(2)
    pa.observe(activity(2), torch.tensor([.3, -.7]))
    pb.observe(activity(2), torch.tensor([-.7, .3]))
    pa.synchronize(); pb.synchronize()
    torch.testing.assert_close(a.magnitudes, b.magnitudes, rtol=0, atol=0)


def test_eligibility_does_not_change_forward_and_weight_bounds_preserve_signs():
    n, p = setup()
    control, _ = setup()
    p.observe(activity(1, observed=1.), torch.ones(1))
    # Proposals are not applied until the shared snapshot boundary.
    torch.testing.assert_close(n.step(torch.ones(1, 3)).spikes, control.step(torch.ones(1, 3)).spikes)
    p.proposals.fill_(-100.)
    p.synchronize()
    assert n.magnitudes.tolist() == [0., 0.]
    assert n.pre.tolist() == [0, 0] and n.post.tolist() == [1, 2]

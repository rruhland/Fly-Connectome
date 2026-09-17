import torch
import pytest

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


def test_delayed_arrival_pairs_with_later_post_spike():
    g = Graph.from_contacts([10, 20], [10], [20], [1], [1, 1], .5)
    n = Network(g, [5], ['behavioral'])
    p = Plasticity(n, LearningConfig(eta_reward=.1, homeostasis_rate=0.))
    def tick(spikes, arrived=False):
        p.observe(Activity(torch.tensor([spikes]), torch.zeros(1, 2), torch.zeros(1, 2),
            torch.tensor([0] if arrived else [], dtype=torch.long),
            torch.tensor([0] if arrived else [], dtype=torch.long)), torch.zeros(1))
    tick([True, False])
    for _ in range(4):
        tick([False, False])
    tick([False, False], arrived=True)
    tick([False, True])
    p.reward(torch.ones(1))
    p.synchronize()
    assert n.magnitudes[0] > .59


def test_feedforward_observation_does_not_train_its_own_target():
    g = Graph.from_contacts([10, 20], [10], [20], [1], [1, 1], .5)
    n = Network(g, [1], ['feedforward'])
    p = Plasticity(n, LearningConfig(eta_prediction=.1, homeostasis_rate=0.))
    p.observe(Activity(torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0., .5]]),
                       torch.zeros(1, 2), torch.tensor([0]), torch.tensor([0])), torch.zeros(1))
    p.synchronize()
    assert n.magnitudes[0] == .5


def test_homeostasis_uses_simulated_time_not_sync_count():
    g = Graph.from_contacts([10, 20], [10], [20], [1], [1, 1], .5)
    a, b = Network(g, [1], ['behavioral']), Network(g, [1], ['behavioral'])
    cfg = LearningConfig(homeostasis_rate=.1, tau_homeostasis=1e30, maximum_rate=1.)
    pa, pb = Plasticity(a, cfg), Plasticity(b, cfg)
    pa.rates.fill_(10.); pb.rates.fill_(10.)
    silent = Activity(torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2), torch.zeros(1, 2),
                      torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    for _ in range(10):
        pa.observe(silent, torch.zeros(1)); pa.synchronize()
        pb.observe(silent, torch.zeros(1))
    pb.synchronize()
    torch.testing.assert_close(a.magnitudes, b.magnitudes, atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_cuda_plasticity_matches_cpu_and_batch_order():
    g = Graph.from_contacts([10, 20], [10], [20], [1], [1, 1], .5)
    cpu = Network(g, [1], ['behavioral'], batch=3)
    cuda = Network(g, [1], ['behavioral'], batch=3, device='cuda')
    a, b = Plasticity(cpu), Plasticity(cuda)
    for step in range(20):
        x = torch.tensor([[30., 0.], [0., 30.], [30., 30.]]) if step % 3 == 0 else torch.zeros(3, 2)
        a.observe(cpu.step(x), torch.tensor([.1, -.2, .3]))
        b.observe(cuda.step(x.cuda()), torch.tensor([.1, -.2, .3], device='cuda'))
        a.synchronize(); b.synchronize()
    torch.testing.assert_close(cpu.magnitudes, cuda.magnitudes.cpu(), atol=1e-6, rtol=1e-5)


def test_canonical_reduction_handles_cancellation_and_permutation():
    from fly_connectome.plasticity import _sum_sorted
    keys = torch.tensor([0, 0, 0, 1, 1])
    values = torch.tensor([1e5, -1e5, .03, .2, -.1])
    perm = torch.tensor([3, 1, 4, 2, 0])
    torch.testing.assert_close(_sum_sorted(keys, values, 2), _sum_sorted(keys[perm], values[perm], 2), atol=0, rtol=0)

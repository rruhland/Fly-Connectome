"""Local current prediction and R-STDP with sparse private eligibility."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class LearningConfig:
    eta_prediction: float = .0001
    eta_reward: float = .0001
    tau_eligibility: float = 1.
    tau_pair: float = .02
    tau_homeostasis: float = 30.
    homeostasis_rate: float = .00001
    maximum_rate: float = 50.
    maximum_weight: float = 10.
    prune_epsilon: float = 1e-8


def _sum_sorted(keys, values, size):
    """Canonical order within each edge makes reduction independent of batch order."""
    value_order = torch.argsort(values, stable=True)
    order = value_order[torch.argsort(keys[value_order], stable=True)]
    result = torch.zeros(size, device=values.device, dtype=values.dtype)
    result.index_add_(0, keys[order], values[order])
    return result


class Plasticity:
    @torch.no_grad()
    def __init__(self, network, config=LearningConfig()):
        self.network, self.config = network, config
        if min(config.tau_eligibility, config.tau_pair, config.tau_homeostasis, config.maximum_weight) <= 0:
            raise ValueError("positive plasticity timescales and bounds required")
        if min(config.eta_prediction, config.eta_reward, config.homeostasis_rate, config.prune_epsilon) < 0:
            raise ValueError("learning rates and pruning threshold must be nonnegative")
        self.keys = torch.empty(0, dtype=torch.long, device=network.device)
        self.values = torch.empty(0, device=network.device)
        self.pre_trace = torch.zeros_like(network.voltage)
        self.post_trace = torch.zeros_like(network.voltage)
        self.expected = torch.zeros_like(network.voltage)
        self.rates = torch.zeros_like(network.voltage)
        self.proposals = torch.zeros(network.e, device=network.device)

    @torch.no_grad()
    def observe(self, activity, reward):
        n, cfg = self.network, self.config
        if reward.shape != (n.batch,) or not torch.isfinite(reward).all():
            raise ValueError("one finite scalar reward per environment required")
        dt = n.config.dt
        self.values.mul_(math.exp(-dt / cfg.tau_eligibility))
        pair_decay = math.exp(-dt / cfg.tau_pair)
        self.pre_trace.mul_(pair_decay)
        self.post_trace.mul_(pair_decay)
        env, edges = activity.arrival_environments, activity.arrival_edges
        arrival_keys = env * n.e + edges
        keys = torch.unique(torch.cat((self.keys, arrival_keys)), sorted=True)
        values = torch.zeros(len(keys), device=n.device)
        values[torch.searchsorted(keys, self.keys)] = self.values
        arrivals = torch.zeros(len(keys), device=n.device)
        arrivals.index_add_(0, torch.searchsorted(keys, arrival_keys), torch.ones(len(edges), device=n.device))
        environments, edge_ids = keys.div(n.e, rounding_mode='floor'), keys.remainder(n.e)
        post = n.post[edge_ids]
        pre = n.pre[edge_ids]
        behavioral = n.pathways[edge_ids] == 2
        # Causal arrival followed by a spike potentiates; post-before-pre depresses.
        pairing = (activity.spikes[environments, post] * (self.pre_trace[environments, pre] + arrivals)
                   - arrivals * self.post_trace[environments, post])
        values.add_(torch.where(behavioral, pairing, arrivals * n.signs[edge_ids]))
        alive = values.abs() > cfg.prune_epsilon
        self.keys, self.values = keys[alive], values[alive]
        environments, edge_ids, post, behavioral = (x[alive] for x in (environments, edge_ids, post, behavioral))
        # Prediction issued on the preceding tick is tested against this tick's local
        # feedforward current. Recurrent current cannot confirm its own prediction.
        # Current is normalized by the cell's baseline firing threshold, not a learned head.
        observed = (activity.observed / n.config.threshold).clamp(0, 1)
        delta = observed[environments, post] - self.expected[environments, post]
        proposal = torch.where(behavioral,
            cfg.eta_reward * reward.detach().clamp(-1, 1)[environments] * self.values,
            cfg.eta_prediction * delta * self.values)
        self.proposals.add_(_sum_sorted(edge_ids, proposal, n.e) / n.batch)
        self.expected.copy_((activity.predicted / n.config.threshold).clamp(0, 1))
        self.pre_trace.add_(activity.spikes)
        self.post_trace.add_(activity.spikes)
        self.rates.lerp_(activity.spikes.float() / dt, 1 - math.exp(-dt / cfg.tau_homeostasis))

    @torch.no_grad()
    def synchronize(self):
        n, cfg = self.network, self.config
        n.magnitudes.add_(self.proposals).clamp_(0, cfg.maximum_weight)
        overload = (self.rates.mean(0)[n.post] - cfg.maximum_rate).clamp(min=0)
        n.magnitudes.mul_(torch.exp(-cfg.homeostasis_rate * overload))
        self.proposals.zero_()

"""Local current prediction and R-STDP with sparse private eligibility."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class LearningConfig:
    prediction_encoding: str = 'rectified-current-v1'
    visual_target: str = 'filtered-current-v1'
    visual_eligibility: str = 'legacy-v1'
    eta_prediction: float = .0001
    eta_reward: float = .0001
    tau_eligibility: float = 1.
    tau_pair: float = .02
    tau_homeostasis: float = 30.
    homeostasis_rate: float = .00001
    maximum_rate: float = 50.
    maximum_weight: float = 10.
    prune_epsilon: float = 1e-8

    def __post_init__(self):
        if self.prediction_encoding not in ('rectified-current-v1', 'signed-current-v1'):
            raise ValueError('unknown prediction encoding')
        if self.visual_target not in ('filtered-current-v1', 'input-arrivals-v1'):
            raise ValueError('unknown visual target')
        if self.visual_eligibility not in ('legacy-v1', 'forecast-causal-v1'):
            raise ValueError('unknown visual eligibility')

    @property
    def prediction_signature(self):
        return self.prediction_encoding, self.visual_target, self.visual_eligibility

    def encode(self, current, threshold):
        return (current / threshold).clamp(-1 if self.prediction_encoding == 'signed-current-v1' else 0, 1)

    def observation(self, activity, threshold, sensory_mask, sensory_gain):
        if self.visual_target == 'filtered-current-v1':
            return self.encode(activity.observed, threshold)
        if activity.feedforward_arrivals is None or activity.sensory_input is None:
            raise ValueError('input-arrivals target requires captured local increments')
        return torch.where(sensory_mask, self.encode(activity.sensory_input, sensory_gain),
                           self.encode(activity.feedforward_arrivals, threshold))


def _sum_sorted(keys, values, size):
    """Canonical order within each edge makes reduction independent of batch order."""
    value_order = torch.argsort(values, stable=True)
    order = value_order[torch.argsort(keys[value_order], stable=True)]
    result = torch.zeros(size, device=values.device, dtype=values.dtype)
    if values.numel():
        unique, counts = torch.unique_consecutive(keys[order], return_counts=True)
        sums = torch.segment_reduce(values[order], 'sum', lengths=counts)
        result[unique] = sums
    return result


class Plasticity:
    @torch.no_grad()
    def __init__(self, network, config=LearningConfig(), *, sensory_mask=None, sensory_gain=1.):
        self.network, self.config = network, config
        self.sensory_mask = (torch.zeros(network.n, dtype=torch.bool, device=network.device)
                             if sensory_mask is None else sensory_mask.to(network.device))
        self.sensory_gain = sensory_gain
        if min(config.tau_eligibility, config.tau_pair, config.tau_homeostasis, config.maximum_weight) <= 0:
            raise ValueError("positive plasticity timescales and bounds required")
        if min(config.eta_prediction, config.eta_reward, config.homeostasis_rate, config.prune_epsilon) < 0:
            raise ValueError("learning rates and pruning threshold must be nonnegative")
        self.keys = torch.empty(0, dtype=torch.long, device=network.device)
        self.values = torch.empty(0, device=network.device)
        self.arrival_trace = torch.empty(0, device=network.device)
        self.post_trace = torch.zeros_like(network.voltage)
        self.expected = torch.zeros_like(network.voltage)
        self.rates = torch.zeros_like(network.voltage)
        self.proposals = torch.zeros(network.e, device=network.device)
        self.homeostatic_exponent = torch.zeros(network.e, device=network.device)

    @torch.no_grad()
    def observe(self, activity, reward):
        n, cfg = self.network, self.config
        if reward.shape != (n.batch,) or not torch.isfinite(reward).all():
            raise ValueError("one finite scalar reward per environment required")
        dt = n.config.dt
        observed = cfg.observation(activity, n.config.threshold, self.sensory_mask, self.sensory_gain)
        causal = cfg.visual_eligibility == 'forecast-causal-v1'
        if causal:
            old_env, old_edges = self.keys.div(n.e, rounding_mode='floor'), self.keys.remainder(n.e)
            old_post = n.post[old_edges]
            visual = n.pathways[old_edges] == 1
            error = observed[old_env, old_post] - self.expected[old_env, old_post]
            self._accumulate(old_edges[visual], cfg.eta_prediction*error[visual]*self.values[visual])
            self.values.mul_(torch.where(visual, n.current_decay[old_post], math.exp(-dt/cfg.tau_eligibility)))
        else:
            self.values.mul_(math.exp(-dt / cfg.tau_eligibility))
        pair_decay = math.exp(-dt / cfg.tau_pair)
        self.arrival_trace.mul_(pair_decay)
        self.post_trace.mul_(pair_decay)
        env, edges = activity.arrival_environments, activity.arrival_edges
        plastic = n.pathways[edges] != 0
        env, edges = env[plastic], edges[plastic]
        arrival_keys = env * n.e + edges
        keys = torch.unique(torch.cat((self.keys, arrival_keys)), sorted=True)
        values = torch.zeros(len(keys), device=n.device)
        values[torch.searchsorted(keys, self.keys)] = self.values
        traces = torch.zeros(len(keys), device=n.device)
        traces[torch.searchsorted(keys, self.keys)] = self.arrival_trace
        arrivals = torch.zeros(len(keys), device=n.device)
        arrivals.index_add_(0, torch.searchsorted(keys, arrival_keys), torch.ones(len(edges), device=n.device))
        traces.add_(arrivals)
        environments, edge_ids = keys.div(n.e, rounding_mode='floor'), keys.remainder(n.e)
        post = n.post[edge_ids]
        behavioral = n.pathways[edge_ids] == 2
        # Causal arrival followed by a spike potentiates; post-before-pre depresses.
        pairing = (activity.spikes[environments, post] * traces
                   - arrivals * self.post_trace[environments, post])
        values.add_(torch.where(behavioral, pairing, arrivals * n.signs[edge_ids]))
        alive = (values.abs() > cfg.prune_epsilon) | (traces > cfg.prune_epsilon)
        self.keys, self.values = keys[alive], values[alive]
        self.arrival_trace = traces[alive]
        environments, edge_ids, post, behavioral = (x[alive] for x in (environments, edge_ids, post, behavioral))
        # Prediction issued on the preceding tick is tested against this tick's local
        # feedforward current. Recurrent current cannot confirm its own prediction.
        # Current is normalized by the cell's baseline firing threshold, not a learned head.
        delta = observed[environments, post] - self.expected[environments, post]
        proposal = torch.where(behavioral,
            cfg.eta_reward * reward.detach().clamp(-1, 1)[environments] * self.values,
            torch.zeros_like(self.values) if causal else cfg.eta_prediction * delta * self.values)
        self._accumulate(edge_ids, proposal)
        self.expected.copy_(cfg.encode(activity.predicted, n.config.threshold))
        self.post_trace.add_(activity.spikes)
        self.rates.lerp_(activity.spikes.float() / dt, 1 - math.exp(-dt / cfg.tau_homeostasis))
        overload = (self.rates.mean(0)[n.post] - cfg.maximum_rate).clamp(min=0)
        self.homeostatic_exponent.add_(overload, alpha=cfg.homeostasis_rate * dt)

    @torch.no_grad()
    def reward(self, reward):
        n = self.network
        env, edges = self.keys.div(n.e, rounding_mode='floor'), self.keys.remainder(n.e)
        mask = n.pathways[edges] == 2
        proposals = self.config.eta_reward * reward.detach().clamp(-1, 1)[env[mask]] * self.values[mask]
        self._accumulate(edges[mask], proposals)

    def _accumulate(self, edges, proposals):
        if self.network.batch == 1:
            # Sparse keys are unique (environment, edge) pairs. With B=1 there
            # are no duplicate edge proposals to sort or reduce.
            self.proposals[edges] += proposals
        else:
            self.proposals.add_(_sum_sorted(edges, proposals, self.network.e) / self.network.batch)

    @torch.no_grad()
    def synchronize(self):
        n, cfg = self.network, self.config
        n.magnitudes.add_(self.proposals).clamp_(0, cfg.maximum_weight)
        n.magnitudes.mul_(torch.exp(-self.homeostatic_exponent))
        self.proposals.zero_()
        self.homeostatic_exponent.zero_()

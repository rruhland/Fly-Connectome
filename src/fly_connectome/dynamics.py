"""Sparse tensor adaptive LIF reference, shared by CPU and CUDA. No autograd."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class NeuronConfig:
    dt: float = .001
    tau_membrane: float = .02
    tau_current: float = .005
    threshold: float = 1.
    refractory_steps: int = 2
    tau_adaptation: float = 1.
    adaptation_jump: float = .05

    def __post_init__(self):
        if min(self.dt, self.tau_membrane, self.tau_current, self.tau_adaptation, self.threshold) <= 0:
            raise ValueError("positive neuron timescales and threshold required")
        if self.refractory_steps < 0 or self.adaptation_jump < 0:
            raise ValueError("refractory/adaptation must be nonnegative")


@dataclass(frozen=True)
class Activity:
    spikes: torch.Tensor
    observed: torch.Tensor
    predicted: torch.Tensor
    arrival_environments: torch.Tensor
    arrival_edges: torch.Tensor


class Network:
    @torch.no_grad()
    def __init__(self, graph, delays, pathways, *, batch=1, config=NeuronConfig(), device='cpu'):
        self.graph, self.config, self.batch, self.device = graph, config, batch, device
        self.n, self.e = len(graph.body_ids), len(graph.pre)
        if batch < 1 or len(delays) != self.e or len(pathways) != self.e:
            raise ValueError("one fixed delay/pathway per edge and positive batch required")
        if any(type(d) is not int or d < 1 for d in delays):
            raise ValueError("delays must be positive integer simulation steps")
        if not set(pathways) <= {'feedforward', 'predictive', 'behavioral'}:
            raise ValueError("unknown anatomical pathway")
        self.delays = torch.tensor(delays, dtype=torch.long, device=device)
        self.pathways = torch.tensor([{'feedforward': 0, 'predictive': 1, 'behavioral': 2}[x] for x in pathways], device=device)
        self.pre = torch.tensor(graph.pre.copy(), device=device)
        self.post = torch.tensor(graph.post.copy(), device=device)
        self.signs = torch.tensor(graph.signs[graph.pre].copy(), dtype=torch.float32, device=device)
        self.magnitudes = torch.tensor(abs(graph.initial_weights()), dtype=torch.float32, device=device)
        self.degree = torch.bincount(self.pre, minlength=self.n)
        self.starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), self.degree.cumsum(0)))
        self.history_length = max(delays, default=1) + 1
        self.history = torch.zeros(self.history_length, batch, self.n, dtype=torch.bool, device=device)
        self.voltage = torch.zeros(batch, self.n, device=device)
        self.feedforward_current = torch.zeros_like(self.voltage)
        self.predictive_current = torch.zeros_like(self.voltage)
        self.behavioral_current = torch.zeros_like(self.voltage)
        self.adaptation = torch.zeros_like(self.voltage)
        self.refractory = torch.zeros(batch, self.n, dtype=torch.long, device=device)
        self.silenced = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.step_index = 0

    def _arrivals(self):
        slots, environments, neurons = self.history.nonzero(as_tuple=True)
        degrees = self.degree[neurons]
        owner = torch.repeat_interleave(torch.arange(len(neurons), device=self.device), degrees)
        offsets = torch.cumsum(degrees, 0) - degrees
        edges = (self.starts[neurons[owner]] + torch.arange(len(owner), device=self.device)
                 - offsets[owner])
        ages = (self.step_index - slots[owner]).remainder(self.history_length)
        keep = ages == self.delays[edges]
        return environments[owner[keep]], edges[keep]

    @torch.no_grad()
    def step(self, sensory_current):
        if sensory_current.shape != (self.batch, self.n):
            raise ValueError("sensory current shape does not match network")
        cfg = self.config
        env, edges = self._arrivals()
        targets = env * self.n + self.post[edges]
        weights = self.magnitudes[edges] * self.signs[edges]
        leak = math.exp(-cfg.dt / cfg.tau_current)
        # Only three fixed pathway classes; no loop over neurons, edges or spikes.
        self.feedforward_current.mul_(leak)
        self.predictive_current.mul_(leak)
        self.behavioral_current.mul_(leak)
        ff, pred, behavior = self.pathways[edges] == 0, self.pathways[edges] == 1, self.pathways[edges] == 2
        self.feedforward_current.view(-1).index_add_(0, targets[ff], weights[ff])
        self.predictive_current.view(-1).index_add_(0, targets[pred], weights[pred])
        self.behavioral_current.view(-1).index_add_(0, targets[behavior], weights[behavior])
        observed = self.feedforward_current + sensory_current.detach()
        predicted = self.predictive_current.clone()
        current = observed + predicted + self.behavioral_current
        self.adaptation.mul_(math.exp(-cfg.dt / cfg.tau_adaptation))
        eligible = self.refractory == 0
        self.refractory.sub_(1).clamp_(min=0)
        decay = math.exp(-cfg.dt / cfg.tau_membrane)
        self.voltage.mul_(decay).add_(current, alpha=1 - decay)
        self.voltage.masked_fill_(~eligible | self.silenced[None, :], 0.)
        spikes = eligible & ~self.silenced[None, :] & (self.voltage >= cfg.threshold + self.adaptation)
        self.voltage.masked_fill_(spikes, 0.)
        self.refractory.masked_fill_(spikes, cfg.refractory_steps)
        self.adaptation.add_(spikes * cfg.adaptation_jump)
        self.history[self.step_index % self.history_length].copy_(spikes)
        self.step_index += 1
        return Activity(spikes, observed, predicted, env, edges)

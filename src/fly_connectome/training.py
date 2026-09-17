"""Headless orchestration, exact-resume checkpoints, and measured graph expansion."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile

import torch

from .dynamics import Network, NeuronConfig
from .graph import Graph
from .plasticity import Plasticity, LearningConfig
from .pong import Pong, Physics, Curriculum
from .sensor import EventCamera, Retina


@dataclass(frozen=True)
class RunConfig:
    neural_steps: int = 8
    sync_steps: int = 1
    sensory_gain: float = 30.
    tau_motor_rate: float = .05
    motor_reference_hz: float = 100.
    stage: str = 'M1B'

    def __post_init__(self):
        if self.neural_steps < 1 or self.sync_steps < 1 or min(self.sensory_gain, self.tau_motor_rate, self.motor_reference_hz) <= 0:
            raise ValueError("positive run timing and gains required")
        if self.stage not in ('M1A', 'M1B'):
            raise ValueError("stage must be M1A or M1B")


def _tensors(obj):
    return {k: v.detach().cpu().clone() for k, v in vars(obj).items() if isinstance(v, torch.Tensor)}


class Trainer:
    def __init__(self, graph, delays, pathways, retina, motor_up, motor_down, seeds, *,
                 config=RunConfig(), physics=Physics(), neurons=None, learning=LearningConfig(),
                 curriculum=Curriculum(), manifest=None, device='cpu', evaluation=False):
        if not motor_up or not motor_down or set(motor_up) & set(motor_down):
            raise ValueError("disjoint nonempty opponent motor populations required")
        if not set(motor_up + motor_down) <= set(graph.body_ids.tolist()):
            raise ValueError("motor bodies must be in measured roster")
        if retina.spec['cell_types'] and len(retina.spec['cell_types']) != len(graph.body_ids):
            raise ValueError("retina roster must match network")
        self.config, self.curriculum, self.manifest = config, curriculum, manifest or {}
        self.device, self.evaluation = device, evaluation
        self.seeds, self.motor_up, self.motor_down = list(seeds), motor_up, motor_down
        neurons = neurons or NeuronConfig(dt=physics.dt / config.neural_steps)
        self.network = Network(graph, delays, pathways, batch=len(seeds), config=neurons, device=device)
        self.learning_config = learning
        self.plasticity = None if evaluation else Plasticity(self.network, learning)
        self.environment = Pong(seeds, physics, device)
        self.retina = retina
        self.camera = EventCamera(len(seeds), retina.spec['height'], retina.spec['width'], device)
        index = {int(body): i for i, body in enumerate(graph.body_ids)}
        self.up_indices = torch.tensor([index[b] for b in motor_up], device=device)
        self.down_indices = torch.tensor([index[b] for b in motor_down], device=device)
        self.motor_rates = torch.zeros(len(seeds), len(graph.body_ids), device=device)
        self.step_index = 0
        self.statistics = torch.zeros(8, dtype=torch.float64, device=device)
        self.previous_observed = torch.zeros_like(self.network.voltage)
        self.previous_predicted = torch.zeros_like(self.network.voltage)

    @property
    def metrics(self):
        names = ('points', 'misses', 'hits', 'reward', 'prediction_squared_error',
                 'persistence_squared_error', 'prediction_samples', 'spikes')
        return dict(zip(names, self.statistics.cpu().tolist()))

    @torch.no_grad()
    def step(self, control='learned', human_drive=None):
        cfg, env, net = self.config, self.environment, self.network
        frame = env.render(self.retina.spec['height'], self.retina.spec['width'])
        events = self.camera.observe(frame)
        injection = self.retina.project(events) * cfg.sensory_gain
        # Observation boundary: only camera-derived currents enter the network.
        for tick in range(cfg.neural_steps):
            activity = net.step(injection if tick == 0 else torch.zeros_like(injection))
            observed = (activity.observed / net.config.threshold).clamp(0, 1)
            self.statistics[4] += ((observed - self.previous_predicted) ** 2).sum()
            self.statistics[5] += ((observed - self.previous_observed) ** 2).sum()
            self.statistics[6] += observed.numel()
            self.statistics[7] += activity.spikes.sum()
            self.previous_observed.copy_(observed)
            self.previous_predicted.copy_((activity.predicted / net.config.threshold).clamp(0, 1))
            self.motor_rates.lerp_(activity.spikes.float() / net.config.dt,
                                   1 - math.exp(-net.config.dt / cfg.tau_motor_rate))
            if self.plasticity is not None:
                self.plasticity.observe(activity, torch.zeros(env.batch, device=self.device))
        drive = (self.motor_rates[:, self.up_indices].mean(1) - self.motor_rates[:, self.down_indices].mean(1)) / cfg.motor_reference_hz
        if cfg.stage == 'M1A' or control == 'scripted':
            drive = -(env.ball[:, 1] - env.body.position) * 10
        elif control == 'random':
            # Independent deterministic benchmark drive from the environment PRNG.
            env.rng = (env.rng * 48271 + 1).remainder(2147483647)
            drive = env.rng.float() / 2147483647 * 2 - 1
        elif control == 'human':
            if human_drive is None:
                raise ValueError("human control requires explicit continuous drive")
            drive = human_drive
        elif control != 'learned':
            raise ValueError("unknown control mode")
        outcome = env.step(drive)
        reward = (self.curriculum.score_scale * outcome.scores if self.evaluation
                  else self.curriculum.reward(self.step_index, outcome.scores, outcome.hits))
        if self.plasticity is not None and cfg.stage == 'M1B':
            self.plasticity.reward(reward)
        self.step_index += 1
        if self.plasticity is not None and self.step_index % cfg.sync_steps == 0:
            self.plasticity.synchronize()
        self.statistics[0] += (outcome.scores > 0).sum()
        self.statistics[1] += (outcome.scores < 0).sum()
        self.statistics[2] += outcome.hits.sum()
        self.statistics[3] += reward.sum()
        return activity, events, outcome

    def run(self, steps, control='learned'):
        for _ in range(steps):
            self.step(control)
        return dict(self.metrics)

    def reset_environment(self):
        self.environment.reset(torch.ones(self.environment.batch, device=self.device, dtype=torch.bool))

    def save(self, path):
        if self.evaluation:
            raise ValueError("evaluation checkpoints are read-only")
        graph = self.network.graph
        metadata = dict(graph={k: getattr(graph, k).tolist() for k in ('body_ids', 'pre', 'post', 'contacts', 'signs')},
                        gain=graph.gain, delays=self.network.delays.tolist(),
                        pathways=[('feedforward', 'predictive', 'behavioral')[i] for i in self.network.pathways.tolist()],
                        retina=self.retina.spec, motor_up=self.motor_up, motor_down=self.motor_down,
                        seeds=self.seeds, config=asdict(self.config), physics=asdict(self.environment.config),
                        neurons=asdict(self.network.config), learning=asdict(self.learning_config),
                        curriculum=asdict(self.curriculum), manifest=self.manifest)
        state = dict(network=_tensors(self.network), plasticity=_tensors(self.plasticity),
                     environment=_tensors(self.environment), body=_tensors(self.environment.body),
                     camera=_tensors(self.camera), trainer=_tensors(self),
                     neural_step=self.network.step_index, step=self.step_index, metrics=self.metrics)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
        os.close(fd)
        try:
            torch.save(dict(schema_version=1, metadata=metadata, state=state), temp)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def load_checkpoint(path, *, device='cpu', evaluation=False):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['schema_version'] != 1:
        raise ValueError("unsupported checkpoint schema")
    m, s = payload['metadata'], payload['state']
    graph = Graph(**m['graph'], gain=m['gain'])
    if m['manifest'].get('graph_sha256', graph.identity()) != graph.identity():
        raise ValueError("checkpoint graph/manifest mismatch")
    trainer = Trainer(graph, m['delays'], m['pathways'], Retina(**m['retina'], device=device),
                      m['motor_up'], m['motor_down'], m['seeds'], config=RunConfig(**m['config']),
                      physics=Physics(**m['physics']), neurons=NeuronConfig(**m['neurons']),
                      learning=LearningConfig(**m['learning']), curriculum=Curriculum(**m['curriculum']),
                      manifest=m['manifest'], device=device, evaluation=evaluation)
    for obj, name in ((trainer.network, 'network'), (trainer.plasticity, 'plasticity'),
                      (trainer.environment, 'environment'), (trainer.environment.body, 'body'),
                      (trainer.camera, 'camera'), (trainer, 'trainer')):
        if obj is not None:
            for key, value in s[name].items():
                setattr(obj, key, value.to(device).clone())
    trainer.network.step_index, trainer.step_index = s['neural_step'], s['step']
    return trainer


@torch.no_grad()
def warm_start(source, target):
    old = {(int(source.graph.body_ids[i]), int(source.graph.body_ids[j])): e
           for e, (i, j) in enumerate(zip(source.graph.pre, source.graph.post))}
    for edge, (pre, post) in enumerate(zip(target.graph.pre, target.graph.post)):
        match = old.get((int(target.graph.body_ids[pre]), int(target.graph.body_ids[post])))
        if match is not None:
            if source.signs[match].item() != target.signs[edge].item():
                raise ValueError("warm expansion cannot change a learned edge's sign")
            target.magnitudes[edge] = source.magnitudes[match].to(target.device)

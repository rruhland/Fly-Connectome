"""Headless orchestration, exact-resume checkpoints, and measured graph expansion."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
import time

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
    warmup_steps: int = 0
    metrics_mode: str = 'full'

    def __post_init__(self):
        if self.neural_steps < 1 or self.sync_steps < 1 or min(self.sensory_gain, self.tau_motor_rate, self.motor_reference_hz) <= 0:
            raise ValueError("positive run timing and gains required")
        if self.stage not in ('M1A', 'M1B'):
            raise ValueError("stage must be M1A or M1B")
        if self.warmup_steps < 0:
            raise ValueError('warmup steps must be nonnegative')
        if self.metrics_mode not in ('full','events'):
            raise ValueError('unknown metrics mode')


def _tensors(obj):
    return {k: v.detach().cpu().clone() for k, v in vars(obj).items() if isinstance(v, torch.Tensor)}


class Trainer:
    def __init__(self, graph, delays, pathways, retina, motor_up, motor_down, seeds, *,
                 config=RunConfig(), physics=Physics(), neurons=None, learning=LearningConfig(),
                 curriculum=Curriculum(), manifest=None, device='cpu', evaluation=False):
        if (config.stage == 'M1B' and (not motor_up or not motor_down)) or set(motor_up) & set(motor_down):
            raise ValueError("disjoint nonempty opponent motor populations required")
        if not set(motor_up + motor_down) <= set(graph.body_ids.tolist()):
            raise ValueError("motor bodies must be in measured roster")
        if retina.spec['cell_types'] and len(retina.spec['cell_types']) != len(graph.body_ids):
            raise ValueError("retina roster must match network")
        self.config, self.curriculum, self.manifest = config, curriculum, manifest or {}
        self.device, self.evaluation = device, evaluation
        self.seeds, self.motor_up, self.motor_down = list(seeds), motor_up, motor_down
        neurons = neurons or NeuronConfig(dt=physics.dt / config.neural_steps)
        self.network = Network(graph, delays, pathways, batch=len(seeds), config=neurons, device=device,
                               cell_types=retina.spec['cell_types'])
        self.learning_config = learning
        self.plasticity = None if evaluation else Plasticity(self.network, learning,
            sensory_mask=retina.injected, sensory_gain=config.sensory_gain)
        self.environment = Pong(seeds, physics, device)
        self.retina = retina
        self.camera = EventCamera(len(seeds), retina.spec['height'], retina.spec['width'], device)
        index = {int(body): i for i, body in enumerate(graph.body_ids)}
        self.up_indices = torch.tensor([index[b] for b in motor_up], dtype=torch.long, device=device)
        self.down_indices = torch.tensor([index[b] for b in motor_down], dtype=torch.long, device=device)
        self.motor_rates = torch.zeros(len(seeds), len(graph.body_ids), device=device)
        self.control_rng = (self.environment.rng + 123457).remainder(2147483647)
        self.step_index = 0
        self.statistics = torch.zeros(11, dtype=torch.float64, device=device)
        self.previous_observed = torch.zeros_like(self.network.voltage)
        self.previous_predicted = torch.zeros_like(self.network.voltage)
        self.previous_events = torch.zeros_like(self.network.voltage)
        self.learning_statistics = torch.zeros(3, dtype=torch.float64, device=device)
        self.previous_learning_observed = torch.zeros_like(self.network.voltage)
        if evaluation:
            self.event_zero_error = torch.zeros((), dtype=torch.float64, device=device)
            self.visible_spikes = torch.zeros_like(self.network.voltage, dtype=torch.bool)
            self.evaluation_spike_counts = torch.zeros_like(self.network.voltage, dtype=torch.int64)
            self.visible_events = None

    @property
    def metrics(self):
        names = ('points', 'misses', 'hits', 'reward', 'prediction_squared_error',
                 'persistence_squared_error', 'prediction_samples', 'spikes',
                 'event_prediction_squared_error', 'event_persistence_squared_error', 'event_samples')
        result = dict(zip(names, self.statistics.cpu().tolist()))
        values = (self.learning_statistics.cpu().tolist() if self.learning_config.visual_target == 'input-arrivals-v1'
                  else [result[name] for name in ('prediction_squared_error','persistence_squared_error','prediction_samples')])
        result.update(zip(('learning_squared_error','learning_persistence_squared_error','learning_samples'), values))
        return result

    @torch.no_grad()
    def step(self, control='learned', human_drive=None):
        cfg, env, net = self.config, self.environment, self.network
        frame = env.render(self.retina.spec['height'], self.retina.spec['width'])
        events = self.camera.observe(frame)
        if self.evaluation:
            self.visible_events = events
            self.visible_spikes.zero_()
        injection = self.retina.project(events) * cfg.sensory_gain
        sensed = injection[:, self.retina.injected] / cfg.sensory_gain
        if self.evaluation:
            self.event_zero_error += sensed.square().sum()
        self.statistics[8] += ((sensed - self.previous_predicted[:, self.retina.injected]) ** 2).sum()
        self.statistics[9] += ((sensed - self.previous_events[:, self.retina.injected]) ** 2).sum()
        self.statistics[10] += sensed.numel()
        self.previous_events.copy_(injection / cfg.sensory_gain)
        # Observation boundary: only camera-derived currents enter the network.
        for tick in range(cfg.neural_steps):
            activity = net.step(injection if tick == 0 else torch.zeros_like(injection),
                                capture_increments=self.learning_config.visual_target == 'input-arrivals-v1')
            if self.evaluation:
                self.visible_spikes |= activity.spikes
                self.evaluation_spike_counts += activity.spikes
            self.statistics[7] += activity.spikes.sum()
            full_metrics = cfg.metrics_mode == 'full' or self.evaluation
            if full_metrics or tick == cfg.neural_steps-1:
                observed = self.learning_config.encode(activity.observed, net.config.threshold)
                if full_metrics:
                    self.statistics[4] += ((observed - self.previous_predicted) ** 2).sum()
                    self.statistics[5] += ((observed - self.previous_observed) ** 2).sum()
                    self.statistics[6] += observed.numel()
                if self.learning_config.visual_target == 'input-arrivals-v1':
                    target = self.learning_config.observation(activity, net.config.threshold, self.retina.injected, cfg.sensory_gain)
                    if full_metrics:
                        self.learning_statistics[0] += (target-self.previous_predicted).square().sum()
                        self.learning_statistics[1] += (target-self.previous_learning_observed).square().sum()
                        self.learning_statistics[2] += target.numel()
                    self.previous_learning_observed.copy_(target)
                self.previous_observed.copy_(observed)
                self.previous_predicted.copy_(self.learning_config.encode(activity.predicted, net.config.threshold))
            self.motor_rates.lerp_(activity.spikes.float() / net.config.dt,
                                   1 - math.exp(-net.config.dt / cfg.tau_motor_rate))
            if self.plasticity is not None:
                self.plasticity.observe(activity, torch.zeros(env.batch, device=self.device))
        drive = ((self.motor_rates[:, self.up_indices].mean(1) - self.motor_rates[:, self.down_indices].mean(1)) / cfg.motor_reference_hz
                 if self.motor_up and self.motor_down else torch.zeros(env.batch, device=self.device))
        if (cfg.stage == 'M1A' and not self.evaluation) or control == 'scripted':
            drive = -(env.ball[:, 1] - env.body.position) * 10
        elif control == 'random':
            # Independent deterministic benchmark drive from the environment PRNG.
            self.control_rng = (self.control_rng * 48271 + 1).remainder(2147483647)
            drive = self.control_rng.float() / 2147483647 * 2 - 1
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

    def warmup(self):
        """Fixed no-event neural settling, independent of Pong and plasticity."""
        for _ in range(self.config.warmup_steps):
            self.network.step(torch.zeros_like(self.network.voltage))

    def reset_environment(self):
        self.environment.reset(torch.ones(self.environment.batch, device=self.device, dtype=torch.bool))

    def snapshot(self):
        if hasattr(self,'_materialize'):
            self._materialize()
        env = self.environment
        alpha = 0. if self.evaluation else self.curriculum.alpha(self.step_index)
        phase = 'score-only' if alpha == 0 else ('bootstrap' if self.step_index < self.curriculum.bootstrap else 'fade')
        snapshot = dict(step=self.step_index, sampled_environment=0, sampled=True,
                    metrics_mode='full' if self.evaluation else self.config.metrics_mode,
                    ball=env.ball[0].cpu().tolist(), player_y=env.body.position[0].item(),
                    opponent_y=env.opponent[0].item(), activation=env.body.activation[0].item(),
                    velocity=env.body.velocity[0].item(), phase=phase, alpha=alpha,
                    device=str(self.device), environments=env.batch, neurons=self.network.n,
                    edges=self.network.e, graph_threshold=self.manifest.get('threshold'),
                    physics=asdict(env.config), metrics=self.metrics, evaluation=self.evaluation,
                    rally_seconds=env.rally_steps[0].item() * env.config.dt,
                    tensor_memory_bytes=sum(v.numel() * v.element_size() for obj in
                        (self.network, self.plasticity, self) if obj is not None
                        for v in vars(obj).values() if isinstance(v, torch.Tensor)))
        if self.evaluation:
            spikes = self.visible_spikes[0].nonzero().flatten()
            snapshot['spiking_neurons'] = spikes[:2048].cpu().tolist()
            snapshot['spikes_sampled'] = len(spikes) > 2048
            events = self.visible_events
            keep = events.environments == 0 if events is not None else None
            snapshot['events'] = dict(width=self.retina.spec['width'], height=self.retina.spec['height'],
                pixels=events.pixels[keep].cpu().tolist() if events is not None else [],
                on=events.on[keep].cpu().tolist() if events is not None else [])
            types = self.retina.spec['cell_types']
            snapshot['population_rates_hz'] = {}
            for label in ('L1', 'L2', 'L3', 'L4', 'T4', 'T5', 'LPi', 'LC10', 'DNa02'):
                indices = [i for i,t in enumerate(types) if t == label or
                           (label in ('T4', 'T5', 'LPi', 'LC10') and t.startswith(label))]
                if indices:
                    snapshot['population_rates_hz'][label] = self.motor_rates[0, indices].mean().item()
        return snapshot

    def save(self, path):
        if self.evaluation:
            raise ValueError("evaluation checkpoints are read-only")
        if hasattr(self,'_materialize'):
            self._materialize()
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
            with open(temp, 'wb') as stream:
                torch.save(dict(schema_version=2, metadata=metadata, state=state), stream)
            # Windows readers/indexers can temporarily deny atomic replacement.
            # Keep the old checkpoint intact and bound the wait to 30 seconds.
            for attempt in range(31):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError as error:
                    if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 30:
                        raise
                    time.sleep(1)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def load_checkpoint(path, *, device='cpu', evaluation=False, seeds=None, warmup=True):
    if seeds is not None and not evaluation:
        raise ValueError("new seeds are only allowed for fresh frozen evaluation")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['schema_version'] not in (1, 2):
        raise ValueError("unsupported checkpoint schema")
    m, s = payload['metadata'], payload['state']
    graph = Graph(**m['graph'], gain=m['gain'])
    if m['manifest'].get('graph_sha256', graph.identity()) != graph.identity():
        raise ValueError("checkpoint graph/manifest mismatch")
    trainer = Trainer(graph, m['delays'], m['pathways'], Retina(**m['retina'], device=device),
                      m['motor_up'], m['motor_down'], seeds if seeds is not None else m['seeds'], config=RunConfig(**m['config']),
                      physics=Physics(**m['physics']), neurons=NeuronConfig(**m['neurons']),
                      learning=LearningConfig(**m['learning']), curriculum=Curriculum(**m['curriculum']),
                      manifest=m['manifest'], device=device, evaluation=evaluation)
    trainer.training_seeds = m['seeds']
    trainer.training_step = s['step']
    if seeds is not None:
        trainer.network.magnitudes.copy_(s['network']['magnitudes'])
        if warmup:
            trainer.warmup()
        return trainer
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

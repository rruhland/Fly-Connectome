"""Offline frozen visual probes. Imported by evaluation tools, never training."""
from dataclasses import asdict, dataclass
import math
import torch

from .data import checksum
from .training import load_checkpoint


@dataclass(frozen=True)
class Probe:
    kind: str = 'flash'
    steps: int = 100
    start: int = 10
    duration: int = 10
    x: float = 16.
    y: float = 16.
    radius: float = 2.
    direction: float = 0.
    speed: float = 1.
    polarity: str = 'on'

    def __post_init__(self):
        if self.kind not in ('flash', 'moving_edge', 'target', 'neighbor_sequence'):
            raise ValueError("unknown probe kind")
        if self.steps <= 0 or self.duration <= 0 or self.radius <= 0 or self.start < 0:
            raise ValueError("invalid probe timing or size")
        if self.polarity not in ('on', 'off'):
            raise ValueError("probe requires explicit ON/OFF polarity")

    def frame(self, step, height, width, device='cpu'):
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        theta = math.radians(self.direction)
        elapsed = step - self.start
        cx = self.x + elapsed * self.speed * math.cos(theta)
        cy = self.y + elapsed * self.speed * math.sin(theta)
        if self.kind == 'moving_edge':
            foreground = (x - cx) * math.cos(theta) + (y - cy) * math.sin(theta) <= 0
        else:
            if self.kind == 'flash':
                cx, cy = self.x, self.y
            elif self.kind == 'neighbor_sequence':
                cx = self.x + (elapsed // self.duration) * self.speed
                cy = self.y
            foreground = (x - cx) ** 2 + (y - cy) ** 2 <= self.radius ** 2
            if self.kind == 'flash' and not self.start <= step < self.start + self.duration:
                foreground = torch.zeros_like(foreground)
        return (foreground if self.polarity == 'on' else ~foreground).unsqueeze(0)


@torch.no_grad()
def run_probe(checkpoint, probe, *, seed=1000, silenced_types=(), device='cpu'):
    trainer = load_checkpoint(checkpoint, evaluation=True, seeds=[seed], device=device)
    types = trainer.retina.spec['cell_types']
    unknown = set(silenced_types) - set(types)
    if unknown:
        raise ValueError(f"unknown silenced populations: {sorted(unknown)}")
    trainer.network.silenced.copy_(torch.tensor([t in silenced_types for t in types], device=device))
    height, width = trainer.retina.spec['height'], trainer.retina.spec['width']
    # OFF probes begin against a bright reference, then expose controlled transitions.
    if probe.polarity == 'off':
        trainer.camera.previous.fill_(True)
    spikes, inputs = [], []
    for tick in range(probe.steps):
        frame = probe.frame(tick, height, width, device)
        events = trainer.camera.observe(frame)
        current = trainer.retina.project(events) * trainer.config.sensory_gain
        activity = trainer.network.step(current)
        spikes.append(activity.spikes[0].cpu())
        inputs.append(current[0].cpu())
    spikes = torch.stack(spikes)
    populations = {}
    for population in sorted(set(types)):
        ids = [i for i, t in enumerate(types) if t == population]
        response = spikes[:, ids].float().mean(1)
        active = torch.nonzero(response).flatten()
        populations[population] = dict(response=response, first_spike_tick=int(active[0]) if len(active) else None,
                                       last_spike_tick=int(active[-1]) if len(active) else None,
                                       sparsity=float(1 - response.mean()))
    return dict(schema_version=1, checkpoint_sha256=checksum(checkpoint), manifest=trainer.manifest,
                stimulus=asdict(probe), seed=seed, backend=device, silenced_types=list(silenced_types),
                dt=trainer.network.config.dt, spikes=spikes, sensory_current=torch.stack(inputs),
                populations=populations)

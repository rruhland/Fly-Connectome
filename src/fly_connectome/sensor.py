"""Binary ON/OFF event camera and fixed, nonlearned retinotopic projection."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Events:
    environments: torch.Tensor
    pixels: torch.Tensor
    on: torch.Tensor
    offsets: torch.Tensor


class EventCamera:
    def __init__(self, batch, height, width, device='cpu'):
        self.previous = torch.zeros(batch, height, width, dtype=torch.bool, device=device)

    @torch.no_grad()
    def observe(self, frame):
        if frame.dtype != torch.bool or frame.shape != self.previous.shape:
            raise ValueError("camera requires binary boolean frames of its configured shape")
        changed = (frame != self.previous).flatten(1)
        env, pixel = changed.nonzero(as_tuple=True)
        polarity = frame.flatten(1)[env, pixel]
        counts = changed.sum(1)
        offsets = torch.cat((torch.zeros(1, device=frame.device, dtype=torch.long), counts.cumsum(0)))
        self.previous.copy_(frame)
        return Events(env, pixel, polarity, offsets)


class Retina:
    def __init__(self, height, width, hex_columns, neuron_columns, cell_types, injection, device='cpu'):
        self.spec = dict(height=height, width=width, hex_columns=hex_columns,
                         neuron_columns=neuron_columns, cell_types=cell_types, injection=injection)
        if not set(injection) <= {'L1', 'L2', 'L3'}:
            raise ValueError("direct sensor injection is restricted to L1-L3")
        if not set(injection.values()) <= {'on', 'off', 'contrast'}:
            raise ValueError("each injected cell type requires an explicit polarity or signed contrast")
        self.device = device
        columns = torch.tensor(hex_columns, dtype=torch.float64, device=device)
        if columns.ndim != 2 or columns.shape[1] != 2 or not len(columns) or not torch.isfinite(columns).all():
            raise ValueError("finite axial hex coordinates required")
        xy = torch.stack((columns[:, 0] + .5 * columns[:, 1], columns[:, 1] * math.sqrt(3) / 2), 1)
        extent = xy.amax(0) - xy.amin(0)
        # Fit the complete render field into the measured extent with one scale.
        scale = torch.min(torch.where(extent > 0, extent / torch.tensor([width, height], device=device), torch.inf))
        if not torch.isfinite(scale):
            scale = torch.tensor(1., device=device)
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        pixels = torch.stack((x.flatten() + .5 - width / 2, y.flatten() + .5 - height / 2), 1)
        pixels = pixels * scale + (xy.amax(0) + xy.amin(0)) / 2
        self.pixel_bins = torch.cdist(pixels.to(torch.float64), xy).argmin(1)
        self.n_columns = len(columns)
        self.neuron_columns = torch.tensor(neuron_columns, dtype=torch.long, device=device)
        if len(neuron_columns) != len(cell_types):
            raise ValueError("one column index required per neuron")
        self.injected = torch.tensor([t in injection for t in cell_types], dtype=torch.bool, device=device)
        selected = self.neuron_columns[self.injected]
        if torch.any(selected < 0) or torch.any(selected >= self.n_columns):
            raise ValueError("injected neurons require a measured retinotopic column")
        self.polarity = torch.tensor([injection.get(t) == 'on' for t in cell_types], dtype=torch.long, device=device)
        self.contrast = torch.tensor([injection.get(t) == 'contrast' for t in cell_types], dtype=torch.bool, device=device)
        # Fixed retinotopy: resolve roster masks once, not on every camera frame.
        self._injected_indices = self.injected.nonzero().flatten()
        self._injected_columns = self.neuron_columns[self._injected_indices]
        self._injected_polarity = self.polarity[self._injected_indices]
        self._contrast_indices = self.contrast.nonzero().flatten()
        self._contrast_columns = self.neuron_columns[self._contrast_indices]

    @torch.no_grad()
    def project(self, events):
        batch = len(events.offsets) - 1
        bins = torch.zeros(batch * 2 * self.n_columns, dtype=torch.float32, device=self.device)
        indices = (events.environments * 2 + events.on.long()) * self.n_columns + self.pixel_bins[events.pixels]
        bins.index_fill_(0, indices, 1.)
        bins = bins.view(batch, 2, self.n_columns)
        result = torch.zeros(batch, len(self.neuron_columns), device=self.device)
        result[:, self._injected_indices] = bins[:, self._injected_polarity, self._injected_columns]
        result[:, self._contrast_indices] = (bins[:, 0, self._contrast_columns]
                                            - bins[:, 1, self._contrast_columns])
        return result

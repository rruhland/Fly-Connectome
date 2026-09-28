"""Generic visual transport; structured beliefs remain the source of truth."""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F


def _copy_records(records):
    copied, memo = [], {}
    memo[id(records)] = copied
    for record in records:
        item = memo.get(id(record))
        if item is None:
            item = {}
            memo[id(record)] = item
            for key, value in record.items():
                if type(value) is torch.Tensor and not value.requires_grad and value._base is None:
                    clone = memo.get(id(value))
                    if clone is None:
                        clone = value.clone()
                        memo[id(value)] = clone
                    item[copy.deepcopy(key, memo)] = clone
                else:
                    item[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        copied.append(item)
    return copied


def _ordered_components(forecasts):
    rows = []
    for forecast in forecasts:
        weights = forecast['mixture_weights'].detach().to(device='cpu', dtype=torch.float64)
        if not len(weights):
            continue
        centers = forecast['normalized_centers'].detach().to(device='cpu', dtype=torch.float64)
        horizon = torch.full((len(weights), 1), forecast['horizon_samples']-1,
                             dtype=torch.float64)
        rows.append(torch.cat((horizon, centers, weights[:, None]), 1))
    if not rows:
        return torch.empty(0, 4, dtype=torch.get_default_dtype())
    values = torch.cat(rows).numpy()
    order = np.lexsort((values[:, 3], values[:, 2], values[:, 1], values[:, 0]))
    return torch.from_numpy(values[order]).to(torch.get_default_dtype())


class VisualStateEncoder:
    """Encode at a fixed declared sensor cadence, without learning or game labels.

    Velocities use normalized image units per second. Sparse bins contain sums,
    not probabilities of occupancy. Call reset_state with the visual scene reset.
    """

    def __init__(self, height, width, sample_period_seconds):
        if height <= 0 or width <= 0 or not math.isfinite(sample_period_seconds) or sample_period_seconds <= 0:
            raise ValueError('positive dimensions and finite positive sample period required')
        self.height, self.width = height, width
        self.period = sample_period_seconds
        self.scale = max(height, width)
        self.center = torch.tensor([(height-1)/2, (width-1)/2])
        self.half_extent = torch.tensor([height, width])/(2*self.scale)
        sizes = dict(current=7*16*16, future=8*16*16, future_outside=8,
                     relations=5*9*9, context=2*8*8, metadata=8)
        self.layout, offset = {}, 0
        for name, size in sizes.items():
            self.layout[name] = (offset, offset+size)
            offset += size
        self.dimension = offset
        self.reset_state()

    def reset_state(self):
        self.sample = None
        self.history = {}
        self.context = torch.zeros(8, 8)
        self.context_sample = None

    def _normalize(self, position):
        return (position-self.center)/self.scale

    @staticmethod
    def _bin(position, side):
        return ((position+.5)*side).floor().long().clamp(0, side-1)

    @torch.no_grad()
    def encode(self, state, frame):
        sample = state['sample']
        if self.sample is not None and sample <= self.sample:
            raise ValueError('samples must increase; call reset_state for a new scene')
        if frame is not None and (frame.device.type != 'cpu' or frame.shape != (self.height, self.width)
                                  or not torch.isfinite(frame).all()):
            raise ValueError('frame must be a finite CPU image matching encoder dimensions')
        if any(not 1 <= f['horizon_samples'] <= 8 for f in state['forecasts']):
            raise ValueError('transport v1 supports horizons 1 through 8 samples')
        elapsed = self.period if self.sample is None else (sample-self.sample)*self.period
        self.sample = sample
        entities, forecasts = _copy_records(state['entities']), _copy_records(state['forecasts'])
        active = {e['id'] for e in entities}
        self.history = {k: v for k, v in self.history.items() if k in active}
        for entity in entities:
            identity, position = entity['id'], entity['position']
            previous = self.history.get(identity)
            velocity, velocity_sample = (None, None) if previous is None else previous[2:]
            if entity['observed']:
                if previous is not None:
                    velocity = (position-previous[0])/(self.scale*(sample-previous[1])*self.period)
                    velocity_sample = sample
                self.history[identity] = (position.clone(), sample, velocity, velocity_sample)
            entity.update(normalized_position=self._normalize(position),
                          observed_velocity=torch.zeros(2) if velocity is None else velocity.clone(),
                          velocity_known=velocity is not None,
                          velocity_age_seconds=None if velocity_sample is None else (sample-velocity_sample)*self.period,
                          observation_age_seconds=entity['observation_age']*self.period)
        for forecast in forecasts:
            forecast.update(normalized_centers=self._normalize(forecast['mixture_centers']),
                            normalized_position=self._normalize(forecast['position']),
                            normalized_interval_90=self._normalize(forecast['marginal_interval_90']),
                            normalized_component_sigma=forecast['component_sigma_pixels']/self.scale,
                            horizon_seconds=forecast['horizon_samples']*self.period,
                            evidence_age_seconds=forecast['evidence_age']*self.period)
        if frame is not None:
            contrast = frame.float()-frame.float().mean()
            y, x = self.scale-self.height, self.scale-self.width
            square = F.pad(contrast, (x//2, x-x//2, y//2, y-y//2))
            self.context = F.adaptive_avg_pool2d(square[None, None], (8, 8))[0, 0]
            self.context_sample = sample
        context_age = None if self.context_sample is None else (sample-self.context_sample)*self.period
        vector = torch.zeros(self.dimension)
        current = vector[slice(*self.layout['current'])].view(7, 16, 16)
        future = vector[slice(*self.layout['future'])].view(8, 16, 16)
        outside = vector[slice(*self.layout['future_outside'])]
        relations = vector[slice(*self.layout['relations'])].view(5, 9, 9)
        # Canonical content order makes floating-point reductions ID/order invariant.
        ordered = sorted(entities, key=lambda e: (*e['normalized_position'].tolist(),
                         e['observed'], *e['observed_velocity'].tolist(), e['velocity_known']))
        off_field = pairs = 0
        for entity in ordered:
            p, v = entity['normalized_position'], entity['observed_velocity']
            if bool((p.abs() > self.half_extent).any()):
                off_field += 1
                continue
            y, x = self._bin(p, 16)
            current[0 if entity['observed'] else 1, y, x] += 1
            if entity['velocity_known']:
                current[2:6, y, x] += torch.stack((v[0].clamp(min=0), (-v[0]).clamp(min=0),
                                                  v[1].clamp(min=0), (-v[1]).clamp(min=0)))
                current[6, y, x] += 1
            neighbors = sorted((other for other in ordered if other is not entity),
                               key=lambda other: float((other['normalized_position']-p).square().sum()))[:4]
            for other in neighbors:
                delta = other['normalized_position']-p
                y, x = self._bin(delta/2, 9)
                relations[0, y, x] += 1
                pairs += 1
                if entity['velocity_known'] and other['velocity_known']:
                    dv = other['observed_velocity']-v
                    relations[1:, y, x] += torch.stack((dv[0].clamp(min=0), (-dv[0]).clamp(min=0),
                                                       dv[1].clamp(min=0), (-dv[1]).clamp(min=0)))
        component = _ordered_components(forecasts)
        if len(component):
            horizon, position, weight = component[:, 0].long(), component[:, 1:3], component[:, 3]
            off_image = (position.abs() > self.half_extent).any(1)
            outside.index_add_(0, horizon[off_image], weight[off_image])
            bins = self._bin(position[~off_image], 16)
            indices = horizon[~off_image]*256+bins[:, 0]*16+bins[:, 1]
            future.flatten().index_add_(0, indices, weight[~off_image])
        vector[slice(*self.layout['context'])] = torch.stack((self.context.clamp(min=0),
                                                            (-self.context).clamp(min=0))).flatten()
        vector[slice(*self.layout['metadata'])] = torch.tensor([
            len(entities), sum(e['observed'] for e in entities), elapsed, frame is not None,
            self.context_sample is not None, 0 if context_age is None else context_age, off_field, pairs])
        indices = vector.nonzero().flatten()
        return dict(schema='visual-transport-v1', sample=sample, sample_period_seconds=self.period,
                    image_shape=(self.height, self.width), entities=entities, forecasts=forecasts,
                    coarse_context=self.context.clone(), context_available=frame is not None,
                    context_known=self.context_sample is not None, context_age_seconds=context_age,
                    layout=self.layout.copy(), dimension=self.dimension, indices=indices,
                    values=vector[indices], relation_pairs=pairs)

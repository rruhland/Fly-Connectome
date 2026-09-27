"""Opt-in generic visual evidence, learned context and spatial future beliefs."""

import torch

from local_metric_association import LocalMetricAssociation
from spatial_belief import SpatialBelief, mixture_quantile
from statistical_visual_state import StatisticalVisualCandidate


class ProbabilisticVisualState(StatisticalVisualCandidate):
    def __init__(self, **kwargs):
        kwargs.setdefault('dynamics', {h: SpatialBelief(horizon=h) for h in (1, 4, 8)})
        super().__init__(**kwargs)

    @torch.no_grad()
    def step(self, events, visible_frame, *, learn=False):
        result = super().step(events, visible_frame, learn=learn)
        for forecast in result['forecasts']:
            if forecast['evidence_age']:
                continue
            history = torch.stack(list(self.histories[forecast['id']]))
            model = self.dynamics[forecast['horizon_samples']]
            centers, weights = model.distribution(history[1:]-history[:-1])
            low, high = model.calibration.bounds()
            forecast.update(mixture_centers=centers+history[-1], mixture_weights=weights,
                component_sigma_pixels=1., point_semantics='mixture_mean',
                marginal_interval_90=torch.stack((mixture_quantile(centers, weights, low),
                                                  mixture_quantile(centers, weights, high)))+history[-1],
                interval_calibration_samples=len(model.calibration.ranks))
        return result

    def save(self, path):
        if type(self.state.memory) is not LocalMetricAssociation or any(
                type(m) is not SpatialBelief for m in self.dynamics.values()):
            raise ValueError('unsupported probabilistic candidate component')
        torch.save(dict(version=1, architecture='ProbabilisticVisualState',
            sensor='grayscale-every-sample-plus-events', height=self.observer.height,
            width=self.observer.width, observer=dict(weights=self.observer.weights, bias=self.observer.bias),
            memory={name: getattr(self.state.memory, name) for name in ('keys', 'values', 'metric')},
            dynamics={h: dict(capacity=m.capacity, keys=m.keys, values=m.values, seen=m.seen,
                generator=m.generator.get_state(), ranks=list(m.calibration.ranks)) for h, m in self.dynamics.items()},
            calibration={h: dict(outcomes=list(c.outcomes), residuals=list(c.residuals))
                         for h, c in self.calibration.items()}), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=True)
        if data['version'] != 1 or data['architecture'] != 'ProbabilisticVisualState':
            raise ValueError('unsupported probabilistic candidate checkpoint')
        dynamics = {h: SpatialBelief(capacity=d['capacity'], horizon=h) for h, d in data['dynamics'].items()}
        model = cls(height=data['height'], width=data['width'], dynamics=dynamics,
                    memory=LocalMetricAssociation(dimensions=data['memory']['keys'].shape[1]))
        for name, value in data['observer'].items():
            setattr(model.observer, name, value)
        for name, value in data['memory'].items():
            setattr(model.state.memory, name, value)
        for h, values in data['dynamics'].items():
            for name in ('keys', 'values', 'seen'):
                setattr(dynamics[h], name, values[name])
            dynamics[h].generator.set_state(values['generator'])
            dynamics[h].calibration.ranks.extend(values['ranks'])
        for h, values in data['calibration'].items():
            model.calibration[h].outcomes.extend(values['outcomes'])
            model.calibration[h].residuals.extend(values['residuals'])
        return model

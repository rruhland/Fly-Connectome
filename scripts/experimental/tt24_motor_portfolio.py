"""Select from reward-learned motor readouts using current optical context."""

import numpy as np

from .tt24_motor import optical_sightings


def context(observation):
    ball, blade = optical_sightings(observation.get('rgb'))
    if ball is None or blade is None:
        return None
    return np.r_[(ball.center - blade.center) / max(.04, blade.half_span),
                 np.log(ball.radius / blade.half_span), observation['joints']]


class Portfolio:
    """Predict relative return locally in retinal context using practice outcomes."""

    def __init__(self, trials, repeats, *, count=4):
        if repeats < 2 or len(trials) < 2 * repeats:
            raise ValueError('need two complete matched practice blocks')
        groups = [trials[i:i + repeats] for i in range(0, len(trials), repeats)]
        examples = [(row['seed'], row['initial_context']) for row in groups[0] if row['initial_context'] is not None]
        if len(examples) < 2:
            raise ValueError('need at least two visible practice contexts')
        launches, features = zip(*examples, strict=True)
        self.features = np.asarray(features)
        axes = self.features.shape[1] - 3
        self.width = np.maximum(self.features.std(axis=0), np.r_[.14, .14, .12, np.full(axes, .045)])
        if len(launches) != len(set(launches)):
            raise ValueError('use a block of distinct matched starts for each readout')
        weights, values = [], []
        for group in groups:
            by_seed = {row['seed']: row['score'] for row in group}
            if all(seed in by_seed for seed in launches):
                weights.append(group[0]['weights'])
                values.append([by_seed[seed] for seed in launches])
        if len(values) < 2:
            raise ValueError('need at least two readouts tested on the same visible practice contexts')
        initial, *rest = np.asarray(values)
        delta = np.asarray(rest) - initial
        top = np.argsort(-delta.mean(axis=1), kind='stable')[:count]
        self.weights = np.asarray(weights)[np.r_[0, 1 + top]]
        self.values = np.r_[np.zeros_like(delta[:1]), delta[top]]
        self.global_value = self.values.mean(axis=1)

    def choose(self, observation):
        observed = context(observation)
        if observed is None:
            return self.weights[self.global_value.argmax()]
        distance = np.mean(((self.features - observed) / self.width) ** 2, axis=1)
        similarity = np.exp(-.5 * distance)
        values = (self.global_value + self.values @ similarity) / (1. + similarity.sum())
        return self.weights[values.argmax()]

"""Read-only retinotopic T4/T5 activity from the recovered measured graph."""

import torch

from fly_connectome.sensor import Events
from run_decisive_representation_audit import make_graph, SUBTYPES


class MeasuredMotionAdapter:
    def __init__(self):
        self.graph = make_graph()
        graph = self.graph
        columns = torch.as_tensor(graph['columns'])
        stages = torch.tensor([SUBTYPES.index(t) if t in SUBTYPES else -1
                               for t in graph['types']])
        valid = (stages >= 0) & (columns >= 0) & (columns < graph['retina'].n_columns)
        self.nodes = valid.nonzero().flatten()
        self.bins = stages[valid]*graph['retina'].n_columns+columns[valid]
        self.denominator = torch.bincount(self.bins, minlength=8*graph['retina'].n_columns).clamp(min=1)
        self.reset_state()

    def reset_state(self):
        for name, value in self.graph['settled'].items():
            getattr(self.graph['net'], name).copy_(value)
        self.graph['net'].step_index = self.graph['settled_tick']

    @torch.no_grad()
    def step(self, dense_events):
        polarity, y, x = (dense_events > .5).nonzero(as_tuple=True)
        sparse = Events(torch.zeros_like(y), y*64+x, polarity.bool(), torch.tensor([0, len(y)]))
        graph = self.graph
        current = graph['retina'].project(sparse)*graph['metadata']['config']['sensory_gain']
        counts = torch.zeros(8*graph['retina'].n_columns)
        for tick in range(8):
            activity = graph['net'].step(current if tick == 0 else graph['zero'])
            counts.index_add_(0, self.bins, activity.spikes[0, self.nodes].float())
        rates = (counts/(8*self.denominator)).reshape(8, -1)
        return rates[:, graph['retina'].pixel_bins].reshape(8, 32, 64)

    def verify_frozen(self):
        if not torch.equal(self.graph['net'].magnitudes, self.graph['original']):
            raise AssertionError('measured connectome weights changed')

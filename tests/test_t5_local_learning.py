import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from t5_local_learning import T5FramePrediction
from fly_connectome.dynamics import Network, NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig


def test_t5_frame_rule_captures_only_local_target_edges():
    graph = Graph(np.array([1, 2, 3]), np.array([0, 0]),
                  np.array([1, 2]), np.array([2, 3]),
                  np.ones(3, dtype=np.int64), .1)
    net = Network(graph, [1, 1], ['predictive', 'predictive'],
        config=NeuronConfig(tau_sensory=.02))
    cfg = LearningConfig(visual_eligibility='forecast-causal-v1')
    rule = T5FramePrediction(net, cfg,
        target_mask=torch.tensor([False, True, False]))
    rule.keys = torch.tensor([0, 1])
    rule.values = torch.tensor([.5, .5])
    rule._capture_forecast()
    assert torch.equal(rule.forecast[0], torch.tensor([0]))
    rule._accumulate(torch.tensor([0, 1]), torch.tensor([1., 1.]))
    assert torch.equal(rule.proposals, torch.zeros(2))

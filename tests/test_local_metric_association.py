import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_metric_association import LocalMetricAssociation


def test_local_context_can_retain_opposing_relations_without_task_labels():
    memory = LocalMetricAssociation(dimensions=2)
    for _ in range(20):
        for a in (-1., 1.):
            for b in (-1., 1.):
                memory.observe(torch.tensor([a, b]), torch.tensor([a*b, 0.]))
    for a in (-1., 1.):
        for b in (-1., 1.):
            prediction = memory.predict(torch.tensor([.95*a, 1.05*b]))
            assert prediction[0]*a*b > .9

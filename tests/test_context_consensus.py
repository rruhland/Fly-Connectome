import torch
import pytest

from fly_connectome.vision import load_legacy_default, ProbabilisticVisualState
from fly_connectome.vision.memory import ConsensusLocalMetricAssociation, LocalMetricAssociation


def test_conflicting_corrections_abstain_without_disabling_learning():
    memory = ConsensusLocalMetricAssociation(dimensions=2)
    key = torch.zeros(2)
    for outcome in ((3., -1.), (1., 2.5), (-1.5, 2.5), (-1.5, 2.)):
        memory.observe(key, torch.tensor(outcome))
    assert len(memory.keys) == 4
    assert torch.equal(memory.predict(key), torch.zeros(2))


def test_agreed_axis_keeps_learned_magnitude_and_other_axis_abstains():
    memory = ConsensusLocalMetricAssociation(dimensions=2)
    key = torch.zeros(2)
    for outcome in ((1., -1.), (2., 1.), (3., -1.), (4., 1.)):
        memory.observe(key, torch.tensor(outcome))
    assert torch.equal(memory.predict(key), torch.tensor([2.5, 0.]))


def test_local_learning_acquires_opposing_contexts_from_empty_memory():
    memory = ConsensusLocalMetricAssociation(dimensions=2)
    for _ in range(20):
        for a in (-1., 1.):
            for b in (-1., 1.):
                memory.observe(torch.tensor([a, b]), torch.tensor([a*b, 0.]))
    for a in (-1., 1.):
        for b in (-1., 1.):
            assert memory.predict(torch.tensor([.95*a, 1.05*b]))[0]*a*b > .9


def test_upgrade_versions_consensus_and_legacy_v2_still_loads(tmp_path):
    legacy = load_legacy_default()
    assert type(legacy.state.memory) is LocalMetricAssociation
    model = legacy.upgrade_temporal_context()
    assert type(model.state.memory) is ConsensusLocalMetricAssociation
    path = tmp_path/'vision.pt'
    model.save(path)
    loaded = ProbabilisticVisualState.load(path)
    assert type(loaded.state.memory) is ConsensusLocalMetricAssociation
    assert torch.equal(loaded.state.memory.values, model.state.memory.values)
    data = torch.load(path, weights_only=True)
    assert data.pop('context_rule') == 'local-sign-consensus'
    assert data['version'] == 3
    # Original version-2 candidates did not have the new context rule.
    data['version'] = 2
    torch.save(data, path)
    old = ProbabilisticVisualState.load(path)
    assert type(old.state.memory) is LocalMetricAssociation
    data['version'] = 3
    data['context_rule'] = 'unknown'
    torch.save(data, path)
    with pytest.raises(ValueError, match='context rule'):
        ProbabilisticVisualState.load(path)

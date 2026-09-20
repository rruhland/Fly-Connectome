from pathlib import Path
import importlib.util

import pytest
import torch

from test_training import trainer

spec = importlib.util.spec_from_file_location('prediction_audit', Path(__file__).parents[1]/'scripts/audit_prediction.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prediction_audit_separates_true_events_from_false_alarms():
    target = torch.tensor([-1., 0., 1., 0.])
    prediction = torch.tensor([-.5, .2, .5, -.2])
    result = module.summary(module.sums(target, prediction, torch.zeros(4)))
    assert result['model_mse'] == pytest.approx(.145)
    assert result['zero_mse'] == .5
    assert result['active_mse'] == .25
    assert result['quiet_mse'] == pytest.approx(.04)
    assert result['samples'] == 4 and result['active_samples'] == 2
    assert result['zero_mse'] - result['model_mse'] == pytest.approx(
        2*result['target_prediction_cross_mean']-result['prediction_power'])


def test_lagged_audit_labels_delayed_response_without_calling_it_prediction():
    targets = torch.tensor([0.,1.,0.,-1.,0.,0.])
    predictions = torch.tensor([0.,0.,1.,0.,-1.,0.])
    result = module.lagged_scores(targets,predictions,(-1,0,1))
    assert result['1']['model_mse'] == 0
    assert result['0']['model_mse'] > result['0']['zero_mse']
    assert result['-1']['model_mse'] > 0


def test_frozen_audit_agrees_with_evaluation_and_preserves_checkpoint(tmp_path):
    from fly_connectome.evaluation import evaluate
    path = tmp_path/'fixture.pt'
    trainer().save(path)
    original = path.read_bytes()
    report = module.audit(str(path), 10, [100],{'all_sensory':trainer().retina.injected})
    ordinary = evaluate(path, [100], 10, 'scripted')
    assert report['events']['L2']['model_mse'] == pytest.approx(ordinary['metrics']['sensory_event_prediction_mse'])
    assert report['events']['L2']['zero_mse'] == pytest.approx(ordinary['metrics']['sensory_event_zero_mse'])
    assert report['population_spikes']['L2'] == ordinary['population_spikes']['L2']
    assert report['event_groups']['all_sensory']==report['events']['L2']
    assert path.read_bytes() == original


def test_filtered_current_target_is_not_a_camera_event_target():
    from fly_connectome.graph import Graph
    from fly_connectome.dynamics import Network, NeuronConfig
    from fly_connectome.plasticity import LearningConfig
    graph = Graph.from_contacts([10,20], [10], [20], [1], [1,1], .1)
    net = Network(graph, [1], ['predictive'], config=NeuronConfig(dt=1/960, tau_sensory=.02))
    cfg = LearningConfig(prediction_encoding='signed-current-v1')
    net.step(torch.tensor([[-30.,0.]]))
    for _ in range(8):
        activity = net.step(torch.zeros(1,2))
    assert activity.observed[0,0].item() == pytest.approx(-19.7772, abs=.0001)
    filtered_target = cfg.encode(activity.observed, 1.)[0,0]
    assert filtered_target == -1
    # A perfect forecast for this approved local target is wrong about the next
    # camera frame's absence of an event, even though there is no timing leak.
    assert (filtered_target-0).square() == 1

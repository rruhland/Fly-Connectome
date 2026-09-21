import importlib.util
from pathlib import Path
from dataclasses import replace
import pytest
import torch


@pytest.fixture
def audit_module(monkeypatch):
    folder=Path(__file__).parents[1]/'scripts'
    path=folder/'audit_supported_forecasts.py'
    assert path.exists(), 'supported forecast audit not implemented'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('supported_forecasts',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supported_event_errors_and_quiet_false_alarms(audit_module):
    target=torch.tensor([-1.,1.,0.]).view(3,1,1)
    prediction=torch.tensor([-.5,-.2,.3]).view(3,1,1)
    result=audit_module.group_scores(target,prediction,torch.tensor([False]),torch.tensor([True]))
    assert result['on_supported']['samples']==1
    assert result['on_supported']['model_mse']==.25
    assert result['off_unsupported']['model_mse']==pytest.approx(1.44)
    assert result['off_unsupported']['correct_sign_fraction']==0.
    assert result['quiet']['model_mse']==pytest.approx(.09)
    assert result['off_supported'] is None
    assert result['all']['model_mse']==pytest.approx((.25+1.44+.09)/3)


def test_lag_positive_is_post_event_not_prediction(audit_module):
    target=torch.tensor([0.,1.,0.,-1.,0.]).view(5,1,1)
    timeline=torch.tensor([0.,0.,1.,0.,-1.,0.]).view(6,1,1)
    result=audit_module.lag_scores(target,timeline,1,torch.tensor([True]),torch.tensor([True]),(-1,0,1))
    assert result['scores']['1']['all']['model_mse']==0.
    assert result['scores']['0']['all']['model_mse']>0.
    assert len({r['all']['samples'] for r in result['scores'].values()})==1


def test_frozen_frame_forecast_matches_ordinary_evaluation(audit_module,tmp_path):
    from test_training import trainer
    from fly_connectome.sensor import Retina
    from fly_connectome.evaluation import evaluate
    from fly_connectome.dynamics import NeuronConfig
    from fly_connectome.graph import Graph
    from fly_connectome.training import Trainer
    base=trainer()
    retina=Retina(**dict(base.retina.spec,injection={'L2':'contrast'},
                        cell_types=['L2','auto','DNa02','DNa02']))
    graph=Graph.from_contacts([10,20,30,40],[10,20,20,20],[20,10,30,40],
                              [5]*4,[1,-1,1,1],.1)
    model=Trainer(graph,[1,2,1,1],['feedforward','predictive','behavioral','behavioral'],
        retina,[30],[40],[1],config=base.config,
        neurons=NeuronConfig(dt=base.network.config.dt,class_parameters={'auto':{'rest_current':1.5}}),
        learning=replace(base.learning_config,prediction_encoding='signed-current-v1',
            visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1'))
    path=tmp_path/'fixture.pt'
    model.save(path)
    before=path.read_bytes()
    report=audit_module.audit(path,20,[1101])
    ordinary=evaluate(path,[1101],20,'scripted')
    assert report['strict']['all']['model_mse']==pytest.approx(ordinary['metrics']['sensory_event_prediction_mse'])
    assert report['strict']['all']['zero_mse']==pytest.approx(ordinary['metrics']['sensory_event_zero_mse'])
    assert report['strict']['all']['prediction_power']>0
    assert report['neural_ticks']==40
    assert path.read_bytes()==before

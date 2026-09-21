import importlib.util
from pathlib import Path
import torch
from fly_connectome.graph import Graph
from fly_connectome.dynamics import Network


def test_support_uses_measured_predictive_signs_not_feedforward_or_current_weights():
    path=Path(__file__).parents[1]/'scripts/audit_predictive_support.py'
    assert path.exists(), 'predictive support diagnostic not implemented'
    spec=importlib.util.spec_from_file_location('support',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph=Graph.from_contacts([10,20,30,40],[10,10,20,20],[30,40,30,40],
                              [1]*4,[1,-1,1,1],.5)
    net=Network(graph,[1]*4,['predictive','feedforward','predictive','predictive'])
    net.magnitudes.zero_()
    positive,negative=module.predictive_support(net)
    assert positive.tolist()==[False,False,True,False]
    assert negative.tolist()==[False,False,True,True]
    targets=torch.tensor([[0.,0.,-1.,1.],[0.,0.,1.,-1.]])
    assert module.unreachable_events(targets,positive,negative).tolist()==[[False,False,False,True],[False,False,False,False]]


def test_bounds_distinguish_active_events_from_all_sensory_samples():
    path=Path(__file__).parents[1]/'scripts/audit_predictive_support.py'
    spec=importlib.util.spec_from_file_location('support',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module,'error_bounds'), 'audit must distinguish the two MSE denominators'
    assert module.error_bounds(1,2,8)==dict(event_conditioned_mse_lower_bound=.5,
                                         all_sample_mse_lower_bound=.125)

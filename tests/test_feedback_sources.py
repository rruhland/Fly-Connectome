import importlib.util
from pathlib import Path
import torch


def test_feedback_source_audit_preserves_state_and_attributes_activity():
    path=Path(__file__).parents[1]/'scripts/audit_feedback_sources.py'
    spec=importlib.util.spec_from_file_location('feedback_sources',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from fly_connectome.graph import Graph
    from fly_connectome.dynamics import Network, NeuronConfig
    graph=Graph.from_contacts([10,20,30],[10,20],[30,30],[5,5],[1,-1,1],.1)
    cfg=NeuronConfig(dt=.01,tau_membrane=.01,adaptation_jump=0,
                    class_parameters={'active':{'rest_current':2.}})
    types=['active','silent','L3']
    a=Network(graph,[1,1],['predictive']*2,config=cfg,cell_types=types)
    b=Network(graph,[1,1],['predictive']*2,config=cfg,cell_types=types)
    audit=module.SourceAudit(a,types,'L3')
    for tick in range(20):
        activity=a.step(torch.zeros(1,3))
        audit.observe(activity,measure=tick>=2)
        b.step(torch.zeros(1,3))
    report=audit.report()
    assert report['classes']['active']['active_feedback_edges']==1
    assert report['classes']['silent']['active_feedback_edges']==0
    assert report['classes']['active']['spikes']>0
    assert report['classes']['silent']['spikes']==0
    assert report['measured_ticks']==18
    assert report['maximum_reconstruction_error']<1e-6
    for name,value in vars(a).items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,getattr(b,name)),name


def test_c2_restoration_only_changes_existing_incoming_predictive_weights():
    path=Path(__file__).parents[1]/'scripts/audit_feedback_sources.py'
    spec=importlib.util.spec_from_file_location('feedback_sources',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from fly_connectome.graph import Graph
    from fly_connectome.dynamics import Network
    graph=Graph.from_contacts([10,20,30],[10,20,30],[20,30,20],[5]*3,[1,-1,1],.1)
    paths=['predictive','predictive','feedforward']
    reference=Network(graph,[1]*3,paths)
    trained=Network(graph,[1]*3,paths)
    trained.magnitudes.zero_()
    assert module.restore_c2_inputs(trained,reference,['source','C2','L3'])==1
    assert trained.magnitudes.tolist()==[.5,0.,0.]
    assert torch.equal(trained.signs,reference.signs)
    assert reference.magnitudes.tolist()==[.5,.5,.5]

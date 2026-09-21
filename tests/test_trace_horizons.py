import importlib.util
from pathlib import Path
import torch


def module(monkeypatch):
    folder=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('trace_horizons',folder/'audit_trace_horizons.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_trace_moments_keep_seeds_separate_and_require_repeated_events(monkeypatch):
    m=module(monkeypatch)
    moments=m.TraceMoments(2,2)
    for y in (0.,1.,0.,-1.)*3:
        target=torch.tensor([[y,y],[-y,-y]])
        trace=torch.tensor([[y,1.],[y,1.]])
        moments.add(target,trace)
    result=moments.metrics()
    assert result['events'].tolist()==[[6.,6.],[6.,6.]]
    torch.testing.assert_close(result['correlation'],torch.tensor([[1.,0.],[-1.,0.]],dtype=torch.float64))
    assert result['covariance'][0,0]==.5


def test_ring_lookup_is_exact_causal_lead(monkeypatch):
    m=module(monkeypatch)
    ring=torch.arange(7,dtype=torch.float32).view(7,1,1)
    # Entries are keyed by completed tick count; current target tick is 5.
    assert m.earlier_trace(ring,5,1).item()==5
    assert m.earlier_trace(ring,5,3).item()==3
    assert m.earlier_trace(ring,12,3).item()==3


def test_next_event_counts_only_distinct_events_actually_referenced(monkeypatch):
    m=module(monkeypatch)
    targets=torch.tensor([0.,0.,0.,1.,-1.,1.,-1.,1.]).view(8,1,1)
    _,waits=m.next_events(targets,5)
    # Decisions 0..2 all reference event 3; events 4..7 are never first-next.
    assert m.referenced_event_counts(waits,0,3).item()==1


def test_frozen_trace_horizons_verify_replay_and_current_reconstruction(monkeypatch,tmp_path):
    m=module(monkeypatch)
    from test_training import trainer
    from fly_connectome.training import Trainer
    from fly_connectome.graph import Graph
    from fly_connectome.sensor import Retina
    from fly_connectome.dynamics import NeuronConfig
    from dataclasses import replace
    base=trainer()
    graph=Graph.from_contacts([10,20,30,40],[20,20,20],[10,30,40],[5]*3,[1]*4,.1)
    retina=Retina(**dict(base.retina.spec,injection={'L2':'contrast'}))
    model=Trainer(graph,[1]*3,['predictive','behavioral','behavioral'],retina,[30],[40],[1],
        config=replace(base.config,warmup_steps=20),
        neurons=NeuronConfig(dt=base.network.config.dt,class_parameters={'visual':{'rest_current':1.5}}),
        learning=replace(base.learning_config,prediction_encoding='signed-current-v1',
                         visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1'))
    path=tmp_path/'source.pt'
    model.save(path)
    before=path.read_bytes()
    result=m.audit(path,40)
    assert result['samples_per_seed']==16
    assert result['maximum_reconstruction_error']<1e-6
    assert set(result['trials'])=={'one_tick','30','50','100','next_event'}
    assert path.read_bytes()==before

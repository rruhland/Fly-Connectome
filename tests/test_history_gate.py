import importlib.util
from pathlib import Path
import torch


def module(monkeypatch):
    folder=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('history_gate',folder/'audit_history_gate.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_history_is_causal_and_separates_never_seen(monkeypatch):
    m=module(monkeypatch)
    target=torch.tensor([0.,1.,0.,0.,-1.,0.]).view(6,1,1)
    strata,last,ages=m.local_history(target,2)
    assert last.flatten().tolist()==[0.,1.,1.,1.,-1.,-1.]
    assert ages.flatten().tolist()==[2,0,1,2,0,1]
    assert strata[0]!=strata[3]
    changed=target.clone(); changed[4]=1
    assert torch.equal(m.local_history(changed,2)[0][:4],strata[:4])


def test_shuffle_preserves_strata_and_reports_constant_groups(monkeypatch):
    m=module(monkeypatch)
    labels=torch.tensor([0.,1.,-1.,1.,1.])
    strata=torch.tensor([0,0,0,1,1])
    shuffled,groups,fraction=m.conditional_shuffle(labels,strata,421)
    assert sorted(shuffled[:3].tolist())==[-1.,0.,1.]
    assert shuffled[3:].tolist()==[1.,1.]
    assert fraction==3/5
    assert groups[1]['permutable'] is False


def test_gate_needs_two_pairs_each_passing_both_seeds(monkeypatch):
    m=module(monkeypatch)
    assert m.gate_status([['pass','pass'],['pass','fail'],['fail','pass']])=='fail'
    assert m.gate_status([['pass','pass'],['pass','pass'],['inconclusive','inconclusive']])=='pass'
    assert m.gate_status([['pass','pass'],['inconclusive','pass'],['fail','fail']])=='inconclusive'


def test_gate_collector_is_frozen_and_reconstructs_current(monkeypatch,tmp_path):
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
    report=m.audit(path,40,pairs=((20,10),))
    assert report['common_frames']==[12,28]
    assert report['maximum_reconstruction_error']<1e-6
    assert len(report['pairs'][0]['results'])==2
    assert path.read_bytes()==before

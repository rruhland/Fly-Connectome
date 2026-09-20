import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('local_signal_audit',Path(__file__).parents[1]/'scripts/audit_local_signals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_local_signal_moments_distinguish_anticipation_from_tonic_activity():
    target = torch.tensor([[0.,0.],[1.,1.],[0.,0.],[-1.,-1.]])
    trace = torch.tensor([[0.,1.],[1.,1.],[0.,1.],[-1.,1.]])
    moments = module.SignalMoments(2)
    for y,e in zip(target,trace):
        moments.add(y[None,:],e[None,:],torch.zeros(1,2))
    result = moments.metrics()
    torch.testing.assert_close(result['covariance'],torch.tensor([.5,0.],dtype=torch.float64))
    torch.testing.assert_close(result['correlation'],torch.tensor([1.,0.],dtype=torch.float64))
    torch.testing.assert_close(result['event_trace_energy_fraction'],torch.tensor([1.,.5],dtype=torch.float64))
    torch.testing.assert_close(result['mean_local_update'],torch.tensor([.5,0.],dtype=torch.float64))


def test_local_signal_audit_is_frozen_and_reconstructs_measured_current(tmp_path):
    from test_training import trainer
    from fly_connectome.plasticity import LearningConfig
    from fly_connectome.training import Trainer
    from fly_connectome.graph import Graph
    from fly_connectome.dynamics import NeuronConfig
    from dataclasses import replace
    base=trainer()
    graph=Graph.from_contacts([10,20,30,40],[20,20,20],[10,30,40],[5]*3,[1]*4,.1)
    model=Trainer(graph,[1]*3,['predictive','behavioral','behavioral'],
        base.retina,[30],[40],[1],config=replace(base.config,warmup_steps=100),
        neurons=NeuronConfig(dt=base.network.config.dt,class_parameters={'visual':{'rest_current':1.5}}),
        learning=LearningConfig(prediction_encoding='signed-current-v1',visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1'))
    path=tmp_path/'model.pt'; model.save(path); before=path.read_bytes()
    report=module.audit(str(path),10,[1101])
    assert report['training_steps']==0
    assert report['measured_ticks']==18
    assert report['maximum_reconstruction_error']<1e-6
    assert report['populations']['L2']['active_edges']==1
    assert path.read_bytes()==before


def test_fast_exposure_replay_matches_actual_scripted_observations(tmp_path):
    from test_training import trainer
    from fly_connectome.training import load_checkpoint
    exposure_spec=importlib.util.spec_from_file_location('exposure',Path(__file__).parents[1]/'scripts/audit_pong_exposure.py')
    exposure_module=importlib.util.module_from_spec(exposure_spec)
    exposure_spec.loader.exec_module(exposure_module)
    path=tmp_path/'model.pt'; trainer().save(path)
    actual=load_checkpoint(path,evaluation=True,seeds=[1101])
    fast=load_checkpoint(path,evaluation=True,seeds=[1101])
    counts=torch.zeros_like(actual.network.voltage,dtype=torch.int64)
    for _ in range(20):
        _,events,_=actual.step('scripted')
        counts += actual.retina.project(events)!=0
    torch.testing.assert_close(exposure_module.exposure(fast,20),counts)
    torch.testing.assert_close(actual.environment.ball,fast.environment.ball,rtol=0,atol=0)
    torch.testing.assert_close(actual.camera.previous,fast.camera.previous,rtol=0,atol=0)

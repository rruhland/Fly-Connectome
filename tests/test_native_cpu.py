import copy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import pytest
import torch
from fly_connectome.dynamics import Network, Activity
from fly_connectome.graph import Graph
from fly_connectome.plasticity import Plasticity, LearningConfig
from fly_connectome.native_cpu import NativeCPU, build_library


@pytest.fixture(scope='session')
def native_library(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('optional native CPU checks require g++')
    output = tmp_path_factory.mktemp('native')/'kernels.dll'
    build_library(output)
    return output


@pytest.mark.parametrize('causal',[False,True])
def test_native_sparse_learning_matches_reference_each_tick(native_library,causal):
    graph=Graph.from_contacts([10,20,30,40],[10,10,20,30],[20,30,40,10],[1]*4,[-1,1,1,1],.5)
    net=Network(graph,[1]*4,['predictive','behavioral','feedforward','predictive'])
    rule=Plasticity(net,LearningConfig(prediction_encoding='signed-current-v1',
        visual_eligibility='forecast-causal-v1' if causal else 'legacy-v1',prune_epsilon=.01))
    other=copy.deepcopy(rule)
    native=NativeCPU(native_library)
    rng=torch.Generator().manual_seed(24)
    for tick in range(100):
        edges=torch.randint(0,4,(7 if tick<50 else 0,),generator=rng)
        activity=Activity(torch.rand((1,4),generator=rng)>.8,
            torch.rand((1,4),generator=rng)*2-1,torch.rand((1,4),generator=rng)*2-1,
            torch.zeros(len(edges),dtype=torch.long),edges)
        reward=torch.tensor([.7 if tick%3 else -.3])
        rule.observe(activity,reward)
        native.observe(other,activity,reward)
        for name,value in vars(rule).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,getattr(other,name),rtol=0,atol=0,msg=lambda m: f'{name} tick {tick}: {m}')
        if tick%8==7:
            rule.synchronize();other.synchronize()
            torch.testing.assert_close(rule.network.magnitudes,other.network.magnitudes,rtol=0,atol=0)

@pytest.mark.parametrize('maximum_rate',[0.,5.,100.])
def test_native_homeostasis_matches_reference_for_active_and_quiet_cells(native_library,maximum_rate):
    graph=Graph.from_contacts([10,20,30],[10,20],[20,30],[1,1],[1,1,1],.5)
    net=Network(graph,[1,1],['predictive','behavioral'])
    rule=Plasticity(net,LearningConfig(maximum_rate=maximum_rate,homeostasis_rate=.2))
    rule.rates[:]=torch.tensor([[0.,10.,3.]])
    other=copy.deepcopy(rule)
    native=NativeCPU(native_library)
    activity=Activity(torch.zeros(1,3,dtype=torch.bool),torch.zeros(1,3),torch.zeros(1,3),
                      torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))
    for _ in range(10):
        rule.observe(activity,torch.zeros(1));native.observe(other,activity,torch.zeros(1))
    torch.testing.assert_close(rule.homeostatic_exponent,other.homeostatic_exponent,rtol=0,atol=0)
    rule.synchronize();other.synchronize()
    torch.testing.assert_close(rule.network.magnitudes,other.network.magnitudes,rtol=0,atol=0)

@pytest.mark.parametrize('classes',[False,True])
def test_native_neural_step_matches_reference(native_library,classes):
    from fly_connectome.dynamics import NeuronConfig
    graph=Graph.from_contacts([10,20,30,40],[10,10,20,30],[20,30,40,10],[1]*4,[-1,1,1,1],.5)
    config=NeuronConfig(tau_sensory=.02,class_parameters={'cell':{'rest_current':1.2}} if classes else {})
    net=Network(graph,[1,2,3,1],['predictive','behavioral','feedforward','predictive'],
                config=config,cell_types=['cell']*4)
    other=copy.deepcopy(net)
    kernel=NativeCPU(native_library)
    rng=torch.Generator().manual_seed(13)
    for tick in range(100):
        current=torch.rand((1,4),generator=rng)*30-15
        a=net.step(current,capture_increments=True)
        b=kernel.step(other,current,capture_increments=True)
        for name,value in vars(net).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,getattr(other,name),rtol=0,atol=0,msg=lambda m:f'{name} tick {tick}: {m}')
        for name in vars(a):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)

@pytest.mark.parametrize('capture',[False,True])
def test_native_empty_graph_silencing_and_unfiltered_input(native_library,capture):
    from fly_connectome.dynamics import NeuronConfig
    graph=Graph.from_contacts([10],[],[],[],[1],.5)
    net=Network(graph,[],[],config=NeuronConfig(dt=.001))
    net.silenced.fill_(True)
    other=copy.deepcopy(net)
    a=net.step(torch.tensor([[100.]]),capture_increments=capture)
    b=NativeCPU(native_library).step(other,torch.tensor([[100.]]),capture_increments=capture)
    assert not b.spikes.any()
    torch.testing.assert_close(a.observed,b.observed,rtol=0,atol=0)
    torch.testing.assert_close(net.voltage,other.voltage,rtol=0,atol=0)
    assert (b.feedforward_arrivals is None)==(not capture)


def test_native_checkpoint_can_resume_on_reference_backend(native_library,tmp_path):
    from test_training import trainer
    from fly_connectome.training import load_checkpoint
    model=trainer()
    reference=copy.deepcopy(model)
    kernel=NativeCPU(native_library)
    kernel.enable(model)
    model.run(11);reference.run(11)
    path=tmp_path/'native.pt'
    model.save(path)
    resumed=load_checkpoint(path)
    resumed.run(13);reference.run(13);model.run(13)
    for obj in (resumed,model):
        for target in ('network','plasticity'):
            for name,value in vars(getattr(reference,target)).items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(value,getattr(getattr(obj,target),name),rtol=0,atol=0)


def test_cli_explicit_native_backend_keeps_checkpoint_portable(native_library,tmp_path):
    import json
    import os
    import subprocess
    import sys
    from test_training import trainer
    from fly_connectome.training import load_checkpoint
    model=trainer()
    source=tmp_path/'source.pt';output=tmp_path/'trained.pt'
    model.save(source)
    process=subprocess.run([sys.executable,'-m','fly_connectome','train',str(source),
        '--steps','4','--output',str(output),'--threads','1','--native-library',str(native_library)],
        capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    model.run(4)
    loaded=load_checkpoint(output)
    torch.testing.assert_close(model.network.magnitudes,loaded.network.magnitudes,rtol=0,atol=0)
    torch.testing.assert_close(model.network.voltage,loaded.network.voltage,rtol=0,atol=0)
    assert json.loads(process.stdout.splitlines()[-1])['backend']=='native-cpu'


def test_parallel_native_sparse_partitions_preserve_exact_order(native_library):
    import numpy as np
    size=20000
    graph=Graph.from_contacts(list(range(1,size+2)),list(range(1,size+1)),list(range(2,size+2)),
        [1]*size,[1]*(size+1),.5)
    net=Network(graph,[1]*size,['behavioral' if i%3==0 else 'predictive' for i in range(size)])
    rule=Plasticity(net,LearningConfig(visual_eligibility='forecast-causal-v1',prune_epsilon=.1))
    rule.keys=torch.arange(0,size,2)
    rule.values=torch.linspace(-.2,.2,len(rule.keys))
    rule.arrival_trace=torch.zeros(len(rule.keys))
    other=copy.deepcopy(rule)
    edges=torch.tensor([0,4998,5000,5000,9999,10000,14998,15000,19999])
    activity=Activity(torch.ones(1,size+1,dtype=torch.bool),torch.ones(1,size+1),
        torch.zeros(1,size+1),torch.zeros(len(edges),dtype=torch.long),edges)
    rule.observe(activity,torch.tensor([.7]))
    NativeCPU(native_library,threads=4).observe(other,activity,torch.tensor([.7]))
    for name,value in vars(rule).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,getattr(other,name),rtol=0,atol=0)


@pytest.mark.parametrize('target',['filtered-current-v1','input-arrivals-v1'])
@pytest.mark.parametrize('threshold',[1.,.7])
def test_native_observation_preparation_preserves_normalization(native_library,target,threshold):
    from fly_connectome.dynamics import NeuronConfig
    graph=Graph.from_contacts([10,20,30],[10,20],[20,30],[1,1],[-1,1,1],.5)
    net=Network(graph,[1,1],['predictive','predictive'],config=NeuronConfig(threshold=threshold))
    rule=Plasticity(net,LearningConfig(visual_target=target,prediction_encoding='signed-current-v1'),
        sensory_mask=torch.tensor([False,True,False]),sensory_gain=2.7)
    rule.post_trace[:]=torch.tensor([[.1,.7,1.3]])
    act=Activity(torch.zeros(1,3,dtype=torch.bool),torch.tensor([[.13,-.2,.7]]),torch.zeros(1,3),
        torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),
        torch.tensor([[.31,-.21,.713]]),torch.tensor([[0.,-.737,0.]]))
    expected=rule.config.observation(act,threshold,rule.sensory_mask,rule.sensory_gain)
    observed=NativeCPU(native_library).prepare(rule,act)
    torch.testing.assert_close(observed,expected,rtol=0,atol=0)


@pytest.mark.parametrize('field,kind',[
    ('observed','strided'),('observed','half'),('observed','short'),
    ('feedforward_arrivals','short'),('sensory_input','half'),
    ('sensory_mask','half'),('post_trace','short'),
])
def test_native_preparation_rejects_invalid_pointers_before_mutation(native_library,field,kind):
    from test_training import trainer
    model=trainer()
    rule=model.plasticity
    rule.config=replace(rule.config,visual_target='input-arrivals-v1')
    n=model.network.n
    activity=Activity(torch.zeros(1,n,dtype=torch.bool),torch.zeros(1,n),torch.zeros(1,n),
        torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.zeros(1,n),torch.zeros(1,n))
    original=getattr(rule if field in ('sensory_mask','post_trace') else activity,field)
    invalid=(torch.zeros(1,n*2)[:,::2] if kind=='strided' else
             original.half() if kind=='half' else original[...,:-1])
    if field in ('sensory_mask','post_trace'):
        setattr(rule,field,invalid)
    else:
        activity=replace(activity,**{field:invalid})
    rule.post_trace.fill_(1.)
    before=rule.post_trace.clone()
    with pytest.raises(ValueError,match='canonical'):
        NativeCPU(native_library).prepare(rule,activity)
    assert torch.equal(before,rule.post_trace)


def test_native_synchronization_preserves_bounds_and_homeostasis(native_library):
    graph=Graph.from_contacts([10,20,30],[10,20],[20,30],[1,1],[1,1,1],.5)
    for exponent in (0.,.03):
        rule=Plasticity(Network(graph,[1,1],['predictive','behavioral']),LearningConfig(maximum_weight=1.))
        rule.proposals[:]=torch.tensor([-10.,10.])
        rule.homeostatic_exponent.fill_(exponent)
        other=copy.deepcopy(rule)
        rule.synchronize()
        NativeCPU(native_library).synchronize(other)
        for name in ('proposals','homeostatic_exponent'):
            torch.testing.assert_close(getattr(rule,name),getattr(other,name),rtol=0,atol=0)
        torch.testing.assert_close(rule.network.magnitudes,other.network.magnitudes,rtol=0,atol=0)


def test_native_arrivals_match_mixed_delays_across_history_wrap(native_library):
    graph=Graph.from_contacts([10,20,30,40],[10,10,20,30],[20,30,40,10],[1]*4,[1]*4,.5)
    net=Network(graph,[1,2,3,1],['predictive']*4)
    native=NativeCPU(native_library)
    rng=torch.Generator().manual_seed(12)
    for tick in range(12):
        net.step_index=tick
        net.history.copy_(torch.rand(net.history.shape,generator=rng)>.5)
        a=net._arrivals();b=native.arrivals(net)
        for expected,actual in zip(a,b):
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)

@pytest.mark.parametrize('eta',[0.,.01,.5,1.,1.2])
def test_native_prediction_updates_preserve_subnormal_float_bits(native_library,eta):
    torch.set_flush_denormal(False)
    size=1024
    graph=Graph.from_contacts(list(range(1,size+2)),list(range(1,size+1)),list(range(2,size+2)),
        [1]*size,[1]*(size+1),.5)
    net=Network(graph,[1]*size,['predictive']*size)
    rule=Plasticity(net,LearningConfig(visual_eligibility='forecast-causal-v1',eta_prediction=eta,
                                     prediction_encoding='signed-current-v1'))
    rng=torch.Generator().manual_seed(89)
    bits=torch.randint(1,2**23,(size+1,),generator=rng,dtype=torch.int32)
    rule.expected[:]=bits.view(torch.float32)
    rule.expected[:,::2].neg_()
    rule.keys=torch.arange(size);rule.values=torch.ones(size);rule.arrival_trace=torch.ones(size)
    other=copy.deepcopy(rule)
    activity=Activity(torch.zeros(1,size+1,dtype=torch.bool),torch.zeros(1,size+1),torch.zeros(1,size+1),
                      torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))
    rule.observe(activity,torch.zeros(1));NativeCPU(native_library).observe(other,activity,torch.zeros(1))
    assert torch.equal(rule.proposals.view(torch.int32),other.proposals.view(torch.int32))

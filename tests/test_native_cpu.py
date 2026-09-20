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
from fly_connectome.native_cpu import NativeCPU


@pytest.fixture(scope='session')
def native_library(tmp_path_factory):
    if not shutil.which('g++'):
        pytest.skip('optional native CPU checks require g++')
    output = tmp_path_factory.mktemp('native')/'kernels.dll'
    subprocess.run(['g++','-O3','-fno-fast-math','-ffp-contract=off','-shared',
        '-static-libgcc','-static-libstdc++','src/fly_connectome/native_cpu.cpp','-o',str(output)],check=True)
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

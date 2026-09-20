import copy
from dataclasses import replace
import pytest
import torch
from fly_connectome.dynamics import Activity, Network
from fly_connectome.graph import Graph
from fly_connectome.plasticity import Plasticity, LearningConfig
from test_native_cpu import native_library


@pytest.mark.parametrize('epsilon',[1e-8,.2])
@pytest.mark.parametrize('aggregate',[False,True])
def test_deferred_visual_updates_match_each_materialization(native_library,epsilon,aggregate):
    from fly_connectome.deferred_cpu import DeferredCPU
    graph=Graph.from_contacts([10,20,30,40],[10,10,20,30],[20,30,40,10],[1]*4,[-1,1,1,1],.5)
    net=Network(graph,[1]*4,['predictive','behavioral','feedforward','predictive'])
    rule=Plasticity(net,LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',prune_epsilon=epsilon))
    other=copy.deepcopy(rule)
    kernel=DeferredCPU(native_library,aggregate=aggregate)
    rng=torch.Generator().manual_seed(244)
    for tick in range(120):
        edges=torch.randint(0,4,(7 if tick<50 else 0,),generator=rng)
        activity=Activity(torch.rand(1,4,generator=rng)>.7,torch.rand(1,4,generator=rng),
            torch.rand(1,4,generator=rng)*8-4,torch.zeros_like(edges),edges,
            torch.rand(1,4,generator=rng)*8-4,torch.zeros(1,4))
        reward=torch.tensor([.7 if tick%3 else -.3])
        rule.observe(activity,reward);kernel.observe(other,activity,reward)
        rule.reward(reward);other.reward(reward)
        if tick%7==6 or tick==119:
            kernel.materialize(other)
            for name,value in vars(rule).items():
                if isinstance(value,torch.Tensor):
                    approximate=aggregate and name=='proposals'
                    torch.testing.assert_close(value,getattr(other,name),rtol=1e-6 if approximate else 0,
                        atol=1e-9 if approximate else 0,msg=lambda m:f'{name}: {m}')
            behavior=rule.network.pathways==2
            torch.testing.assert_close(rule.proposals[behavior],other.proposals[behavior],rtol=0,atol=0)
            rule.synchronize();kernel.synchronize(other)
            torch.testing.assert_close(rule.network.magnitudes,other.network.magnitudes,
                rtol=1e-6 if aggregate else 0,atol=1e-9 if aggregate else 0)


def test_deferred_checkpoint_and_snapshot_materialize_pending_ticks(native_library,tmp_path):
    from test_training import trainer
    from fly_connectome.training import load_checkpoint
    from fly_connectome.deferred_cpu import DeferredCPU
    model=trainer()
    model.learning_config=replace(model.learning_config,visual_target='input-arrivals-v1',
                                 visual_eligibility='forecast-causal-v1')
    model.plasticity.config=model.learning_config
    model.config=replace(model.config,sync_steps=3)
    reference=copy.deepcopy(model)
    kernel=DeferredCPU(native_library)
    kernel.enable(model)
    model.run(2);reference.run(2)
    model.snapshot()
    torch.testing.assert_close(model.plasticity.values,reference.plasticity.values,rtol=0,atol=0)
    model.run(2);reference.run(2)
    path=tmp_path/'deferred.pt'
    model.save(path)
    resumed=load_checkpoint(path)
    assert resumed.manifest['execution_experiment']=='deferred-ticks-v1'
    resumed.run(4);reference.run(4);model.run(4)
    kernel.materialize(model.plasticity)
    for candidate in (resumed,model):
        for obj in ('network','plasticity'):
            for name,value in vars(getattr(reference,obj)).items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(value,getattr(getattr(candidate,obj),name),rtol=0,atol=0)


def test_deferred_parallel_partitions_arrivals_and_pruning(native_library):
    from fly_connectome.deferred_cpu import DeferredCPU
    size=40000
    graph=Graph.from_contacts(list(range(1,size+2)),list(range(1,size+1)),list(range(2,size+2)),
        [1]*size,[1 if i%2 else -1 for i in range(size+1)],.5)
    rule=Plasticity(Network(graph,[1]*size,['predictive']*size),LearningConfig(
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',
        prediction_encoding='signed-current-v1',prune_epsilon=.1))
    rule.keys=torch.arange(0,size,2)
    rule.values=torch.linspace(-.2,.2,len(rule.keys))
    rule.arrival_trace=torch.linspace(0.,.2,len(rule.keys))
    other=copy.deepcopy(rule)
    kernel=DeferredCPU(native_library,threads=4)
    for tick in range(17):
        edges=torch.tensor([0,9999,10000,10000,19999,20000,29999,30000,39999]) if tick in (0,8,16) else torch.empty(0,dtype=torch.long)
        activity=Activity(torch.zeros(1,size+1,dtype=torch.bool),torch.zeros(1,size+1),
            torch.full((1,size+1),(-1.)**tick*3),torch.zeros_like(edges),edges,
            torch.full((1,size+1),(-1.)**tick*.7),torch.zeros(1,size+1))
        rule.observe(activity,torch.zeros(1));kernel.observe(other,activity,torch.zeros(1))
    kernel.materialize(other)
    for name,value in vars(rule).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,getattr(other,name),rtol=0,atol=0)


def test_aggregated_quiet_interval_preserves_trace_and_bounded_local_sum(native_library):
    from fly_connectome.deferred_cpu import DeferredCPU
    graph=Graph.from_contacts([10,20],[10],[20],[1],[1,1],.5)
    rule=Plasticity(Network(graph,[1],['predictive']),LearningConfig(
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1'))
    rule.keys=torch.tensor([0]);rule.values=torch.tensor([.7]);rule.arrival_trace=torch.tensor([1.])
    other=copy.deepcopy(rule)
    kernel=DeferredCPU(native_library,aggregate=True)
    for tick in range(8):
        activity=Activity(torch.zeros(1,2,dtype=torch.bool),torch.zeros(1,2),
            torch.full((1,2),tick*.17),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),
            torch.full((1,2),.13),torch.zeros(1,2))
        rule.observe(activity,torch.zeros(1));kernel.observe(other,activity,torch.zeros(1))
    kernel.materialize(other)
    torch.testing.assert_close(rule.values,other.values,rtol=0,atol=0)
    torch.testing.assert_close(rule.arrival_trace,other.arrival_trace,rtol=0,atol=0)
    torch.testing.assert_close(rule.proposals,other.proposals,rtol=1e-6,atol=1e-10)


@pytest.mark.parametrize('field,kind',[
    ('proposals','strided'),('proposals','half'),('proposals','short'),('current_decay','half')])
def test_deferred_materialization_validates_mutable_pointers(native_library,field,kind):
    from fly_connectome.deferred_cpu import DeferredCPU
    graph=Graph.from_contacts([10,20,30],[10,20],[20,30],[1,1],[1]*3,.5)
    rule=Plasticity(Network(graph,[1,1],['predictive']*2),LearningConfig(
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1'))
    kernel=DeferredCPU(native_library)
    activity=Activity(torch.zeros(1,3,dtype=torch.bool),torch.zeros(1,3),torch.zeros(1,3),
        torch.tensor([0]),torch.tensor([0]),torch.ones(1,3),torch.zeros(1,3))
    kernel.observe(rule,activity,torch.zeros(1))
    owner=rule if field=='proposals' else rule.network
    original=getattr(owner,field)
    invalid=(torch.zeros(4)[::2] if kind=='strided' else
             original.half() if kind=='half' else original[:-1])
    setattr(owner,field,invalid)
    with pytest.raises(ValueError,match='canonical'):
        kernel.synchronize(rule)
    assert len(kernel.pending[rule]['errors'])==1

import sys
from pathlib import Path
import math
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from short_term import DepressionNetwork, FacilitationNetwork
from signed_kinetics import AreaMatchedKineticsNetwork
from frame_prediction import FramePrediction
from fly_connectome.graph import Graph
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.plasticity import LearningConfig


def make(cls,batch=1,**kwargs):
    g=Graph.from_contacts([10,20,30],[10,20],[30,30],[1,1],[1,-1,1],.5)
    return cls(g,[1,1],['predictive']*2,batch=batch,config=NeuronConfig(threshold=100.),**kwargs)


def rule(n):
    return FramePrediction(n,LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',
        eta_prediction=.1,homeostasis_rate=0.))


def inject(n,tick,env=0):
    n.history[(tick-1)%n.history_length,env,:2]=True


@pytest.mark.parametrize('cls,direction',[(DepressionNetwork,-1),(FacilitationNetwork,1)])
@pytest.mark.parametrize('interval',[10,30,100,300])
def test_pair_recovery_and_first_impulse_area(cls,direction,interval):
    n=make(cls);base=make(AreaMatchedKineticsNetwork)
    area=0.;base_area=0.
    for t in range(interval+2):
        if t in [1,interval+1]:inject(n,t)
        if t==1:inject(base,t)
        a=n.step(torch.zeros(1,3));b=base.step(torch.zeros(1,3))
        if t<interval+1:
            torch.testing.assert_close(a.predicted,b.predicted,rtol=0,atol=0)
            area+=float(a.predicted[0,2]);base_area+=float(b.predicted[0,2])
        if t in [1,interval+1]:
            expected=1. if t==1 else 1+direction*.5*math.exp(-interval*.001/.1)
            torch.testing.assert_close(n.arrival_gains,torch.full((2,),expected))
            state=n.release_state.clone()
            for _ in range(3):n.visual_arrival_impulse(a.arrival_environments,a.arrival_edges)
            assert torch.equal(state,n.release_state)
    assert area==base_area


@pytest.mark.parametrize('cls',[DepressionNetwork,FacilitationNetwork])
def test_burst_currents_and_credit_reconstruct_per_arrival(cls):
    n=make(cls);r=rule(n);events=[]
    for t in range(80):
        if t in [1,2,3,4,12,50]:inject(n,t)
        a=n.step(torch.zeros(1,3),capture_increments=True)
        if len(n.arrival_gains):events.append((t,n.arrival_gains.clone()))
        r.observe(a,torch.zeros(1))
        expected=torch.zeros(2)
        for stamp,gain in events:
            expected+=gain*torch.tensor([n.excitatory_gain,-1.])*torch.tensor([
                math.exp(-(t-stamp)*.001/.020),math.exp(-(t-stamp)*.001/.005)])
        torch.testing.assert_close(a.predicted[0,2],.5*expected.sum(),atol=1e-6,rtol=1e-5)
        if len(r.keys):torch.testing.assert_close(r.values,expected[r.keys],atol=1e-6,rtol=1e-5)
        if len(n.arrival_gains):
            low,high=(0,1) if cls is DepressionNetwork else (1,2)
            assert ((n.arrival_gains>=low)&(n.arrival_gains<=high)).all()


@pytest.mark.parametrize('cls',[DepressionNetwork,FacilitationNetwork])
def test_environments_have_private_release_and_silent_state_is_lazy(cls):
    n=make(cls,batch=2)
    for t in range(20):
        if t in [1,2]:inject(n,t,0)
        if t==2:inject(n,t,1)
        before=n.release_state.clone()
        a=n.step(torch.zeros(2,3))
        if t==2:
            gains=n.visual_arrival_impulse(a.arrival_environments,a.arrival_edges)
            base=AreaMatchedKineticsNetwork.visual_impulse(n,a.arrival_edges)
            torch.testing.assert_close(gains[a.arrival_environments==1],base[a.arrival_environments==1])
            assert not torch.equal(gains[a.arrival_environments==0],base[a.arrival_environments==0])
        if t>2:assert torch.equal(before,n.release_state)


@pytest.mark.parametrize('cls',[DepressionNetwork,FacilitationNetwork])
def test_unit_release_learning_matches_baseline_exactly(cls):
    a=make(cls,unit_release=True);b=make(AreaMatchedKineticsNetwork)
    ra,rb=rule(a),rule(b)
    for t in range(40):
        if t%3==1:
            for n in [a,b]:inject(n,t)
        for n,r in [(a,ra),(b,rb)]:
            activity=n.step(torch.zeros(1,3),capture_increments=True)
            r.observe(activity,torch.zeros(1));r.synchronize()
        for name in ['voltage','adaptation','history','magnitudes','predictive_current']:
            assert torch.equal(getattr(a,name),getattr(b,name))
        for name in ['expected','keys','values','proposals']:
            assert torch.equal(getattr(ra,name),getattr(rb,name))


@pytest.mark.parametrize('cls',[DepressionNetwork,FacilitationNetwork])
def test_nonpredictive_transmission_has_no_release_state_or_modulation(cls):
    g=Graph.from_contacts([10,20,30,40],[10,20,30],[40]*3,[1]*3,[1,-1,1,1],.5)
    args=(g,[1]*3,['feedforward','predictive','behavioral'])
    n=cls(*args,config=NeuronConfig(threshold=100.))
    b=AreaMatchedKineticsNetwork(*args,config=NeuronConfig(threshold=100.))
    assert n.release_state.shape==(1,1)
    for t in range(12):
        if t in [1,2,3]:
            for obj in [n,b]:obj.history[(t-1)%obj.history_length,0,:3]=True
        for obj in [n,b]:obj.step(torch.zeros(1,4))
        for key in ['feedforward_current','behavioral_current']:
            assert torch.equal(getattr(n,key),getattr(b,key))


@pytest.mark.parametrize('kind,cls',[('short-term-D-v1',DepressionNetwork),('short-term-F-v1',FacilitationNetwork)])
def test_runner_requires_frame_supervision_and_runs_matching_model(tmp_path,kind,cls):
    from test_training import trainer
    from controlled_visual import crop_payload,make_network,run_sequence,trajectory
    model=trainer();path=tmp_path/'source.pt';model.save(path)
    payload=torch.load(path,weights_only=True);m=payload['metadata']
    m['config']['neural_steps']=8;m['learning']['visual_eligibility']='forecast-causal-v1'
    crop,_,_=crop_payload(payload,[0,1,2,3])
    n=make_network(crop,m,predictive_kinetics=kind)
    assert isinstance(n,cls)
    with pytest.raises(ValueError,match='requires frame-horizon'):
        run_sequence(n,crop,m,[],[0],learning=True)
    r=run_sequence(n,crop,m,trajectory(32,64,[1,2],0,blank=1),[0],learning=True,visual_schedule='frame-horizon-v1')
    assert r['stability']['finite'] and r['stability']['within_weight_bounds']

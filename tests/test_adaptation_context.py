import sys
from pathlib import Path
import math
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from adaptation_context import AdaptationContextNetwork, AdaptationExcitatoryNetwork, AdaptationPrediction, context_gains
from signed_kinetics import AreaMatchedKineticsNetwork
from fly_connectome.graph import Graph
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.plasticity import LearningConfig


def make(cls):
    graph=Graph.from_contacts([10,20,30],[10,20],[30,30],[1,1],[1,-1,1],.5)
    return cls(graph,[1,1],['predictive','predictive'],config=NeuronConfig(dt=1/960,threshold=100.))


def test_gains_bounded_complementary_and_identity_at_threshold():
    a=torch.tensor([0.,1.,10.,1e6])
    e,i=context_gains(a,1.)
    assert ((e>=0)&(e<=2)&(i>=0)&(i<=2)).all()
    torch.testing.assert_close(e+i,torch.full_like(a,2.))
    assert e[1]==i[1]==1 and e[0]==2 and i[0]==0


@pytest.mark.parametrize('cls',[AdaptationContextNetwork,AdaptationExcitatoryNetwork])
def test_frozen_physics_and_independent_signed_impulse_reconstruction(cls):
    net=make(cls);base=make(AreaMatchedKineticsNetwork)
    for n in (net,base):n.adaptation.fill_(50.)
    for tick in range(24):
        if tick==1:
            for n in (net,base):n.history[0,0,:2]=True
        a,b=[n.step(torch.zeros(1,3),capture_increments=True) for n in (net,base)]
        for name in ['voltage','adaptation','refractory','history','predictive_current','excitatory_prediction','inhibitory_prediction']:
            torch.testing.assert_close(getattr(net,name),getattr(base,name),rtol=0,atol=0)
        torch.testing.assert_close(a.spikes,b.spikes,rtol=0,atol=0)
        ge,gi=net.forecast_gains(net.adaptation)
        expected=0. if tick==0 else ge[0,2]*.5*net.excitatory_gain*math.exp(-(tick-1)*net.config.dt/.020)-gi[0,2]*.5*math.exp(-(tick-1)*net.config.dt/.005)
        torch.testing.assert_close(a.predicted[0,2],torch.as_tensor(expected),atol=2e-7,rtol=1e-5)


@pytest.mark.parametrize('cls',[AdaptationContextNetwork,AdaptationExcitatoryNetwork])
def test_credit_uses_issue_time_gain_even_when_adaptation_changes(cls):
    net=make(cls)
    rule=AdaptationPrediction(net,LearningConfig(prediction_encoding='signed-current-v1',
        visual_target='input-arrivals-v1',visual_eligibility='forecast-causal-v1',eta_prediction=.1,homeostasis_rate=0.))
    for tick in range(25):
        net.adaptation.fill_(10.+tick*7.)
        if tick==1:net.history[0,0,:2]=True
        activity=net.step(torch.zeros(1,3),capture_increments=True)
        old=rule.forecast;before=rule.proposals.clone()
        rule.observe(activity,torch.zeros(1))
        if tick%8==0:
            if old is not None:
                keys,credit,prediction=old;expected=torch.zeros(net.e)
                expected[keys]=-.1*prediction*credit
                torch.testing.assert_close(rule.proposals-before,expected,atol=1e-9,rtol=1e-5)
            keys,credit,_=rule.forecast
            ge,gi=net.forecast_gains(net.adaptation[0,net.post[keys]])
            torch.testing.assert_close(credit,rule.values*torch.where(net.signs[keys]>0,ge,gi))
        else:torch.testing.assert_close(rule.proposals,before,rtol=0,atol=0)


def test_unit_gain_control_recovers_original_forecast_bit_for_bit():
    class Identity(AdaptationContextNetwork):
        def forecast_gains(self,a):return torch.ones_like(a),torch.ones_like(a)
    net,base=make(Identity),make(AreaMatchedKineticsNetwork)
    for tick in range(24):
        if tick%4==1:
            for n in (net,base):n.history[(tick-1)%n.history_length,0,:2]=True
        a,b=[n.step(torch.zeros(1,3)) for n in (net,base)]
        torch.testing.assert_close(a.predicted,b.predicted,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['adaptation-A-v1','adaptation-B-v1'])
def test_runner_selects_matching_rule_and_rejects_tick_supervision(tmp_path,monkeypatch,kind):
    from test_training import trainer
    from controlled_visual import crop_payload,make_network,run_sequence,trajectory
    model=trainer();path=tmp_path/'source.pt';model.save(path)
    payload=torch.load(path,weights_only=True);m=payload['metadata']
    m['config']['neural_steps']=8
    m['learning']['visual_eligibility']='forecast-causal-v1'
    crop,_,_=crop_payload(payload,[0,1,2,3])
    net=make_network(crop,m,predictive_kinetics=kind)
    with pytest.raises(ValueError,match='requires frame-horizon'):
        run_sequence(net,crop,m,[],[0],learning=True)
    calls=[];observe=AdaptationPrediction.observe
    def spy(self,activity,reward):
        calls.append(self.tick)
        return observe(self,activity,reward)
    monkeypatch.setattr(AdaptationPrediction,'observe',spy)
    frames=trajectory(32,64,[1,2],0,blank=1)
    r=run_sequence(net,crop,m,frames,[0],learning=True,visual_schedule='frame-horizon-v1')
    assert len(calls)==len(frames)*8 and r['stability']['finite']

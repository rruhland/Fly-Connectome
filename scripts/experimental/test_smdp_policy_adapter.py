"""Runtime SMDP duration, actual-event boundary and public-history integration."""
import numpy as np
from pong_smdp_adaptive_forest import AdaptiveForest,SMDPController
from pong_smdp_policy_adapter import choose,observe


class Forest:
    def __init__(self,option=2):self.option=option;self.calls=[]
    def inspect(self,scene,phase,past_hit):
        self.calls.append((phase,past_hit));return dict(option=self.option,q=[.8,0,1,.2],supported=[True]*4)


def record(hit=0,terminal=False):return dict(hits=hit,terminated=terminal)


def test_exact16_actual_commands_then_reselection():
    f=Forest();c=SMDPController(f);observe(c,record(1));actions=[]
    for i in range(16):
        a,d=choose(c,[],0,[0,0,0]);actions.append(a);assert d['phase']==(i+1)%5;observe(c,record())
    assert actions==[2]*16 and len(f.calls)==1 and c.remaining==0
    assert choose(c,[],0,[0,0,0])[1]['mode']=='earned_start' and len(f.calls)==2


def test_early_actual_hit_or_natural_end_stops_held_option():
    f=Forest();c=SMDPController(f);observe(c,record(1));choose(c,[],0,[0,0,0]);observe(c,record());choose(c,[],0,[0,0,0]);observe(c,record(1));assert c.remaining==0 and c.held_action is None
    choose(c,[],0,[0,0,0]);assert len(f.calls)==2;observe(c,record(0,True));assert c.remaining==0
    fresh=SMDPController(f);assert not fresh.past_hit and fresh.ordinal==0 and fresh.remaining==0


def test_baseline_before_actual_hit_and_onecamera_option3():
    f=Forest(3);c=SMDPController(f)
    assert choose(c,[],0,[0,0,0])[0]==0 and not f.calls;observe(c,record());assert choose(c,[],1,[0,0,0])[0]==1 and not f.calls
    observe(c,record(1));assert choose(c,[],2,[0,0,0])[0]==2;observe(c,record());assert choose(c,[],0,[0,0,0])[0]==0 and len(f.calls)==2 and c.remaining==0


def test_zero_or_unsupported_forest_falls_back():
    shape=(32,1);feature=np.full(shape,-2);threshold=np.zeros(shape);child=np.full(shape,-1);value=np.zeros(shape)
    for count in (0,1):
        f=AdaptiveForest(np.ones(4),feature,threshold,child,child,value,np.full((*shape,4),count))
        c=SMDPController(f);observe(c,record(1));assert choose(c,np.array([[.1,.2,0,0,1,0]]),1,[0,0,0])[0]==1
        assert choose(c,np.zeros((5,6)),0,[0,0,0])[0]==0

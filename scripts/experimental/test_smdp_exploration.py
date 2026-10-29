"""Macro-level probability, actual-duration and reward-context exploration checks."""
from pong_smdp_exploration import Explorer


class Forest:
    def __init__(self):self.calls=0
    def inspect(self,scene,phase,past_hit):
        self.calls+=1;return dict(option=1,q=[.2,.8,.1,.4],supported=[True]*4)


class Draws:
    def __init__(self,values):self.values=iter(values);self.probabilities=[]
    def choice(self,n,p):self.probabilities.append(p.tolist());return next(self.values)


def test_no_exploration_before_real_hit_and_exact_boundary_probabilities():
    f=Forest();c=Explorer(f,11);c.rng=Draws([0,3,2])
    assert c.choose([],2)[1]['behavior_p']==[0,0,0,1] and f.calls==0;c.observe(0)
    assert c.choose([],0)[0]==0 and not c.rng.probabilities;c.observe(1)
    action,d=c.choose([],1);assert action==0 and d['selected_option']==0 and d['greedy_option']==1 and d['option_decision']
    assert d['behavior_p']==[.0625,.8125,.0625,.0625] and sum(d['behavior_p'])==1
    c.observe(0)
    for i in range(15):
        action,d=c.choose([],2);assert action==0 and not d['option_decision'] and d['selected_option']==-1 and d['behavior_p'] is None;c.observe(0)
    assert c.remaining==0 and len(c.rng.probabilities)==1
    assert c.choose([],2)[0]==2 and c.remaining==0;c.observe(0)
    assert c.choose([],0)[0]==2 and len(c.rng.probabilities)==3


def test_actual_hit_and_natural_end_interrupt_held_option():
    c=Explorer(Forest(),1);c.rng=Draws([0,2,1]);c.observe(1);c.choose([],1);c.observe(0);c.choose([],1);c.observe(1)
    assert c.remaining==0 and c.held_action is None
    assert c.choose([],0)[0]==2;c.observe(0,True);assert c.remaining==0


def test_declared_seed_reproduces_full_exploration_history():
    a,b=Explorer(Forest(),2026102903),Explorer(Forest(),2026102903)
    for i in range(300):
        assert a.choose([],i%3)==b.choose([],i%3)
        for c in (a,b):c.observe(int(i in (7,82,201)))
    assert a.ordinal==b.ordinal==300

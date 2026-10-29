import numpy as np
from pong_relational_feedback import RelationalPolicy
from pong_relational_opening_adapter import OpeningController


def test_opening_acts_before_reward_without_faking_reward_history():
    p=RelationalPolicy(2,0,1,0,1,np.zeros(5),np.ones(4));c=OpeningController(p)
    scene=np.array([[.1,.1,0,0,1,0],[.9,.9,0,0,1,0]])
    action,d=c.choose(scene,1)
    assert action==2 and not d['past_hit'] and not c.past_hit
    c.observe(0)
    assert c.previous_action==2 and c.ordinal==1 and not c.past_hit
    c.choose(scene,0);c.observe(1)
    assert c.past_hit


def test_opening_keeps_fallback_and_actual_command_memory():
    p=RelationalPolicy(2,0,1,0,1,[0,0,0,0,2],np.ones(4));c=OpeningController(p)
    for scene,base in ((np.empty((0,6)),0),(np.array([[.1,.1,0,0,1,0]]),2)):
        action,d=c.choose(scene,base);assert action==base and not d['supported']
        c.observe(0)
    equal_y=np.array([[.5,.1,0,0,1,0],[.5,.9,0,0,1,0]])
    action,d=c.choose(equal_y,1)
    assert action==2 and d['previous_actual_action']==2 and not d['past_hit']

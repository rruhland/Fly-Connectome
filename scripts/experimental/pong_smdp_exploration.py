"""Frozen candidate0 with declared macro-boundary epsilon exploration only."""
import numpy as np
from pong_smdp_adaptive_forest import HOLD

CONTRACT='frozen-c0-posthit-epsilon-quarter-fouroptions-macro16-baseline1-v1'
EPSILON=.25


class Explorer:
    def __init__(self,forest,seed):
        self.forest=forest;self.rng=np.random.default_rng(seed);self.ordinal=0;self.past_hit=False;self.remaining=0;self.held_action=None
    def choose(self,scene,baseline_action):
        before=self.remaining
        if not self.past_hit:
            option=greedy=3;p=[0.,0.,0.,1.];decision=dict(mode='baseline_before_hit',option_decision=True)
        elif before:
            option=self.held_action;greedy=-1;p=None;self.remaining-=1;decision=dict(mode='earned_hold',option_decision=False)
        else:
            decision=self.forest.inspect(scene,self.ordinal%5,self.past_hit);greedy=int(decision['option']);p=np.full(4,EPSILON/4);p[greedy]+=1-EPSILON;option=int(self.rng.choice(4,p=p));p=p.tolist();self.held_action=None if option==3 else option;self.remaining=0 if option==3 else HOLD-1;decision.update(mode='baseline_camera' if option==3 else 'earned_start',option_decision=True)
        action=int(baseline_action) if option==3 else int(option)
        decision.update(option=int(option),selected_option=int(option) if decision['option_decision'] else -1,executed_option=int(option),greedy_option=int(greedy),behavior_p=p,action=action,baseline_action=int(baseline_action),past_hit=self.past_hit,ordinal=self.ordinal,phase=self.ordinal%5,hold_before=before,hold_after=self.remaining)
        return action,decision
    def observe(self,hits,terminal=False):
        self.ordinal+=1;self.past_hit=self.past_hit or bool(hits)
        if hits or terminal:self.remaining=0;self.held_action=None


class ExplorationAdapter:
    CONTRACT=CONTRACT
    def __init__(self,seed):self.seed=int(seed)
    def new_controller(self,forest):return Explorer(forest,self.seed)
    def choose(self,controller,scene,baseline_action,baseline_q):return controller.choose(scene,baseline_action)
    def observe(self,controller,record):controller.observe(record['hits'],record['terminated'])

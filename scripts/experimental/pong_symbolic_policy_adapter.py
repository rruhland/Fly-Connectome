"""Stateful public ordinal/reward wrapper for final frozen symbolic evaluation."""
from pong_symbolic_program import load_program

CONTRACT='symbolic-currentanonymouspairs-baselineuntilactualhit-camera-mod5-v1'
load_model=load_program


class Controller:
    def __init__(self,program):
        self.program=program;self.ordinal=0;self.past_hit=False

    def choose(self,scene,baseline_action,baseline_q):
        if self.past_hit:
            action,q=self.program.choose(scene,baseline_q,self.ordinal%5,True)
            mode='symbolic_post_hit'
        else:
            action,q=baseline_action,baseline_q;mode='baseline_before_hit'
        return int(action),dict(mode=mode,action=int(action),baseline_action=int(baseline_action),
            ordinal=self.ordinal,phase=self.ordinal%5,past_hit=self.past_hit)

    def observe(self,row):
        self.ordinal+=1;self.past_hit |= bool(row['hits'])


new_controller=Controller


def choose(controller,scene,baseline_action,baseline_q):
    return controller.choose(scene,baseline_action,baseline_q)


def observe(controller,record):
    controller.observe(record)

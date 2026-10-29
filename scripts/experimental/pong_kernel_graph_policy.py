"""Frozen legal-scene graph control with exact actual-reward/camera history."""
import numpy as np
from fly_connectome.control.scaling import policy_features
from fly_connectome.control.durable import fingerprint
from pong_continuous_visual import canonical_scene
from pong_fixed_rbf import FreshRBFEvaluator

CONTRACT='frozen-kernel-graph-actualhit-publiccamera-mod5-supported-strictgain-v1'


def graph_parameters(graph):
    return (CONTRACT,graph.metric.scales,graph.metric.radius,graph.index.vectors,
            graph.index.keys,graph.index.actions,graph.index.start_ids,graph.index.radius,graph.values)


def freeze_graph(graph):
    for a in (graph.metric.scales,graph.index.vectors,graph.index.keys,graph.index.actions,graph.index.start_ids,graph.values):a.flags.writeable=False
    return graph


class GraphController:
    def __init__(self,graph):self.graph=graph;self.ordinal=0;self.past_hit=False
    def choose(self,scene,baseline_action):
        decision=self.graph.inspect(scene,self.ordinal%5,self.past_hit,baseline_action)
        decision.update(camera_ordinal=self.ordinal,camera_phase=self.ordinal%5,past_hit=self.past_hit)
        return int(decision['action']),decision
    def feedback(self,hits):self.past_hit=self.past_hit or bool(hits);self.ordinal+=1


class GraphEvaluator(FreshRBFEvaluator):
    def __init__(self,baseline,graph,config):
        self.kernel_graph=freeze_graph(graph);self.controller=GraphController(graph);self.graph_choices=[]
        super().__init__(baseline,config);self.initial_learned_hash=self.learned_hash()
    def learned_hash(self):return fingerprint((super().learned_hash(),graph_parameters(self.kernel_graph)))
    def _begin_episode(self):self.controller=GraphController(self.kernel_graph);super()._begin_episode()
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler));action,choice=self.controller.choose(canonical_scene(self.latest_encoded),baseline);self.graph_choices.append(choice);return action,q
    def step(self,**kwargs):
        r=super().step(**kwargs);self.controller.feedback(r['hits']);return r

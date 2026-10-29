"""Actual-reward/public-camera history and conservative graph policy integration."""
import numpy as np
import pytest
from pong_kernel_transition_graph import KernelGraph,SceneMetric,TransitionIndex
from pong_kernel_graph_policy import GraphController,freeze_graph,graph_parameters
from fly_connectome.control.durable import fingerprint


def graph_for(actions=(0,1,2),values=(.2,.8,.8)):
    metric=SceneMetric(np.ones(4),.2);vectors=[];keys=[];acts=[];starts=[];V=[]
    for phase in range(5):
        v,k=metric.encode([],phase,True)
        for action,value in zip(actions,values):
            for start in (11,12):vectors.append(v);keys.append(k);acts.append(action);starts.append(start);V.append(value)
    return KernelGraph(metric,TransitionIndex(vectors,keys,acts,starts,.2),V)


def test_public_preaction_ordinal_and_actual_hit_gate():
    c=GraphController(graph_for())
    for i in range(7):
        action,d=c.choose([],0);assert action==0 and d['camera_ordinal']==i and d['camera_phase']==i%5 and not d['past_hit'];c.feedback(0)
    c.feedback(1)  # The observed scalar reward, not a predicted graph value, enables control.
    action,d=c.choose([],0);assert action==1 and d['past_hit'] and d['camera_phase']==3
    c=GraphController(c.graph);assert c.choose([],0)[0]==0 and c.ordinal==0 and not c.past_hit


def test_strict_gain_baseline_and_alternative_support_and_ties():
    c=GraphController(graph_for());c.feedback(1)
    assert c.choose([],0)[0]==1
    assert c.choose([],2)[0]==2  # equal-valued lower-index action cannot displace baseline
    c=GraphController(graph_for((1,2),(.8,.9)));c.feedback(1)
    assert c.choose([],0)[0]==0  # an unsupported baseline cannot be displaced
    assert c.choose(np.zeros((5,6)),1)[0]==1  # overflow is unsupported


def test_graph_asset_guard_excludes_mutable_controller_state():
    graph=freeze_graph(graph_for());before=fingerprint(graph_parameters(graph));c=GraphController(graph)
    for i in range(12):c.choose([],0);c.feedback(i==3)
    assert fingerprint(graph_parameters(graph))==before
    with pytest.raises(ValueError):graph.values[0]=.5
    with pytest.raises(ValueError):graph.metric.scales[0]=2

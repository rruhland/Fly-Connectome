"""Explicit optional B=1 CPU kernels; tensor state and checkpoint format stay shared."""
import ctypes
from functools import partial
import math
from pathlib import Path
import subprocess
import torch
from .dynamics import Activity


def build_library(output, compiler='g++'):
    """Explicit build only: no compiler or library downloads during normal training."""
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([compiler,'-O3','-fno-fast-math','-ffp-contract=off','-shared',
        '-static-libgcc','-static-libstdc++',str(Path(__file__).with_suffix('.cpp')),
        '-o',str(output)],check=True)


class NativeCPU:
    def __init__(self, library):
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        self.sparse=self.library.sparse_observe
        self.sparse.argtypes=[ctypes.c_int64]*3+[ctypes.c_float]*6+[ctypes.c_void_p]*16
        self.sparse.restype=ctypes.c_int64
        self.neural=self.library.neural_step
        self.neural.argtypes=[ctypes.c_int64]*5+[ctypes.c_float]*7+[ctypes.c_void_p]*21
        self.neural.restype=None

    def enable(self, trainer):
        if trainer.network.batch!=1 or trainer.network.voltage.device.type!='cpu':
            raise ValueError('native CPU backend requires B=1 on CPU')
        trainer.network.step=partial(self.step,trainer.network)
        if trainer.plasticity is not None:
            trainer.plasticity.observe=partial(self.observe,trainer.plasticity)

    @torch.no_grad()
    def step(self, n, sensory_current, *, capture_increments=False):
        if n.batch!=1 or n.voltage.device.type!='cpu':
            raise ValueError('native CPU dynamics requires B=1 CPU tensors')
        if sensory_current.shape!=(1,n.n) or sensory_current.dtype!=torch.float32 or not sensory_current.is_contiguous() or sensory_current.device.type!='cpu':
            raise ValueError('sensory current must match canonical CPU network shape and dtype')
        cfg=n.config
        env,edges=n._arrivals()
        observed,predicted=torch.empty_like(n.voltage),torch.empty_like(n.voltage)
        increments=torch.empty_like(n.voltage) if capture_increments else None
        spikes=torch.empty_like(n.voltage,dtype=torch.bool)
        pointers=(edges,n.post,n.pathways,n.magnitudes,n.signs,n.current_decay,n.membrane_decay,
            n.rest_current,sensory_current,n.silenced,n.feedforward_current,n.predictive_current,
            n.behavioral_current,n.sensory_state,n.adaptation,n.voltage,n.refractory,observed,
            predicted,increments,spikes)
        decay=math.exp(-cfg.dt/cfg.tau_membrane)
        self.neural(n.n,len(edges),bool(cfg.class_parameters),cfg.refractory_steps,bool(cfg.tau_sensory),
            math.exp(-cfg.dt/cfg.tau_current),decay,1-decay,
            math.exp(-cfg.dt/cfg.tau_sensory) if cfg.tau_sensory else 0.,
            math.exp(-cfg.dt/cfg.tau_adaptation),cfg.threshold,cfg.adaptation_jump,
            *(x.data_ptr() if x is not None else None for x in pointers))
        n.history[n.step_index%n.history_length].copy_(spikes)
        n.step_index+=1
        return Activity(spikes,observed,predicted,env,edges,increments,
                        sensory_current.detach() if capture_increments else None)

    @torch.no_grad()
    def observe(self, rule, activity, reward):
        n,cfg=rule.network,rule.config
        if n.batch!=1 or n.voltage.device.type!='cpu':
            raise ValueError('native CPU learning requires B=1 CPU tensors')
        if reward.shape!=(1,) or not torch.isfinite(reward).all():
            raise ValueError('one finite scalar reward per environment required')
        observed=cfg.observation(activity,n.config.threshold,rule.sensory_mask,rule.sensory_gain)
        pair_decay=math.exp(-n.config.dt/cfg.tau_pair)
        rule.post_trace.mul_(pair_decay)
        edges=activity.arrival_edges
        incoming=edges[n.pathways[edges]!=0].clone()
        size=len(rule.keys)+len(incoming)
        keys=rule.keys.new_empty(size)
        values=rule.values.new_empty(size)
        traces=rule.arrival_trace.new_empty(size)
        pointers=(rule.keys,rule.values,rule.arrival_trace,incoming,n.post,n.pathways,
                  n.signs,n.current_decay,observed,rule.expected,rule.post_trace,
                  activity.spikes,rule.proposals,keys,values,traces)
        # All state is canonical contiguous float32/int64; reject pointer reinterpretation.
        for tensor,dtype in zip(pointers,(torch.int64,torch.float32,torch.float32,torch.int64,
                torch.int64,torch.int64,torch.float32,torch.float32,torch.float32,torch.float32,
                torch.float32,torch.bool,torch.float32,torch.int64,torch.float32,torch.float32)):
            if tensor.device.type!='cpu' or tensor.dtype!=dtype or not tensor.is_contiguous():
                raise ValueError('native CPU kernel requires canonical contiguous tensor types')
        count=self.sparse(len(rule.keys),len(incoming),int(cfg.visual_eligibility=='forecast-causal-v1'),
            cfg.eta_prediction,cfg.eta_reward,math.exp(-n.config.dt/cfg.tau_eligibility),pair_decay,
            cfg.prune_epsilon,float(reward.clamp(-1,1)[0]),*(x.data_ptr() for x in pointers))
        rule.keys,rule.values,rule.arrival_trace=keys[:count],values[:count],traces[:count]
        rule.expected.copy_(cfg.encode(activity.predicted,n.config.threshold))
        rule.post_trace.add_(activity.spikes)
        rule.rates.lerp_(activity.spikes.float()/n.config.dt,1-math.exp(-n.config.dt/cfg.tau_homeostasis))
        overload=(rule.rates[0]-cfg.maximum_rate).clamp(min=0)
        if overload.any():
            rule.homeostatic_exponent.add_(overload[n.post],alpha=cfg.homeostasis_rate*n.config.dt)

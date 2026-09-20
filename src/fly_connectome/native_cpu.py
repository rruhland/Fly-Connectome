"""Explicit optional B=1 CPU kernels; tensor state and checkpoint format stay shared."""
import ctypes
from functools import partial
import math
import os
from pathlib import Path
import subprocess
import weakref
import torch
from .dynamics import Activity


def _canonical(tensor, dtype, shape):
    if (tensor.device.type!='cpu' or tensor.dtype!=dtype or
            tensor.shape!=shape or not tensor.is_contiguous()):
        raise ValueError('native CPU kernel requires canonical contiguous tensor types and shapes')


def build_library(output, compiler='g++'):
    """Explicit build only: no compiler or library downloads during normal training."""
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([compiler,'-O3','-fno-fast-math','-ffp-contract=off','-shared','-fopenmp',
        *(['-static'] if os.name=='nt' else []),
        '-static-libgcc','-static-libstdc++',str(Path(__file__).with_suffix('.cpp')),
        '-o',str(output)],check=True)


class NativeCPU:
    def __init__(self, library, threads=1):
        if not isinstance(threads,int) or threads<1:
            raise ValueError('native thread count must be positive')
        self.threads=threads
        self.topologies=weakref.WeakKeyDictionary()
        self.rules=weakref.WeakSet()
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        version=getattr(self.library,'native_abi_version',None)
        if version is None:
            raise ValueError('native library ABI is outdated; rebuild with build-native')
        version.argtypes=[]
        version.restype=ctypes.c_int
        if version()!=8:
            raise ValueError('native library ABI is unsupported; rebuild with build-native')
        self.sparse=self.library.sparse_observe
        self.sparse.argtypes=[ctypes.c_int64]*5+[ctypes.c_float]*6+[ctypes.c_void_p]*16
        self.sparse.restype=ctypes.c_int64
        self.neural=self.library.neural_step
        self.neural.argtypes=[ctypes.c_int64]*6+[ctypes.c_float]*7+[ctypes.c_void_p]*21
        self.neural.restype=None
        self.prepare_fn=self.library.prepare_observation
        self.prepare_fn.argtypes=[ctypes.c_int64]*3+[ctypes.c_float]*4+[ctypes.c_void_p]*6
        self.prepare_fn.restype=None
        self.sync_fn=self.library.synchronize_weights
        self.sync_fn.argtypes=[ctypes.c_int64]*2+[ctypes.c_float]+[ctypes.c_void_p]*3
        self.sync_fn.restype=ctypes.c_int
        self.arrivals_fn=self.library.spike_arrivals
        self.arrivals_fn.argtypes=[ctypes.c_int64]*3+[ctypes.c_void_p]*4
        self.arrivals_fn.restype=ctypes.c_int64
        self.finish_fn=self.library.finish_observation
        self.finish_fn.argtypes=[ctypes.c_int64]*2+[ctypes.c_float]*3+[ctypes.c_void_p]*6
        self.finish_fn.restype=None

    @torch.no_grad()
    def finish(self, rule, activity):
        n,cfg=rule.network,rule.config
        for tensor in (activity.predicted,rule.expected,rule.post_trace,rule.rates):
            _canonical(tensor,torch.float32,(1,n.n))
        _canonical(activity.spikes,torch.bool,(1,n.n))
        # Torch uses hardware-dependent vector/tail rounding for lerp; retain it.
        rule.rates.lerp_(activity.spikes.float()/n.config.dt,1-math.exp(-n.config.dt/cfg.tau_homeostasis))
        overload=torch.empty(n.n,dtype=torch.float32,device='cpu')
        self.finish_fn(n.n,self.threads,n.config.threshold,
            -1. if cfg.prediction_encoding=='signed-current-v1' else 0.,cfg.maximum_rate,
            *(x.data_ptr() for x in (activity.predicted,activity.spikes,rule.expected,
                                     rule.post_trace,rule.rates,overload)))
        return overload

    @torch.no_grad()
    def arrivals(self, n):
        self.topology(n)
        args=(n.n,n.history_length,n.step_index,n.history.data_ptr(),n.starts.data_ptr(),n.delays.data_ptr())
        count=self.arrivals_fn(*args,None)
        edges=torch.empty(count,dtype=torch.int64,device='cpu')
        self.arrivals_fn(*args,edges.data_ptr())
        return torch.zeros_like(edges),edges

    @torch.no_grad()
    def synchronize(self, rule):
        for tensor in (rule.network.magnitudes,rule.proposals,rule.homeostatic_exponent):
            _canonical(tensor,torch.float32,(rule.network.e,))
        done=self.sync_fn(rule.network.e,self.threads,rule.config.maximum_weight,
            rule.network.magnitudes.data_ptr(),rule.proposals.data_ptr(),rule.homeostatic_exponent.data_ptr())
        if not done:
            from .plasticity import Plasticity
            Plasticity.synchronize(rule)
            return

    @torch.no_grad()
    def prepare(self, rule, activity):
        n,cfg=rule.network,rule.config
        increments=cfg.visual_target=='input-arrivals-v1'
        if increments and (activity.feedforward_arrivals is None or activity.sensory_input is None):
            raise ValueError('input-arrivals target requires captured local increments')
        for tensor in (activity.observed,rule.expected,rule.post_trace):
            _canonical(tensor,torch.float32,(1,n.n))
        if increments:
            for tensor in (activity.feedforward_arrivals,activity.sensory_input):
                _canonical(tensor,torch.float32,(1,n.n))
        _canonical(rule.sensory_mask,torch.bool,(n.n,))
        observed=torch.empty_like(rule.expected)
        self.prepare_fn(n.n,increments,self.threads,n.config.threshold,rule.sensory_gain,
            math.exp(-n.config.dt/cfg.tau_pair),-1. if cfg.prediction_encoding=='signed-current-v1' else 0.,
            *(x.data_ptr() if x is not None else None for x in (activity.observed,
              activity.feedforward_arrivals,activity.sensory_input,rule.sensory_mask,rule.post_trace,observed)))
        return observed

    def topology(self, network):
        if network.n>2**31-1:
            raise ValueError('native CPU node indices require a roster below 2**31')
        if network not in self.topologies:
            n=network
            if n.batch!=1:
                raise ValueError('native CPU backend requires B=1 on CPU')
            for name in ('voltage','feedforward_current','predictive_current','behavioral_current',
                         'sensory_state','adaptation'):
                _canonical(getattr(n,name),torch.float32,(1,n.n))
            for name in ('rest_current','membrane_decay','current_decay'):
                _canonical(getattr(n,name),torch.float32,(n.n,))
            _canonical(n.refractory,torch.int64,(1,n.n))
            _canonical(n.silenced,torch.bool,(n.n,))
            _canonical(n.history,torch.bool,(n.history_length,1,n.n))
            _canonical(n.starts,torch.int64,(n.n+1,))
            for name in ('post','delays'):
                _canonical(getattr(n,name),torch.int64,(n.e,))
            for name in ('magnitudes','signs'):
                _canonical(getattr(n,name),torch.float32,(n.e,))
            # Empty pathway lists have the reference's default floating dtype.
            if n.e:
                _canonical(n.pathways,torch.int64,(n.e,))
            if (n.starts[0]!=0 or n.starts[-1]!=n.e or (n.starts[1:]<n.starts[:-1]).any()
                    or ((n.post<0)|(n.post>=n.n)).any()
                    or ((n.pathways<0)|(n.pathways>2)).any()
                    or ((n.signs!=-1)&(n.signs!=0)&(n.signs!=1)).any()):
                raise ValueError('native CPU topology indices and signs must be canonical')
            # A private compact cache of immutable anatomy, not a different graph.
            self.topologies[network]=(network.post.to(torch.int32),network.pathways.to(torch.uint8),
                                      network.signs.to(torch.int8))
        return self.topologies[network]

    def enable(self, trainer):
        if trainer.network.batch!=1 or trainer.network.voltage.device.type!='cpu':
            raise ValueError('native CPU backend requires B=1 on CPU')
        self.topology(trainer.network)
        # Flush private lazy state before replacing its execution callbacks.
        if hasattr(trainer,'_materialize'):
            trainer._materialize()
            del trainer._materialize
        trainer.network._arrivals=partial(self.arrivals,trainer.network)
        trainer.network.step=partial(self.step,trainer.network)
        if trainer.plasticity is not None:
            trainer.plasticity.observe=partial(self.observe,trainer.plasticity)
            trainer.plasticity.synchronize=partial(self.synchronize,trainer.plasticity)

    @torch.no_grad()
    def step(self, n, sensory_current, *, capture_increments=False):
        if n.batch!=1 or n.voltage.device.type!='cpu':
            raise ValueError('native CPU dynamics requires B=1 CPU tensors')
        if sensory_current.shape!=(1,n.n) or sensory_current.dtype!=torch.float32 or not sensory_current.is_contiguous() or sensory_current.device.type!='cpu':
            raise ValueError('sensory current must match canonical CPU network shape and dtype')
        cfg=n.config
        post,pathways,signs=self.topology(n)
        env,edges=n._arrivals()
        observed,predicted=torch.empty_like(n.voltage),torch.empty_like(n.voltage)
        increments=torch.empty_like(n.voltage) if capture_increments else None
        spikes=torch.empty_like(n.voltage,dtype=torch.bool)
        pointers=(edges,post,pathways,n.magnitudes,signs,n.current_decay,n.membrane_decay,
            n.rest_current,sensory_current,n.silenced,n.feedforward_current,n.predictive_current,
            n.behavioral_current,n.sensory_state,n.adaptation,n.voltage,n.refractory,observed,
            predicted,increments,spikes)
        decay=math.exp(-cfg.dt/cfg.tau_membrane)
        self.neural(n.n,len(edges),bool(cfg.class_parameters),cfg.refractory_steps,bool(cfg.tau_sensory),self.threads,
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
        post,pathways,signs=self.topology(n)
        for tensor in (rule.expected,rule.post_trace,rule.rates,activity.predicted):
            _canonical(tensor,torch.float32,(1,n.n))
        _canonical(activity.spikes,torch.bool,(1,n.n))
        for tensor in (rule.proposals,rule.homeostatic_exponent):
            _canonical(tensor,torch.float32,(n.e,))
        _canonical(rule.keys,torch.int64,(len(rule.keys),))
        for tensor in (rule.values,rule.arrival_trace):
            _canonical(tensor,torch.float32,(len(rule.keys),))
        if rule not in self.rules:
            if (((rule.keys<0)|(rule.keys>=n.e)).any() or (rule.keys[1:]<=rule.keys[:-1]).any()):
                raise ValueError('native CPU eligibility keys must be sorted unique valid edges')
            self.rules.add(rule)
        _canonical(activity.arrival_edges,torch.int64,(len(activity.arrival_edges),))
        _canonical(activity.arrival_environments,torch.int64,activity.arrival_edges.shape)
        if (activity.arrival_environments.any() or
                ((activity.arrival_edges<0)|(activity.arrival_edges>=n.e)).any()):
            raise ValueError('native CPU arrivals must address valid B=1 edges')
        observed=self.prepare(rule,activity)
        pair_decay=math.exp(-n.config.dt/cfg.tau_pair)
        edges=activity.arrival_edges
        incoming=edges[pathways[edges]!=0].clone()
        size=len(rule.keys)+len(incoming)
        keys=rule.keys.new_empty(size)
        values=rule.values.new_empty(size)
        traces=rule.arrival_trace.new_empty(size)
        pointers=(rule.keys,rule.values,rule.arrival_trace,incoming,post,pathways,
                  signs,n.current_decay,observed,rule.expected,rule.post_trace,
                  activity.spikes,rule.proposals,keys,values,traces)
        # All state is canonical contiguous float32/int64; reject pointer reinterpretation.
        for tensor,dtype in zip(pointers,(torch.int64,torch.float32,torch.float32,torch.int64,
                torch.int32,torch.uint8,torch.int8,torch.float32,torch.float32,torch.float32,
                torch.float32,torch.bool,torch.float32,torch.int64,torch.float32,torch.float32)):
            if tensor.device.type!='cpu' or tensor.dtype!=dtype or not tensor.is_contiguous():
                raise ValueError('native CPU kernel requires canonical contiguous tensor types')
        count=self.sparse(len(rule.keys),len(incoming),int(cfg.visual_eligibility=='forecast-causal-v1'),self.threads,n.n,
            cfg.eta_prediction,cfg.eta_reward,math.exp(-n.config.dt/cfg.tau_eligibility),pair_decay,
            cfg.prune_epsilon,float(reward.clamp(-1,1)[0]),*(x.data_ptr() for x in pointers))
        rule.keys,rule.values,rule.arrival_trace=keys[:count],values[:count],traces[:count]
        overload=self.finish(rule,activity)
        if overload.any():
            rule.homeostatic_exponent.add_(overload[n.post],alpha=cfg.homeostasis_rate*n.config.dt)

"""Experimental bounded visual-loop fusion; weight visibility remains unchanged."""
import ctypes
from dataclasses import replace
from functools import partial
import math
import weakref
import torch
from .native_cpu import NativeCPU, _canonical


class DeferredCPU(NativeCPU):
    def __init__(self, library, threads=1, *, aggregate=False):
        super().__init__(library,threads)
        version=getattr(self.library,'deferred_abi_version',None)
        if version is None or version()!=2:
            raise ValueError('deferred experiment requires a rebuilt native library')
        self.pending=weakref.WeakKeyDictionary()
        self.aggregate=aggregate
        self.deferred=self.library.deferred_visual
        self.deferred.argtypes=[ctypes.c_int64]*6+[ctypes.c_float]*2+[ctypes.c_void_p]*12
        self.deferred.restype=ctypes.c_int64
        self.error_fn=self.library.deferred_error
        self.error_fn.argtypes=[ctypes.c_int64]*2+[ctypes.c_float]+[ctypes.c_void_p]*3
        self.error_fn.restype=None

    @staticmethod
    def check_config(rule):
        if (rule.config.visual_target!='input-arrivals-v1' or
                rule.config.visual_eligibility!='forecast-causal-v1'):
            raise ValueError('deferred experiment requires input-arrivals and forecast-causal learning')

    def enable(self, trainer):
        if trainer.plasticity is None:
            raise ValueError('deferred experiment requires online learning')
        self.check_config(trainer.plasticity)
        super().enable(trainer)
        trainer.manifest=dict(trainer.manifest,execution_experiment=(
            'deferred-geometric-v1' if self.aggregate else 'deferred-ticks-v1'))
        trainer._materialize=partial(self.materialize,trainer.plasticity)

    def prepare(self, rule, activity):
        observed=super().prepare(rule,activity)
        errors=torch.empty_like(observed)
        self.error_fn(rule.network.n,self.threads,rule.config.eta_prediction,
            observed.data_ptr(),rule.expected.data_ptr(),errors.data_ptr())
        self.pending[rule]['errors'].append(errors)
        return observed

    @torch.no_grad()
    def observe(self, rule, activity, reward):
        self.check_config(rule)
        n=rule.network
        _,paths,_=self.topology(n)
        _canonical(activity.arrival_edges,torch.int64,(len(activity.arrival_edges),))
        _canonical(activity.arrival_environments,torch.int64,activity.arrival_edges.shape)
        if (activity.arrival_environments.any() or
                ((activity.arrival_edges<0)|(activity.arrival_edges>=n.e)).any()):
            raise ValueError('native CPU arrivals must address valid B=1 edges')
        if rule in self.pending and len(self.pending[rule]['errors'])==8:
            self.materialize(rule,canonical=False)
        if rule not in self.pending:
            _canonical(rule.keys,torch.int64,(len(rule.keys),))
            for tensor in (rule.values,rule.arrival_trace):
                _canonical(tensor,torch.float32,rule.keys.shape)
            if (((rule.keys<0)|(rule.keys>=n.e)).any() or (rule.keys[1:]<=rule.keys[:-1]).any()):
                raise ValueError('eligibility keys must be sorted unique valid edges')
            visual=paths[rule.keys]==1
            self.pending[rule]=dict(keys=rule.keys[visual],values=rule.values[visual],
                traces=rule.arrival_trace[visual],errors=[],arrivals=[])
            rule.keys,rule.values,rule.arrival_trace=(x[~visual] for x in
                                                    (rule.keys,rule.values,rule.arrival_trace))
        visual=paths[activity.arrival_edges]==1
        behavior=paths[activity.arrival_edges]==2
        super().observe(rule,replace(activity,arrival_edges=activity.arrival_edges[behavior],
                                    arrival_environments=activity.arrival_environments[behavior]),reward)
        self.pending[rule]['arrivals'].append(activity.arrival_edges[visual])

    @torch.no_grad()
    def materialize(self, rule, canonical=True):
        state=self.pending.get(rule)
        if state is None:
            return
        steps=len(state['errors'])
        if steps:
            n=rule.network
            _canonical(rule.proposals,torch.float32,(n.e,))
            _canonical(n.current_decay,torch.float32,(n.n,))
            post,_,signs=self.topology(n)
            incoming=torch.cat([edges*steps+tick for tick,edges in enumerate(state['arrivals'])])
            errors=torch.stack(state['errors'],dim=-1).contiguous()
            size=len(state['keys'])+len(incoming)
            keys=state['keys'].new_empty(size)
            values=state['values'].new_empty(size)
            traces=state['traces'].new_empty(size)
            count=self.deferred(len(state['keys']),len(incoming),steps,n.n,self.threads,self.aggregate,
                math.exp(-n.config.dt/rule.config.tau_pair),rule.config.prune_epsilon,
                *(x.data_ptr() for x in (state['keys'],state['values'],state['traces'],incoming,
                    post,signs,n.current_decay,errors,rule.proposals,keys,values,traces)))
            state.update(keys=keys[:count],values=values[:count],traces=traces[:count],errors=[],arrivals=[])
        if not canonical:
            return
        keys,values,traces=state['keys'],state['values'],state['traces']
        if rule.keys.numel():
            keys=torch.cat((keys,rule.keys))
            order=keys.argsort()
            keys=keys[order]
            values=torch.cat((values,rule.values))[order]
            traces=torch.cat((traces,rule.arrival_trace))[order]
        rule.keys,rule.values,rule.arrival_trace=keys,values,traces
        del self.pending[rule]

    def synchronize(self, rule):
        self.materialize(rule,canonical=False)
        super().synchronize(rule)

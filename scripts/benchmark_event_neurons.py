"""Experimental lazy discrete neurons; numerical regrouping is not a default.

Exclusive state ownership and immutable configuration/silencing are required.
Materialize BEFORE external edits, then invalidate. Weight updates need neither:
weights affect future arrivals only. Public snapshots/save materialize via trainer.
"""
import ctypes
import math
import os
from pathlib import Path
import subprocess
import weakref

import numpy as np
import torch
from fly_connectome.native_cpu import NativeCPU


def build_event(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['g++','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fopenmp',
        *(['-static'] if os.name=='nt' else []),'-static-libgcc','-static-libstdc++',
        str(Path(__file__).with_name('native_event_neurons.cpp')),'-o',str(output)],check=True)


class EventCPU(NativeCPU):
    def __init__(self, library, event_library, threads=4, horizon=128):
        super().__init__(library,threads=threads)
        if type(horizon) is not int or not 1 <= horizon <= 1024:
            raise ValueError('horizon must be an integer in [1,1024]')
        self.horizon = horizon
        self.event_library = ctypes.CDLL(str(Path(event_library).resolve()))
        self.event_fn = self.event_library.event_step
        self.event_fn.argtypes = self.neural.argtypes+[ctypes.c_int64]*2+[ctypes.c_void_p]*7
        self.event_fn.restype = None
        self.materialize_fn = self.event_library.event_materialize
        self.materialize_fn.argtypes = [ctypes.c_int64]*3+[ctypes.c_void_p]*13
        self.materialize_fn.restype = None
        self.cache = weakref.WeakKeyDictionary()
        self.neural = self._neural

    def invalidate(self, n):
        self.topology(n)
        cfg = n.config
        decay = math.exp(-cfg.dt/cfg.tau_membrane)
        a = n.membrane_decay.numpy() if cfg.class_parameters else np.full(n.n,decay,dtype=np.float32)
        g = np.float32(1)-a if cfg.class_parameters else np.full(n.n,1-decay,dtype=np.float32)
        d = n.current_decay.numpy() if cfg.class_parameters else np.full(n.n,math.exp(-cfg.dt/cfg.tau_current),dtype=np.float32)
        s = np.full(n.n,math.exp(-cfg.dt/cfg.tau_sensory) if cfg.tau_sensory else 0.,dtype=np.float32)
        c = np.full(n.n,math.exp(-cfg.dt/cfg.tau_adaptation),dtype=np.float32)
        parameters, groups = np.unique(np.stack((a,d,s,c,g),axis=1),axis=0,return_inverse=True)
        if not np.isfinite(parameters).all() or (parameters<0).any() or (parameters>1).any():
            raise ValueError('event recurrence requires finite decay/gain factors in [0,1]')
        table = np.zeros((len(parameters),self.horizon+1,7),dtype=np.float64)
        table[:,0,:4] = 1.
        for k in range(1,self.horizon+1):
            previous, row = table[:,k-1], table[:,k]
            row[:,:4] = previous[:,:4]*parameters[:,:4]
            row[:,4] = parameters[:,0]*previous[:,4]+parameters[:,4]
            row[:,5] = parameters[:,0]*previous[:,5]+parameters[:,4]*row[:,1]
            row[:,6] = parameters[:,0]*previous[:,6]+parameters[:,4]*row[:,2]
        bounds = np.zeros((len(parameters),self.horizon+1,8),dtype=np.float64)
        terms = table[:,1:,[0,4,5,6]]
        bounds[:,1:,0::2] = np.minimum.accumulate(terms,axis=1)
        bounds[:,1:,1::2] = np.maximum.accumulate(terms,axis=1)
        self.cache[n] = dict(groups=torch.from_numpy(groups.astype(np.int32)),
            coefficients=torch.from_numpy(table),bounds=torch.from_numpy(bounds),
            last=torch.full((n.n,),n.step_index-1,dtype=torch.int64,device='cpu'),
            due=torch.full((n.n,),n.step_index,dtype=torch.int64,device='cpu'),
            awake=torch.zeros(n.n,dtype=torch.uint8,device='cpu'),
            counters=torch.zeros(4,dtype=torch.int64,device='cpu'))

    def _neural(self, *args):
        n = self.current_network
        state = self.cache[n]
        self.event_fn(*args,n.step_index,self.horizon,
            *(state[key].data_ptr() for key in ('groups','coefficients','bounds','last','due','awake','counters')))

    def step(self, n, sensory_current, **kwargs):
        if n not in self.cache:
            self.invalidate(n)
        self.current_network = n
        return super().step(n,sensory_current,**kwargs)

    def materialize(self, n):
        if n not in self.cache:
            return
        state = self.cache[n]
        self.materialize_fn(n.n,n.step_index-1,self.horizon,
            *(x.data_ptr() for x in (state['groups'],state['coefficients'],n.silenced,
                n.feedforward_current,n.predictive_current,n.behavioral_current,n.sensory_state,
                n.adaptation,n.voltage,n.refractory,n.rest_current,state['last'],state['due'])))

    def enable(self, trainer):
        super().enable(trainer)
        self.cache.pop(trainer.network,None)
        def materialize():
            self.materialize(trainer.network)
        trainer._materialize = materialize
        trainer.manifest['execution_experiment'] = 'event-neurons-v1'


if __name__ == '__main__':
    import argparse
    from dataclasses import replace
    import json
    import time
    from fly_connectome.data import checksum
    from fly_connectome.training import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--library',required=True)
    parser.add_argument('--event-library',required=True)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--frames',type=int,default=1000)
    parser.add_argument('--horizon',type=int,default=128)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('positive frame count required')
    if args.build:
        build_event(args.event_library)
    torch.set_num_threads(1)
    identity = checksum(args.checkpoint)
    sources = {str(path):checksum(path) for path in (
        Path(__file__),Path(__file__).with_name('native_event_neurons.cpp'),
        Path(args.library),Path(args.event_library))}
    # Reload for each run: bound native methods/caches must not be deep-copied.
    runs, reference_spikes, reference_state = [], [], {}
    for number,label in enumerate(('reference','event','event','reference')):
        model = load_checkpoint(args.checkpoint)
        model.config = replace(model.config,metrics_mode='events')
        NativeCPU(args.library,threads=4).enable(model)
        model.run(5)
        kernel = (NativeCPU(args.library,threads=4) if label=='reference' else
                  EventCPU(args.library,args.event_library,threads=4,horizon=args.horizon))
        kernel.enable(model)
        original = model.network.step
        tick = changed_ticks = changed_spikes = 0
        first_divergence = None
        def record(*a,**kw):
            global tick,changed_ticks,changed_spikes,first_divergence
            result = original(*a,**kw)
            packed = np.packbits(result.spikes.numpy()).tobytes()
            if number == 0:
                reference_spikes.append(packed)
            elif packed != reference_spikes[tick]:
                if first_divergence is None:
                    first_divergence = tick
                changed_ticks += 1
                changed_spikes += sum((x^y).bit_count() for x,y in zip(packed,reference_spikes[tick]))
            tick += 1
            return result
        model.network.step = record
        start = time.perf_counter()
        model.run(args.frames)
        if label == 'event':
            kernel.materialize(model.network)
        elapsed = time.perf_counter()-start
        differences = {}
        for owner in ('network','plasticity'):
            for name,value in vars(getattr(model,owner)).items():
                if not isinstance(value,torch.Tensor):
                    continue
                key = owner+'.'+name
                if number == 0:
                    reference_state[key] = value.clone()
                elif not torch.equal(value,reference_state[key]):
                    ref = reference_state[key]
                    if ref.shape != value.shape:
                        differences[key] = dict(reference_shape=list(ref.shape),shape=list(value.shape))
                    else:
                        delta = (value.double()-ref.double()).abs().flatten()
                        differences[key] = dict(changed=int((delta!=0).sum()),maximum=float(delta.max()),
                            mean=float(delta.mean()),p99=float(torch.quantile(delta,.99)))
        result = dict(backend=label,seconds=elapsed,fps=args.frames/elapsed,
            different_spike_ticks=changed_ticks,different_spikes=changed_spikes,
            first_divergent_tick=first_divergence,state_differences=differences,
            counters=kernel.cache[model.network]['counters'].tolist() if label=='event' else None,
            deferred_tick_fraction=(1-int(kernel.cache[model.network]['counters'][0])/
                (tick*model.network.n) if label=='event' else None))
        runs.append(result)
        print(json.dumps(result),flush=True)
    assert checksum(args.checkpoint) == identity
    assert all(checksum(path)==digest for path,digest in sources.items())
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,sources=sources,
        frames=args.frames,horizon=args.horizon,warmup_frames=5,native_threads=4,
        torch_threads=1,runs=runs),indent=2)+'\n',encoding='utf-8')

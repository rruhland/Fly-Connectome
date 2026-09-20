"""Balanced deferred/reference online runs, with state and spike-time auditing."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from benchmark_sparse_merge import tensor_hash
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.deferred_cpu import DeferredCPU


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--library',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',required=True)
    p.add_argument('--aggregate',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    identity=checksum(args.checkpoint)
    base=load_checkpoint(args.checkpoint)
    base.config=replace(base.config,metrics_mode='events')
    base.run(5)
    runs=[];reference_spikes=[];reference_state={}
    for number,label in enumerate(('reference','deferred','deferred','reference')):
        model=copy.deepcopy(base)
        kernel=(DeferredCPU(args.library,threads=args.threads,aggregate=args.aggregate)
                if label=='deferred' else NativeCPU(args.library,threads=args.threads))
        kernel.enable(model)
        original=model.network.step
        tick=0;different_ticks=0;different_spikes=0
        def record(*a,**kw):
            global tick,different_ticks,different_spikes
            result=original(*a,**kw)
            packed=np.packbits(result.spikes.numpy()).tobytes()
            if number==0:
                reference_spikes.append(packed)
            elif packed!=reference_spikes[tick]:
                different_ticks+=1
                different_spikes+=sum((a^b).bit_count() for a,b in zip(packed,reference_spikes[tick]))
            tick+=1
            return result
        model.network.step=record
        start,cpu=time.perf_counter(),time.process_time()
        model.run(args.steps)
        if label=='deferred':
            kernel.materialize(model.plasticity)
        elapsed,cpu=time.perf_counter()-start,time.process_time()-cpu
        differences={}
        for owner in ('network','plasticity'):
            for name,value in vars(getattr(model,owner)).items():
                if not isinstance(value,torch.Tensor):
                    continue
                key=owner+'.'+name
                if number==0:
                    reference_state[key]=value.clone()
                elif not torch.equal(value,reference_state[key]):
                    ref=reference_state[key]
                    if ref.shape!=value.shape:
                        differences[key]=dict(reference_shape=list(ref.shape),shape=list(value.shape))
                    else:
                        delta=(value.double()-ref.double()).abs().flatten()
                        differences[key]=dict(changed=int((delta!=0).sum()),maximum=float(delta.max()),
                            mean=float(delta.mean()),p99=float(torch.quantile(delta,.99)))
        runs.append(dict(backend=label,seconds=elapsed,cpu_seconds=cpu,fps=args.steps/elapsed,
            state_sha256=tensor_hash(model),different_spike_ticks=different_ticks,
            different_spikes=different_spikes,state_differences=differences,event_metrics=model.metrics))
        print(json.dumps(runs[-1]),flush=True)
        del model,kernel
    exact=len({run['state_sha256'] for run in runs})==1 and all(r['different_spikes']==0 for r in runs)
    assert checksum(args.checkpoint)==identity
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,steps=args.steps,
        warmup_frames=5,native_threads=args.threads,metrics_mode='events',aggregate=args.aggregate,
        exact=exact,runs=runs),indent=2)+'\n')
    if not args.aggregate:
        assert exact,'Deferred trajectory diverged: inspect recorded state and spike-time differences'

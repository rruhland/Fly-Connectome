"""Balanced exact-state comparison of online CPU learning against the tensor reference."""
import argparse
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import torch
from benchmark_sparse_merge import tensor_hash
from fly_connectome.data import checksum
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.training import load_checkpoint

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--library',required=True)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    identity=checksum(args.checkpoint)
    base=load_checkpoint(args.checkpoint)
    base.run(5)
    kernel=NativeCPU(args.library)
    runs=[]
    for label in ('reference','native','native','reference'):
        model=copy.deepcopy(base)
        if label=='native':
            kernel.enable(model)
        digest=hashlib.sha256()
        original=model.network.step
        def record(*a,**kw):
            result=original(*a,**kw)
            digest.update(result.spikes.numpy().tobytes())
            return result
        model.network.step=record
        start,cpu=time.perf_counter(),time.process_time()
        model.run(args.steps)
        wall,cpu=time.perf_counter()-start,time.process_time()-cpu
        runs.append(dict(backend=label,seconds=wall,cpu_seconds=cpu,fps=args.steps/wall,
                         spikes_sha256=digest.hexdigest(),state_sha256=tensor_hash(model)))
        print(json.dumps(runs[-1]),flush=True)
        del model
    exact=len({r['state_sha256'] for r in runs})==1 and len({r['spikes_sha256'] for r in runs})==1
    assert checksum(args.checkpoint)==identity
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,steps=args.steps,
        threads=1,warmup_frames=5,exact=exact,runs=runs),indent=2)+'\n')
    assert exact,'Native trajectory differs; investigate before adoption'

"""Controlled native thread sweep on identical online trajectories, optional CPU affinity."""
import argparse
import copy
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from benchmark_sparse_merge import tensor_hash
from fly_connectome.training import load_checkpoint
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.data import checksum

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--library',required=True)
    p.add_argument('--cpu-mask',type=lambda v:int(v,0))
    p.add_argument('--steps',type=int,default=100);p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.cpu_mask is not None:
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        if not k.SetProcessAffinityMask(k.GetCurrentProcess(),args.cpu_mask):
            raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1)
    identity=checksum(args.checkpoint)
    base=load_checkpoint(args.checkpoint)
    base.config=replace(base.config,metrics_mode='events')
    runs=[]
    for threads in (1,2,4,6,6,4,2,1):
        model=copy.deepcopy(base)
        kernel=NativeCPU(args.library,threads=threads)
        kernel.enable(model)
        model.run(40)
        start,cpu=time.perf_counter(),time.process_time()
        model.run(args.steps)
        seconds,cpu=time.perf_counter()-start,time.process_time()-cpu
        runs.append(dict(threads=threads,seconds=seconds,cpu_seconds=cpu,
                         fps=args.steps/seconds,state_sha256=tensor_hash(model)))
        print(json.dumps(runs[-1]),flush=True)
        del model,kernel
    assert len({r['state_sha256'] for r in runs})==1
    assert checksum(args.checkpoint)==identity
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,steps=args.steps,
        warmup_frames=40,cpu_mask=args.cpu_mask,metrics_mode='events',runs=runs),indent=2)+'\n')

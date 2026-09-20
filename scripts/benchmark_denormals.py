"""Separate-process CPU denormal experiment; never changes source checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    before=checksum(args.checkpoint)
    base=load_checkpoint(args.checkpoint)
    subnormals={name:int(((value.abs()>0)&(value.abs()<torch.finfo(value.dtype).tiny)).sum())
                for name,value in vars(base.network).items() if isinstance(value,torch.Tensor) and value.is_floating_point()}
    runs=[]
    reference=None
    for flush in (False,True,True,False):
        model=copy.deepcopy(base)
        spikes=[]
        original=model.network.step
        def record(*args,**kwargs):
            activity=original(*args,**kwargs)
            spikes.append(activity.spikes.clone())
            return activity
        model.network.step=record
        supported=torch.set_flush_denormal(flush)
        started=time.perf_counter()
        model.run(args.steps)
        elapsed=time.perf_counter()-started
        torch.set_flush_denormal(False)
        spike_trace=torch.stack(spikes)
        state={name:value.clone() for name,value in vars(model.network).items() if isinstance(value,torch.Tensor)}
        if reference is None:
            reference=(spike_trace,state)
        differences={name:float((value-reference[1][name]).abs().max()) for name,value in state.items()
                     if value.is_floating_point() and value.numel()}
        runs.append(dict(flush=flush,supported=supported,seconds=elapsed,
            identical_spikes=torch.equal(spike_trace,reference[0]),maximum_state_differences=differences))
        print(json.dumps(runs[-1]),flush=True)
        del model,spikes,state
    assert checksum(args.checkpoint)==before
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=before,steps=args.steps,
        threads=1,initial_subnormal_counts=subnormals,runs=runs,
        scope='Short training-trajectory experiment with spike recording overhead, not a long-run equivalence claim or canonical backend change.'),indent=2)+'\n')

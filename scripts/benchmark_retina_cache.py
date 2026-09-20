"""Balanced fixed-projection cache audit with complete online training."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import torch
from benchmark_sleeping import tensor_hash
from fly_connectome.data import checksum
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.training import load_checkpoint


@torch.no_grad()
def reference_project(self,events):
    batch = len(events.offsets)-1
    bins = torch.zeros(batch*2*self.n_columns,dtype=torch.float32,device=self.device)
    indices = (events.environments*2+events.on.long())*self.n_columns+self.pixel_bins[events.pixels]
    bins.index_fill_(0,indices,1.)
    bins = bins.view(batch,2,self.n_columns)
    result = torch.zeros(batch,len(self.neuron_columns),device=self.device)
    result[:,self.injected] = bins[:,self.polarity[self.injected],self.neuron_columns[self.injected]]
    result[:,self.contrast] = bins[:,0,self.neuron_columns[self.contrast]]-bins[:,1,self.neuron_columns[self.contrast]]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--library',required=True)
    parser.add_argument('--frames',type=int,default=1000)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    identity = checksum(args.checkpoint)
    base = load_checkpoint(args.checkpoint)
    base.config = replace(base.config,metrics_mode='events')
    runs = []
    for mode in ('reference','cached','cached','reference'):
        model = copy.deepcopy(base)
        if mode == 'reference':
            model.retina.project = MethodType(reference_project,model.retina)
        NativeCPU(args.library,threads=4).enable(model)
        model.run(5)
        spikes = hashlib.sha256()
        original = model.network.step
        def record(*a,**kw):
            activity = original(*a,**kw)
            spikes.update(activity.spikes.numpy().tobytes())
            return activity
        model.network.step = record
        start = time.perf_counter()
        model.run(args.frames)
        elapsed = time.perf_counter()-start
        runs.append(dict(mode=mode,seconds=elapsed,fps=args.frames/elapsed,
            spikes_sha256=spikes.hexdigest(),state_sha256=tensor_hash(model)))
        print(json.dumps(runs[-1]),flush=True)
    exact = len({r['state_sha256'] for r in runs}) == len({r['spikes_sha256'] for r in runs}) == 1
    assert checksum(args.checkpoint) == identity
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,frames=args.frames,
        warmup_frames=5,torch_threads=1,native_threads=4,exact=exact,runs=runs),indent=2)+'\n',encoding='utf-8')
    assert exact

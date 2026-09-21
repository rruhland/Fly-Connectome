"""Two bounded exact experiments: reusable native scratch and arrival filtering."""
import ctypes
import os
from pathlib import Path
import subprocess

import torch
from fly_connectome.native_cpu import NativeCPU


def build_overhead(output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    source=(Path(__file__).parents[1]/'src/fly_connectome/native_cpu.cpp').read_text(encoding='utf-8')
    # Explicit checked transformations keep the experiment tied to the reference
    # arithmetic without maintaining a second copy of every native kernel.
    substitutions={
        'std::vector<float> errors(n);': '''thread_local std::vector<float> error_storage;
    error_storage.resize(n);
    float* errors=error_storage.data();''',
        'std::vector<std::int64_t> offsets(chunks),counts(chunks);': '''thread_local std::vector<std::int64_t> offset_storage,count_storage;
    offset_storage.resize(chunks);count_storage.resize(chunks);
    auto* offsets=offset_storage.data();auto* counts=count_storage.data();'''}
    # Only sparse_observe is changed; deferred kernels retain their reference code.
    end=source.index('extern "C" EXPORT void neural_step')
    prefix=source[:end]
    for old,new in substitutions.items():
        if prefix.count(old)!=1:
            raise ValueError('reference kernel changed; review experiment transformation')
        prefix=prefix.replace(old,new)
    source=prefix+source[end:]+'''
extern "C" EXPORT std::int64_t filter_learning_arrivals(std::int64_t count,
    const std::int64_t* edges, const std::uint8_t* paths, std::int64_t* output) {
    std::int64_t kept=0;
    for (std::int64_t i=0;i<count;++i)
        if (paths[edges[i]]!=0) output[kept++]=edges[i];
    return kept;
}
'''
    generated=output.with_suffix('.cpp')
    generated.write_text(source,encoding='utf-8')
    subprocess.run(['g++','-O3','-fno-fast-math','-ffp-contract=off','-shared','-fopenmp',
        *(['-static'] if os.name=='nt' else []),'-static-libgcc','-static-libstdc++',
        str(generated),'-o',str(output)],check=True)


class OverheadCPU(NativeCPU):
    def __init__(self,library,threads=4):
        super().__init__(library,threads=threads)
        self.filter_fn=self.library.filter_learning_arrivals
        self.filter_fn.argtypes=[ctypes.c_int64]+[ctypes.c_void_p]*3
        self.filter_fn.restype=ctypes.c_int64

    def learning_arrivals(self,edges,pathways):
        # NativeCPU.observe already validates canonical pointers and edge bounds.
        output=torch.empty_like(edges)
        count=self.filter_fn(len(edges),edges.data_ptr(),pathways.data_ptr(),output.data_ptr())
        return output[:count]


if __name__=='__main__':
    import argparse
    import copy
    from dataclasses import replace
    import hashlib
    import json
    import time
    from benchmark_sleeping import tensor_hash
    from fly_connectome.data import checksum
    from fly_connectome.training import load_checkpoint

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--reference-library',required=True)
    parser.add_argument('--library',required=True)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--frames',type=int,default=1000)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.frames<1:
        parser.error('positive frame count required')
    if args.build:
        build_overhead(args.library)
    torch.set_num_threads(1)
    identity=checksum(args.checkpoint)
    sources={str(p):checksum(p) for p in (Path(__file__),
        Path(__file__).parents[1]/'src/fly_connectome/native_cpu.py',
        Path(__file__).parents[1]/'src/fly_connectome/native_cpu.cpp',
        Path(args.reference_library),Path(args.library))}
    base=load_checkpoint(args.checkpoint)
    base.config=replace(base.config,metrics_mode='events')
    runs=[]
    for label in ('reference','scratch','filtered','filtered','scratch','reference'):
        model=copy.deepcopy(base)
        kernel=(OverheadCPU(args.library) if label=='filtered' else
                NativeCPU(args.reference_library if label=='reference' else args.library,threads=4))
        kernel.enable(model)
        model.run(5)
        original=model.network.step
        digest=hashlib.sha256()
        def record(*a,**kw):
            result=original(*a,**kw)
            digest.update(result.spikes.numpy().tobytes())
            return result
        model.network.step=record
        start=time.perf_counter()
        model.run(args.frames)
        elapsed=time.perf_counter()-start
        runs.append(dict(mode=label,seconds=elapsed,fps=args.frames/elapsed,
            spikes_sha256=digest.hexdigest(),state_sha256=tensor_hash(model)))
        print(json.dumps(runs[-1]),flush=True)
    exact=len({r['state_sha256'] for r in runs})==len({r['spikes_sha256'] for r in runs})==1
    assert checksum(args.checkpoint)==identity
    assert all(checksum(p)==digest for p,digest in sources.items())
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity,sources=sources,
        frames=args.frames,warmup_frames=5,torch_threads=1,native_threads=4,
        exact=exact,runs=runs),indent=2)+'\n',encoding='utf-8')
    assert exact,'Experiment diverged: retain the reference'

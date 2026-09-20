"""Compare original sparse merge with optimized merge on an identical resumed run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
import fly_connectome.plasticity as plasticity
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint


def reference_merge(old_keys, old_values, old_traces, arrival_keys):
    keys = torch.unique(torch.cat((old_keys, arrival_keys)), sorted=True)
    values = old_values.new_zeros(len(keys))
    values[torch.searchsorted(keys, old_keys)] = old_values
    traces = old_traces.new_zeros(len(keys))
    traces[torch.searchsorted(keys, old_keys)] = old_traces
    arrivals = old_traces.new_zeros(len(keys))
    arrivals.index_add_(0, torch.searchsorted(keys, arrival_keys), old_traces.new_ones(len(arrival_keys)))
    traces.add_(arrivals)
    return keys, values, traces, arrivals


def tensor_hash(model):
    digest = hashlib.sha256()
    for obj in (model, model.network, model.plasticity, model.environment, model.environment.body, model.camera):
        for name, value in sorted(vars(obj).items()):
            if isinstance(value, torch.Tensor):
                digest.update(name.encode())
                digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output', required=True)
    parser.add_argument('--native-library', help='optional separately compiled experimental CPU merge')
    args = parser.parse_args()
    torch.set_num_threads(1)
    identity = checksum(args.checkpoint)
    base = load_checkpoint(args.checkpoint)
    base.run(5)
    tensor_merge = plasticity._merge_arrivals
    if args.native_library:
        from native_sparse_merge import load_merge
        optimized = load_merge(args.native_library)
        reference_merge = tensor_merge
    else:
        optimized = tensor_merge
    runs = []
    for label, merge in (('reference', reference_merge), ('optimized', optimized),
                         ('optimized', optimized), ('reference', reference_merge)):
        model = copy.deepcopy(base)
        plasticity._merge_arrivals = merge
        spike_hash = hashlib.sha256()
        original = model.network.step
        def record(*a, **kw):
            result = original(*a, **kw)
            spike_hash.update(result.spikes.numpy().tobytes())
            return result
        model.network.step = record
        start, cpu = time.perf_counter(), time.process_time()
        model.run(args.steps)
        wall, cpu = time.perf_counter()-start, time.process_time()-cpu
        runs.append(dict(implementation=label, seconds=wall, process_cpu_seconds=cpu,
                         frames_per_second=args.steps/wall, spikes_sha256=spike_hash.hexdigest(),
                         state_sha256=tensor_hash(model)))
        print(json.dumps(runs[-1]), flush=True)
        del model
    plasticity._merge_arrivals = tensor_merge
    assert len({run['spikes_sha256'] for run in runs}) == 1
    assert len({run['state_sha256'] for run in runs}) == 1
    assert checksum(args.checkpoint) == identity
    Path(args.output).write_text(json.dumps(dict(checkpoint_sha256=identity, steps=args.steps,
        warmup_frames=5, threads=1, runs=runs, exact_trajectory_and_state=True,
        native_library=args.native_library), indent=2)+'\n')

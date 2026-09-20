"""Isolated exact fixed-point sleeping experiment; never changes production defaults.

SleepingCPU requires exclusive ownership of dynamic state and immutable config,
topology and silencing between calls. Call invalidate(network) after ANY external
state/parameter/silencing edit. Synaptic weight updates are allowed: arrivals wake
their targets before integration. Cache state is never checkpointed.
"""
import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import weakref

import torch
from fly_connectome.data import checksum
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.training import load_checkpoint


def build_sleeping(output, compiler='g++'):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, '-O3', '-fno-fast-math', '-ffp-contract=off', '-shared', '-fopenmp',
                    *(['-static'] if os.name == 'nt' else []), '-static-libgcc', '-static-libstdc++',
                    str(Path(__file__).with_name('native_sleeping.cpp')), '-o', str(output)], check=True)


class SleepingCPU(NativeCPU):
    def __init__(self, library, sleeping_library, threads=4):
        super().__init__(library, threads=threads)
        self.sleeping_library = ctypes.CDLL(str(Path(sleeping_library).resolve()))
        self.sleeping_fn = self.sleeping_library.sleeping_step
        self.sleeping_fn.argtypes = self.neural.argtypes + [ctypes.c_void_p]
        self.sleeping_fn.restype = None
        self.flags = weakref.WeakKeyDictionary()
        self.neural = self._neural

    def invalidate(self, network):
        self.flags[network] = torch.zeros(network.n, dtype=torch.uint8, device='cpu')

    def _neural(self, *args):
        self.sleeping_fn(*args, self.current_flags.data_ptr())

    def step(self, network, sensory_current, **kwargs):
        if network not in self.flags:
            self.invalidate(network)
        self.current_flags = self.flags[network]
        return super().step(network, sensory_current, **kwargs)


def tensor_hash(model):
    # Same complete tensor scope as benchmark_sparse_merge.tensor_hash.
    digest = hashlib.sha256()
    for obj in (model, model.network, model.plasticity, model.environment,
                model.environment.body, model.camera):
        for name, value in sorted(vars(obj).items()):
            if isinstance(value, torch.Tensor):
                digest.update(name.encode())
                digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def benchmark(checkpoint, library, sleeping_library, frames, repeats):
    torch.set_num_threads(1)
    identity = checksum(checkpoint)
    sources = {str(path): checksum(path) for path in (
        Path(__file__), Path(__file__).with_name('native_sleeping.cpp'),
        Path(__file__).parents[1]/'src/fly_connectome/native_cpu.cpp',
        Path(__file__).parents[1]/'src/fly_connectome/native_cpu.py')}
    libraries = {str(path): checksum(path) for path in (library, sleeping_library)}
    runs = []
    for repeat in range(repeats):
        for mode in (('dense', 'sleeping') if repeat % 2 == 0 else ('sleeping', 'dense')):
            model = load_checkpoint(checkpoint)
            model.config = replace(model.config, metrics_mode='events')
            kernel = (NativeCPU(library, threads=4) if mode == 'dense' else
                      SleepingCPU(library, sleeping_library, threads=4))
            kernel.enable(model)
            spikes = hashlib.sha256()
            original_step = model.network.step

            def record(*args, **kwargs):
                activity = original_step(*args, **kwargs)
                spikes.update(activity.spikes.numpy().tobytes())
                return activity

            model.network.step = record
            model.run(5)
            neural_seconds = 0.
            original = kernel.neural

            def timed(*args):
                nonlocal neural_seconds
                start = time.perf_counter()
                original(*args)
                neural_seconds += time.perf_counter() - start

            kernel.neural = timed
            start = time.perf_counter()
            model.run(frames)
            wall = time.perf_counter() - start
            runs.append(dict(mode=mode, repeat=repeat, frames=frames, seconds=wall,
                             neural_seconds=neural_seconds, frames_per_second=frames/wall,
                             spikes_sha256=spikes.hexdigest(), state_sha256=tensor_hash(model),
                             sleeping_at_end=(int((kernel.flags[model.network] == 1).sum())
                                              if mode == 'sleeping' else None)))
            print(json.dumps(runs[-1]), flush=True)
    after = checksum(checkpoint)
    assert identity == after
    assert len({run['spikes_sha256'] for run in runs}) == 1
    assert len({run['state_sha256'] for run in runs}) == 1
    assert all(checksum(path) == identity for path, identity in sources.items())
    assert all(checksum(path) == identity for path, identity in libraries.items())
    return dict(checkpoint_sha256_before=identity, checkpoint_sha256_after=after,
                source_sha256=sources, library_sha256=libraries,
                exact_trajectory_and_state=True, spike_hash_includes_warmup=True,
                torch_threads=1, native_threads=4, warmup_frames=5, runs=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--native-library', required=True)
    parser.add_argument('--sleeping-library', required=True)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error('positive frames and repeats required')
    if args.build:
        build_sleeping(args.sleeping_library)
    report = benchmark(args.checkpoint, args.native_library, args.sleeping_library,
                       args.frames, args.repeats)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')

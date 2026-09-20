"""Bounded headless CPU profile; source checkpoint is never modified."""
import argparse
import json
from pathlib import Path
import time
from collections import defaultdict
import torch
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    identity = checksum(args.checkpoint)
    model = load_checkpoint(args.checkpoint)
    model.run(5)
    timings = defaultdict(float)
    originals = []
    for obj, method, label in ((model.network, 'step', 'network'),
                               (model.network, '_arrivals', 'arrivals_nested'),
                               (model.plasticity, 'observe', 'learning'),
                               (model.plasticity, 'synchronize', 'synchronize')):
        original = getattr(obj, method)
        originals.append((obj, method, original))
        def timed(*a, _fn=original, _label=label, **kw):
            start = time.perf_counter()
            result = _fn(*a, **kw)
            timings[_label] += time.perf_counter() - start
            return result
        setattr(obj, method, timed)
    start, cpu = time.perf_counter(), time.process_time()
    model.run(args.steps)
    wall, cpu = time.perf_counter()-start, time.process_time()-cpu
    for obj, method, original in originals:
        setattr(obj, method, original)
    report = dict(checkpoint_sha256=identity, steps=args.steps, threads=1,
                  seconds=wall, process_cpu_seconds=cpu, frames_per_second=args.steps/wall,
                  phase_seconds=dict(timings), active_eligibilities=model.plasticity.keys.numel())
    print(json.dumps(report), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        model.run(args.profile_steps)
    report['operator_profile_steps'] = args.profile_steps
    report['operators'] = [dict(name=x.key, self_cpu_ms=x.self_cpu_time_total/1000,
                               calls=x.count) for x in sorted(prof.key_averages(),
                               key=lambda x:x.self_cpu_time_total, reverse=True)[:25]]
    assert checksum(args.checkpoint) == identity
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['operators']), flush=True)

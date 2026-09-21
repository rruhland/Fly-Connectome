# Bounded exact overhead experiments: stop throughput work here

The user approved further exact software/memory optimizations but limited the
pass to a couple of experiments before returning to M1A. We tested two small
prerequisites for a persistent compiled runner, without implementing that larger
rewrite:

1. Reuse per-calling-thread native error/partition scratch arrays. Every element
   used by workers is overwritten before use. The worker loops access ordinary
   pointers to the caller's storage, not separate uninitialized thread-local arrays.
2. Add stable native filtering of predictive/behavioral arrival edges, avoiding
   tensor indexing, nonzero construction and a redundant clone. The sparse local
   update, sorting, duplicate accumulation and numerical operation order stay fixed.

All six 1000-frame online runs have identical full-state and spike hashes.
Tests cover mixed local rules, duplicate/empty arrivals, graph-size changes,
parallel partitions, real nonzero subnormal updates and checkpoint resume.
No biological parameters, time resolution, signs, topology, local objective,
PyTorch rate arithmetic or scientific checkpoints changed.

| Execution | Seconds | Frames/s |
| --- | ---: | ---: |
| Reference | 30.04 | 33.29 |
| Reused scratch | 31.11 | 32.14 |
| Scratch + native filter | 87.29 | 11.46 |
| Scratch + native filter | 32.83 | 30.46 |
| Reused scratch | 41.93 | 23.85 |
| Reference | 66.95 | 14.94 |

**Performance is inconclusive.** The first filtered sample overlapped a short
test process and must not be used for a speed estimate. Even the isolated reference
samples vary by more than 2x, so pooling these wall times would give a misleading
speedup. Exactness evidence is valid; no throughput benefit is established.
Neither candidate is adopted. This is not evidence that scratch reuse or native
filtering must always be slower. No more performance experiments are scheduled
in this pass, per the user's explicit scope limit.

The existing exact backend and prior approximately 49fps measurements remain the
continuation path, not a speed guarantee under current host load. The 120fps goal
is unmet. Returning to M1A is an explicit prioritization, not a claim that all
model-preserving optimizations have been mathematically exhausted.

Raw record: `2026-09-20-overhead-1000.json`, including source/library/checkpoint
hashes. Reproduction (run without concurrent workloads):

```powershell
.venv/Scripts/python scripts/benchmark_overhead_cpu.py checkpoints/event-v1-combined-rate-10000.pt --reference-library runs/native_cpu_deferred.dll --library runs/native_overhead.dll --build --frames 1000 --output runs/overhead-repeat.json
```

Experiment code remains isolated in the script; the only native Python change
extracts the unchanged learning-arrival expression into an overridable method.
Default behavior remains unchanged. The full compiled-frame rewrite is deferred.

# Staged SIMD eligibility experiment: rejected

The isolated ABI-8 replacement in `scripts/native_vector_sparse.cpp` retains
the original sorted merge and duplicate counting, stages candidates into
visual/behavioral and existing/new groups, performs independent lane arithmetic,
then scatters proposals and stably compacts surviving eligibility state.
Every float32 tick, causal score-before-decay update, pairing operation, pruning
decision and canonical edge order remains unchanged. Production C++ is included
as the reference; the experimental override is not part of normal builds.

GCC 13.2 reports the arithmetic loop at line 33 vectorized with 16-byte and
8-byte vectors. Fast-math and floating-point contraction remain disabled.
Vectorization alone did not improve complete online learning:

| Balanced 1000-frame run | Frames/s |
| --- | ---: |
| Existing native | 47.55 |
| Staged SIMD | 32.67 |
| Staged SIMD | 32.03 |
| Existing native | 50.98 |

The staged version takes about 52% more elapsed time in the pooled comparison.
Every spike hash, complete model tensor hash and event statistic agrees across
all four runs. Eight focused regressions cover causal/legacy learning, parallel
partitions, duplicate arrivals, pruning, subnormal updates and checkpoint resume.
The source scientific checkpoint is unchanged.

Staging gathers many fields, performs repeated capacity checks, adds memory
passes and scatters outputs. Those costs outweigh the vector arithmetic benefit
in this implementation. This rejects this implementation, not every possible
vectorization strategy. Retain the existing fused sparse kernel.

```powershell
g++ -O3 -fno-fast-math -ffp-contract=off -shared -fopenmp -static -static-libgcc -static-libstdc++ scripts/native_vector_sparse.cpp -o runs/vector_sparse.dll
.venv/Scripts/python scripts/benchmark_native_cpu.py checkpoints/event-v1-combined-rate-10000.pt --library runs/vector_sparse.dll --reference-library runs/native_cpu_deferred.dll --steps 1000 --native-threads 4 --metrics events --output runs/vector-sparse-repeat.json
```

Raw results: `2026-09-20-vector-sparse-1000.json`. Optional compiler evidence:
add `-fopt-info-vec-all=runs/vector-sparse-codegen.txt` to the build command.

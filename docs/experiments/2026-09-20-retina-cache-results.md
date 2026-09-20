# Fixed retinal projection cache

The fixed injected/contrast roster and column/polarity selections are now
resolved once during Retina construction. Projection still uses the same binary
column bins, polarity rules, contrast subtraction and output assignment order.
This removes repeated boolean-mask scans without changing neural activity.

Balanced 1000-frame online runs, with Torch one/native four threads:

| Run | Frames/s | Full state and spike hashes |
| --- | ---: | --- |
| Original masks | 45.18 | identical |
| Cached indices | 49.79 | identical |
| Cached indices | 46.74 | identical |
| Original masks | 47.48 | identical |

Pooled throughput is 48.22 versus 46.30 frames/s, about 4.1% higher. Host timing
varies, so this is a modest observed benefit, not a guarantee. Every recorded
spike and complete model tensor hash agrees across all four runs. The source
checkpoint is unchanged. The 120fps goal remains unmet.

Tests cover batches, empty events, same-column collisions, mixed ON/OFF/contrast
mapping, uninjected neurons with missing columns and checkpoint reconstruction.
Caches are derived from immutable retinal anatomy; checkpoint schema is unchanged.

Raw data: `2026-09-20-retina-cache-1000.json`. Reproduce with:

```powershell
.venv/Scripts/python scripts/benchmark_retina_cache.py checkpoints/event-v1-combined-rate-10000.pt --library runs/native_cpu_deferred.dll --frames 1000 --output runs/retina-cache-repeat.json
```

The next exact experiment separates eligibility arithmetic from sparse merging
and stable compaction. It must preserve duplicate counting, causal update order,
behavioral pairing, pruning and every float32 tick. Extra memory traffic may
outweigh vectorization, so adoption requires a whole-training comparison.

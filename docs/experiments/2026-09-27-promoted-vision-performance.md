# Promoted M1A.5 vision performance pass

## Scope and result

This pass optimizes the approved `load_default()` visual service with
`step(..., learn=True)` and `VisualStateEncoder.encode`. It preserves the
grayscale-plus-event sensor, all structured outputs, online credit, append-all
forecast memory, and checkpoint version 3. The older measured-graph Pong trainer
is outside the user-selected scope. The production visual service is not yet
connected to Pong physics or motor control, so these are visual-service sample
rates, not game frame rates.

On alternating same-host CPU runs, the optimized median is **14.35 ms/sample
(69.7 samples/s)** for two visible entities and **41.83 ms/sample (23.9
samples/s)** for eight. The two-entity median exceeds the 50 Hz target; the
eight-entity case does not. Neither case demonstrates sustained 120 Hz.

## Measurement

Each process used the existing `scripts/benchmark_visual_interface.py` workload:
64x64 every-sample frames plus events, one PyTorch CPU thread, online learning,
full mixture and interval output, and sparse transport encoding. Each arm used
40 warmup and 200 measured samples. Original and optimized source paths were
verified separately, and process order was original, optimized, optimized,
original. Camera acquisition, rendering, and Pong simulation were excluded.
Host background load was not controlled; paired runs limit its effect on the
comparison but do not establish a hardware-independent latency guarantee.

| Run | Two entities p50 / p95 ms | Eight entities p50 / p95 ms |
|---|---:|---:|
| Original 1 | 34.25 / 53.58 | 143.91 / 196.94 |
| Optimized 1 | 13.64 / 19.44 | 41.75 / 68.05 |
| Optimized 2 | 15.05 / 22.33 | 41.90 / 68.94 |
| Original 2 | 34.64 / 49.71 | 142.03 / 207.05 |

The mean of the paired p50 measurements improves **2.40x** for two entities
and **3.42x** for eight. The eight-entity encoder portion measured 12.27/12.20
ms in the original runs and 7.59/7.60 ms in the optimized runs. An earlier
unpaired clean-worktree baseline was 21.62/85.64 ms; absolute CPU timings
varied substantially across the session, so the alternating comparison is the
primary result. A separate shorter 1/2/4-thread sweep gave eight-entity p50s
of 45.14/44.80/43.87 ms; this did not justify changing the single-thread
reference setting.

## Changes and parity

- The 32-iteration marginal-interval search now runs independent mixtures in
  batches grouped by component count. It uses the original scalar PyTorch
  operations through nested `vmap`. Issued mixture centers/weights and
  per-horizon calibration summaries are reused within each sample.
- Pending online outcomes are appended once per horizon to the short and long
  context banks. Every original example, insertion order, PIT calibration rank,
  and `seen` count is preserved. Other dynamics retain sequential updates.
- Context-key variance is cached until its key tensor changes or is modified
  in place. The sparse transport encoder clones ordinary tensor fields without
  generic tensor deep copy and orders all mixture components through a stable
  lexicographic sort. Structured outputs remain independent of their inputs.

The original checkout and optimized worktree produced **identical SHA-256
digests for every full state and encoder output** over 82 online samples,
including three unavailable camera frames. Their saved learned checkpoints
matched, as did 22 samples after loading into a fresh scene and 30 frozen
evaluation samples with outages. The two trajectory-result JSON files had the
same SHA-256:
`221ca92d0260dc3c7c67b00e7891775f656cd55cac525cd27409e3b70804233e`.
The learned-checkpoint content digest was
`8262882a08163bae8141e8f3014e5d47d322f547bf4fcbb71410f723007a5c69`.
Focused tests cover singleton and 32-component intervals, canonical component
order, tensor aliasing, variance invalidation, sequential-versus-batched bank
updates, online and frozen outputs, and checkpoint state. The full suite passed:
**729 passed, 4 CUDA skips**.

## Remaining cost

The warmed eight-entity profile is now dominated by sequential identity
association and context-state processing, followed by transport assembly and
mixture retrieval. Those branches decide identity, matching, and future credit.
The forecast banks still keep every new example, so exact retrieval and storage
cost continue to grow with experience; the 240-sample benchmark does not prove
long-run 50 Hz operation. A native tracker or a batched retrieval design would
need its own exact trajectory and checkpoint parity evidence before promotion.
Approximate neighbors, fewer retained examples or mixture components, and fewer
interval-search iterations would change the approved learning or outputs and
were not used. PyTorch's first `vmap` call also has one-time import/setup cost;
the throughput measurements include 40 warmup samples and do not describe that
first-use latency.
